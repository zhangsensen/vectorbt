#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子工程模块 - 把信号做厚、把噪音做薄
实现5个核心方向的因子加工，提升IC质量和稳定性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FactorEngineer:
    """因子工程师 - 专业化因子加工"""
    
    def __init__(self):
        """初始化因子工程师"""
        self.engineering_methods = {
            'time_scale': self._time_scale_fusion,
            'cross_sectional': self._cross_sectional_normalize,
            'nonlinear': self._nonlinear_transform,
            'regime_based': self._regime_based_signal,
            'interaction': self._interaction_terms
        }
        
    def process_factors(self, 
                       factor_data: pd.DataFrame, 
                       methods: List[str] = None,
                       config: Dict = None) -> pd.DataFrame:
        """
        因子工程主函数
        
        Args:
            factor_data: 包含原始因子的DataFrame (MultiIndex: symbol, timestamp)
            methods: 要应用的工程方法列表
            config: 配置参数
            
        Returns:
            工程化后的因子DataFrame
        """
        if methods is None:
            methods = ['cross_sectional', 'nonlinear', 'regime_based']
            
        if config is None:
            config = self._get_default_config()
            
        print(f"🔧 开始因子工程，应用方法: {methods}")
        
        result_df = factor_data.copy()
        
        for method in methods:
            if method in self.engineering_methods:
                print(f"   处理: {method}")
                try:
                    result_df = self.engineering_methods[method](result_df, config)
                except Exception as e:
                    print(f"   ⚠️ {method} 处理失败: {e}")
                    
        # 统计新增因子
        original_factors = set(factor_data.columns)
        new_factors = [col for col in result_df.columns if col not in original_factors]
        
        print(f"✅ 因子工程完成，新增 {len(new_factors)} 个工程化因子")
        
        return result_df
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'rolling_window': 252,  # 横截面标准化窗口
            'rank_transform': True,  # 是否进行排序变换
            'winsorize_pct': 0.05,  # 极值处理百分比
            'regime_thresholds': {'low': 0.3, 'high': 0.7},  # 状态分层阈值
            'interaction_top_factors': 5  # 交互项使用的顶级因子数
        }
    
    def _time_scale_fusion(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """时间尺度拼接 - 多周期信号融合"""
        try:
            # 这里需要多时间框架数据，暂时实现单时间框架内的时间融合
            for col in df.select_dtypes(include=[np.number]).columns:
                if 'rsi' in col.lower():
                    # RSI短期/长期比值
                    rsi_short = df[col].rolling(5).mean()
                    rsi_long = df[col].rolling(20).mean()
                    df[f'{col}_ratio_short_long'] = rsi_short / (rsi_long + 1e-8)
                    
                elif 'macd' in col.lower():
                    # MACD动量强度
                    macd_momentum = df[col].rolling(10).std()
                    df[f'{col}_momentum_strength'] = macd_momentum
                    
                elif 'atr' in col.lower():
                    # ATR波动率状态
                    atr_ma = df[col].rolling(20).mean()
                    df[f'{col}_relative_to_ma'] = df[col] / (atr_ma + 1e-8)
                    
        except Exception as e:
            print(f"时间尺度融合失败: {e}")
            
        return df
    
    def _cross_sectional_normalize(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """横截面标准化 - 每期只选相对最强"""
        try:
            # 确保是MultiIndex (symbol, timestamp)
            if not isinstance(df.index, pd.MultiIndex):
                return df
                
            # 获取数值型列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 按时间截面标准化
                def cross_sectional_zscore(group):
                    return (group - group.mean()) / (group.std() + 1e-8)
                
                # 按timestamp分组，对每个截面进行标准化
                df[f'{col}_cs_zscore'] = df.groupby(level='timestamp')[col].transform(cross_sectional_zscore)
                
                # 截面排序
                df[f'{col}_cs_rank'] = df.groupby(level='timestamp')[col].rank(pct=True)
                
        except Exception as e:
            print(f"横截面标准化失败: {e}")
            
        return df
    
    def _nonlinear_transform(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """非线性变换 - 压缩极端值，降低噪音"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 跳过已经处理过的列
                if any(suffix in col for suffix in ['_rank', '_zscore', '_tanh', '_squared']):
                    continue
                    
                values = df[col].dropna()
                if len(values) == 0:
                    continue
                    
                # 1. Rank变换 + 平方
                rank_values = values.rank(pct=True)
                df[f'{col}_rank_squared'] = rank_values ** 2
                
                # 2. tanh变换（压缩极端值）
                normalized_values = (values - values.mean()) / (values.std() + 1e-8)
                df[f'{col}_tanh'] = np.tanh(normalized_values)
                
                # 3. 分位数映射
                df[f'{col}_quantile'] = pd.qcut(values, q=10, labels=False, duplicates='drop')
                
                # 4. Winsorize处理
                winsorize_pct = config.get('winsorize_pct', 0.05)
                lower_bound = values.quantile(winsorize_pct)
                upper_bound = values.quantile(1 - winsorize_pct)
                df[f'{col}_winsorized'] = values.clip(lower_bound, upper_bound)
                
        except Exception as e:
            print(f"非线性变换失败: {e}")
            
        return df
    
    def _regime_based_signal(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """状态分层信号 - 降低换手率"""
        try:
            thresholds = config.get('regime_thresholds', {'low': 0.3, 'high': 0.7})
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 跳过已经处理过的列
                if any(suffix in col for suffix in ['_regime', '_signal', '_threshold']):
                    continue
                    
                values = df[col].dropna()
                if len(values) == 0:
                    continue
                    
                # 计算分位数阈值
                low_threshold = values.quantile(thresholds['low'])
                high_threshold = values.quantile(thresholds['high'])
                
                # 三值信号：-1, 0, 1
                def regime_signal(x):
                    if pd.isna(x):
                        return 0
                    elif x >= high_threshold:
                        return 1
                    elif x <= low_threshold:
                        return -1
                    else:
                        return 0
                
                df[f'{col}_regime_signal'] = df[col].apply(regime_signal)
                
                # 状态持续性（减少频繁切换）
                regime_signal_col = f'{col}_regime_signal'
                if regime_signal_col in df.columns:
                    # 添加滞后性，避免频繁切换
                    df[f'{col}_regime_smooth'] = df[regime_signal_col].rolling(3, center=True).mean()
                
        except Exception as e:
            print(f"状态分层信号失败: {e}")
            
        return df
    
    def _interaction_terms(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """交互项 - 多因子组合信号"""
        try:
            # 获取主要因子
            main_factors = []
            for col in df.columns:
                if any(factor in col.lower() for factor in ['rsi', 'macd', 'atr', 'vwap']):
                    if not any(suffix in col for suffix in ['_cs_', '_rank', '_regime', '_tanh']):
                        main_factors.append(col)
            
            # 限制交互项数量
            top_n = config.get('interaction_top_factors', 5)
            main_factors = main_factors[:top_n]
            
            # 创建交互项
            for i, factor1 in enumerate(main_factors):
                for factor2 in main_factors[i+1:]:
                    try:
                        # 乘积交互
                        df[f'{factor1}_x_{factor2}'] = df[factor1] * df[factor2]
                        
                        # 比值交互
                        df[f'{factor1}_div_{factor2}'] = df[factor1] / (df[factor2] + 1e-8)
                        
                        # 条件交互（factor1>0时的factor2）
                        condition = df[factor1] > df[factor1].median()
                        df[f'{factor2}_when_{factor1}_high'] = df[factor2] * condition.astype(int)
                        
                    except Exception as e:
                        print(f"交互项 {factor1} x {factor2} 创建失败: {e}")
            
        except Exception as e:
            print(f"交互项计算失败: {e}")
            
        return df
    
    def calculate_factor_quality_score(self, 
                                     factor_data: pd.DataFrame,
                                     returns: pd.Series,
                                     factor_name: str) -> Dict:
        """计算因子质量得分"""
        try:
            factor_values = factor_data[factor_name].dropna()
            aligned_returns = returns.reindex(factor_values.index).dropna()
            
            # 对齐数据
            common_idx = factor_values.index.intersection(aligned_returns.index)
            if len(common_idx) < 30:
                return {'quality_score': 0, 'reason': 'insufficient_data'}
                
            factor_aligned = factor_values.loc[common_idx]
            returns_aligned = aligned_returns.loc[common_idx]
            
            # 计算各种质量指标
            ic = factor_aligned.corr(returns_aligned)
            ic_std = pd.Series([factor_aligned.corr(returns_aligned)]).rolling(20).std().iloc[-1]
            ir = ic / (ic_std + 1e-8)
            
            # 单调性检验
            factor_quantiles = pd.qcut(factor_aligned, q=5, labels=False, duplicates='drop')
            quantile_returns = returns_aligned.groupby(factor_quantiles).mean()
            monotonicity = stats.kendalltau(range(len(quantile_returns)), quantile_returns)[0]
            
            # 稳定性检验（滚动IC标准差）
            rolling_ic = pd.Series(index=common_idx, dtype=float)
            for i in range(60, len(common_idx)):
                window_factor = factor_aligned.iloc[i-60:i]
                window_returns = returns_aligned.iloc[i-60:i]
                rolling_ic.iloc[i] = window_factor.corr(window_returns)
            
            stability = 1 - (rolling_ic.std() / (np.abs(rolling_ic.mean()) + 1e-8))
            
            # 综合质量得分
            quality_score = (
                np.abs(ic) * 0.4 +  # IC重要性
                np.abs(monotonicity) * 0.3 +  # 单调性
                stability * 0.3  # 稳定性
            )
            
            return {
                'quality_score': quality_score,
                'ic': ic,
                'ir': ir,
                'monotonicity': monotonicity,
                'stability': stability,
                'sample_size': len(common_idx)
            }
            
        except Exception as e:
            return {'quality_score': 0, 'reason': str(e)}


def test_factor_engineering():
    """测试因子工程"""
    print("🧪 测试因子工程...")
    
    # 创建模拟数据
    np.random.seed(42)
    symbols = ['A', 'B', 'C']
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    
    # 创建MultiIndex DataFrame
    index = pd.MultiIndex.from_product([symbols, dates], names=['symbol', 'timestamp'])
    
    data = pd.DataFrame({
        'rsi_14': np.random.normal(50, 15, len(index)),
        'macd_enhanced': np.random.normal(0, 1, len(index)),
        'atrp': np.abs(np.random.normal(0.02, 0.01, len(index))),
        'vwap_deviation': np.random.normal(0, 0.005, len(index))
    }, index=index)
    
    # 添加一些真实的模式
    data['rsi_14'] = np.clip(data['rsi_14'], 0, 100)
    
    # 应用因子工程
    engineer = FactorEngineer()
    result = engineer.process_factors(data)
    
    print(f"✅ 测试完成:")
    print(f"   输入因子: {len(data.columns)}个")
    print(f"   输出因子: {len(result.columns)}个")
    print(f"   新增因子: {len(result.columns) - len(data.columns)}个")
    
    return result


if __name__ == "__main__":
    test_factor_engineering()
