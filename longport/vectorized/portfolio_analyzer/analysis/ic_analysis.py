#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级IC分析模块 - 让IC测试"更像实盘"
实现滚动IC、衰减曲线、成本调整、行业中性等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AdvancedICAnalyzer:
    """高级IC分析器 - 产业级IC分析"""
    
    def __init__(self):
        """初始化高级IC分析器"""
        self.analysis_methods = {
            'rolling_ic': self._calculate_rolling_ic,
            'decay_curve': self._calculate_decay_curve,
            'cost_adjusted_ic': self._calculate_cost_adjusted_ic,
            'industry_neutral': self._calculate_industry_neutral_ic,
            'asymmetric_labels': self._calculate_asymmetric_ic
        }
        
    def comprehensive_ic_analysis(self, 
                                factor_data: pd.DataFrame,
                                price_data: pd.DataFrame,
                                factor_names: List[str],
                                config: Dict = None) -> Dict:
        """
        综合IC分析
        
        Args:
            factor_data: 因子数据 (MultiIndex: symbol, timestamp)
            price_data: 价格数据 (MultiIndex: symbol, timestamp)
            factor_names: 要分析的因子名称列表
            config: 分析配置
            
        Returns:
            综合分析结果
        """
        if config is None:
            config = self._get_default_config()
            
        print(f"🔍 开始综合IC分析，涉及因子: {len(factor_names)}个")
        
        results = {}
        
        for factor_name in factor_names:
            if factor_name not in factor_data.columns:
                continue
                
            print(f"   分析因子: {factor_name}")
            
            factor_result = {
                'basic_ic': self._calculate_basic_ic(factor_data[factor_name], price_data),
                'rolling_ic': self._calculate_rolling_ic(factor_data[factor_name], price_data, config),
                'decay_curve': self._calculate_decay_curve(factor_data[factor_name], price_data, config),
                'cost_adjusted': self._calculate_cost_adjusted_ic(factor_data[factor_name], price_data, config)
            }
            
            results[factor_name] = factor_result
            
        print(f"✅ 综合IC分析完成")
        
        return results
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'rolling_windows': [21, 63, 126],  # 滚动窗口
            'decay_lags': list(range(1, 21)),  # 衰减分析的滞后期
            'transaction_cost': 0.0017,  # 交易成本 (1.5bp slippage + 0.2bp commission)
            'industry_groups': None,  # 行业分组
            'asymmetric_quantiles': [0.3, 0.7]  # 非对称标签分位数
        }
    
    def _calculate_basic_ic(self, 
                          factor_series: pd.Series, 
                          price_data: pd.DataFrame) -> Dict:
        """计算基础IC"""
        try:
            # 计算收益率
            returns = price_data['close'].groupby(level='symbol').pct_change().shift(-1)
            
            # 对齐数据
            common_idx = factor_series.index.intersection(returns.index)
            factor_aligned = factor_series.loc[common_idx]
            returns_aligned = returns.loc[common_idx]
            
            # 去除NaN
            valid_mask = factor_aligned.notna() & returns_aligned.notna()
            clean_factor = factor_aligned[valid_mask]
            clean_returns = returns_aligned[valid_mask]
            
            if len(clean_factor) < 30:
                return {'ic': 0, 'ic_ir': 0, 'sample_size': 0}
                
            # 计算IC
            ic = clean_factor.corr(clean_returns)
            
            # 计算IC IR（需要时间序列）
            ic_series = self._calculate_time_series_ic(clean_factor, clean_returns)
            ic_ir = ic_series.mean() / (ic_series.std() + 1e-8)
            
            return {
                'ic': ic,
                'ic_ir': ic_ir,
                'sample_size': len(clean_factor),
                'ic_series': ic_series
            }
            
        except Exception as e:
            return {'ic': 0, 'ic_ir': 0, 'sample_size': 0, 'error': str(e)}
    
    def _calculate_time_series_ic(self, factor: pd.Series, returns: pd.Series) -> pd.Series:
        """计算时间序列IC（按日期分组）"""
        try:
            # 假设index是MultiIndex (symbol, timestamp)
            if isinstance(factor.index, pd.MultiIndex):
                # 按timestamp计算截面IC
                ic_series = []
                timestamps = factor.index.get_level_values('timestamp').unique()
                
                for ts in timestamps:
                    ts_factor = factor.xs(ts, level='timestamp')
                    ts_returns = returns.xs(ts, level='timestamp')
                    
                    # 对齐数据
                    common_symbols = ts_factor.index.intersection(ts_returns.index)
                    if len(common_symbols) > 5:  # 至少5只股票
                        ts_ic = ts_factor.loc[common_symbols].corr(ts_returns.loc[common_symbols])
                        if not pd.isna(ts_ic):
                            ic_series.append(ts_ic)
                
                return pd.Series(ic_series)
            else:
                # 滚动窗口计算IC
                window = 21
                ic_series = []
                for i in range(window, len(factor)):
                    window_factor = factor.iloc[i-window:i]
                    window_returns = returns.iloc[i-window:i]
                    window_ic = window_factor.corr(window_returns)
                    if not pd.isna(window_ic):
                        ic_series.append(window_ic)
                        
                return pd.Series(ic_series)
                
        except Exception as e:
            return pd.Series([])
    
    def _calculate_rolling_ic(self, 
                            factor_series: pd.Series, 
                            price_data: pd.DataFrame, 
                            config: Dict) -> Dict:
        """计算滚动IC - 观察因子稳定性"""
        try:
            windows = config.get('rolling_windows', [21, 63, 126])
            returns = price_data['close'].groupby(level='symbol').pct_change().shift(-1)
            
            rolling_results = {}
            
            for window in windows:
                rolling_ics = []
                
                # 假设是MultiIndex，按时间滚动
                if isinstance(factor_series.index, pd.MultiIndex):
                    timestamps = factor_series.index.get_level_values('timestamp').unique()
                    timestamps = sorted(timestamps)
                    
                    for i in range(window, len(timestamps)):
                        # 选择窗口内的数据
                        window_timestamps = timestamps[i-window:i]
                        
                        window_factor = factor_series[
                            factor_series.index.get_level_values('timestamp').isin(window_timestamps)
                        ]
                        window_returns = returns[
                            returns.index.get_level_values('timestamp').isin(window_timestamps)
                        ]
                        
                        # 对齐并计算IC
                        common_idx = window_factor.index.intersection(window_returns.index)
                        if len(common_idx) > 20:
                            window_ic = window_factor.loc[common_idx].corr(window_returns.loc[common_idx])
                            if not pd.isna(window_ic):
                                rolling_ics.append({
                                    'timestamp': timestamps[i],
                                    'ic': window_ic,
                                    'sample_size': len(common_idx)
                                })
                
                rolling_results[f'window_{window}'] = rolling_ics
                
            return rolling_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_decay_curve(self, 
                             factor_series: pd.Series, 
                             price_data: pd.DataFrame, 
                             config: Dict) -> Dict:
        """计算衰减曲线 - 找到最佳持有期"""
        try:
            lags = config.get('decay_lags', list(range(1, 21)))
            decay_results = []
            
            for lag in lags:
                # 计算lag期后的收益率
                returns = price_data['close'].groupby(level='symbol').pct_change(lag).shift(-lag)
                
                # 对齐数据
                common_idx = factor_series.index.intersection(returns.index)
                factor_aligned = factor_series.loc[common_idx]
                returns_aligned = returns.loc[common_idx]
                
                # 去除NaN
                valid_mask = factor_aligned.notna() & returns_aligned.notna()
                clean_factor = factor_aligned[valid_mask]
                clean_returns = returns_aligned[valid_mask]
                
                if len(clean_factor) > 20:
                    ic = clean_factor.corr(clean_returns)
                    decay_results.append({
                        'lag': lag,
                        'ic': ic,
                        'abs_ic': abs(ic),
                        'sample_size': len(clean_factor)
                    })
            
            # 找到最佳滞后期
            if decay_results:
                best_lag = max(decay_results, key=lambda x: x['abs_ic'])
                half_life = self._calculate_half_life(decay_results)
                
                return {
                    'decay_curve': decay_results,
                    'best_lag': best_lag,
                    'half_life': half_life
                }
            else:
                return {'decay_curve': [], 'best_lag': None, 'half_life': None}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_half_life(self, decay_results: List[Dict]) -> Optional[int]:
        """计算IC衰减半衰期"""
        try:
            if not decay_results:
                return None
                
            initial_ic = abs(decay_results[0]['ic'])
            half_ic = initial_ic / 2
            
            for result in decay_results:
                if abs(result['ic']) <= half_ic:
                    return result['lag']
                    
            return None
            
        except Exception as e:
            return None
    
    def _calculate_cost_adjusted_ic(self, 
                                  factor_series: pd.Series, 
                                  price_data: pd.DataFrame, 
                                  config: Dict) -> Dict:
        """计算成本调整后的IC"""
        try:
            transaction_cost = config.get('transaction_cost', 0.0017)
            
            # 计算原始收益率
            returns = price_data['close'].groupby(level='symbol').pct_change().shift(-1)
            
            # 估算换手率（因子值变化程度）
            factor_change = factor_series.groupby(level='symbol').diff().abs()
            
            # 标准化因子变化为换手率代理
            turnover_proxy = factor_change / (factor_series.abs() + 1e-8)
            
            # 成本调整：收益率 - 换手率 * 交易成本
            cost_adjusted_returns = returns - turnover_proxy * transaction_cost
            
            # 计算成本调整后的IC
            common_idx = factor_series.index.intersection(cost_adjusted_returns.index)
            factor_aligned = factor_series.loc[common_idx]
            returns_aligned = cost_adjusted_returns.loc[common_idx]
            
            valid_mask = factor_aligned.notna() & returns_aligned.notna()
            clean_factor = factor_aligned[valid_mask]
            clean_returns = returns_aligned[valid_mask]
            
            if len(clean_factor) > 20:
                cost_adjusted_ic = clean_factor.corr(clean_returns)
                original_ic = clean_factor.corr(returns.loc[common_idx][valid_mask])
                
                return {
                    'cost_adjusted_ic': cost_adjusted_ic,
                    'original_ic': original_ic,
                    'ic_degradation': original_ic - cost_adjusted_ic,
                    'avg_turnover_proxy': turnover_proxy.mean()
                }
            else:
                return {'cost_adjusted_ic': 0, 'original_ic': 0}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_industry_neutral_ic(self, 
                                     factor_series: pd.Series, 
                                     price_data: pd.DataFrame, 
                                     industry_data: pd.Series = None) -> Dict:
        """计算行业中性IC"""
        try:
            if industry_data is None:
                # 没有行业数据，返回原始IC
                return self._calculate_basic_ic(factor_series, price_data)
            
            returns = price_data['close'].groupby(level='symbol').pct_change().shift(-1)
            
            # 行业中性化：在每个行业内标准化因子
            def industry_neutralize(group):
                return (group - group.mean()) / (group.std() + 1e-8)
            
            neutral_factor = factor_series.groupby(industry_data).transform(industry_neutralize)
            
            # 计算中性化后的IC
            common_idx = neutral_factor.index.intersection(returns.index)
            factor_aligned = neutral_factor.loc[common_idx]
            returns_aligned = returns.loc[common_idx]
            
            valid_mask = factor_aligned.notna() & returns_aligned.notna()
            clean_factor = factor_aligned[valid_mask]
            clean_returns = returns_aligned[valid_mask]
            
            if len(clean_factor) > 20:
                neutral_ic = clean_factor.corr(clean_returns)
                original_ic = factor_series.loc[common_idx][valid_mask].corr(clean_returns)
                
                return {
                    'industry_neutral_ic': neutral_ic,
                    'original_ic': original_ic,
                    'neutralization_effect': neutral_ic - original_ic
                }
            else:
                return {'industry_neutral_ic': 0, 'original_ic': 0}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_asymmetric_ic(self, 
                               factor_series: pd.Series, 
                               price_data: pd.DataFrame, 
                               config: Dict) -> Dict:
        """计算非对称标签IC（多空分离）"""
        try:
            quantiles = config.get('asymmetric_quantiles', [0.3, 0.7])
            returns = price_data['close'].groupby(level='symbol').pct_change().shift(-1)
            
            # 创建三值标签：-1（做空）, 0（中性）, 1（做多）
            def create_asymmetric_labels(group):
                low_threshold = group.quantile(quantiles[0])
                high_threshold = group.quantile(quantiles[1])
                
                labels = pd.Series(0, index=group.index)
                labels[group >= high_threshold] = 1
                labels[group <= low_threshold] = -1
                
                return labels
            
            # 按时间截面创建标签
            if isinstance(returns.index, pd.MultiIndex):
                asymmetric_labels = returns.groupby(level='timestamp').transform(create_asymmetric_labels)
            else:
                asymmetric_labels = create_asymmetric_labels(returns)
            
            # 计算因子与非对称标签的相关性
            common_idx = factor_series.index.intersection(asymmetric_labels.index)
            factor_aligned = factor_series.loc[common_idx]
            labels_aligned = asymmetric_labels.loc[common_idx]
            
            valid_mask = factor_aligned.notna() & labels_aligned.notna()
            clean_factor = factor_aligned[valid_mask]
            clean_labels = labels_aligned[valid_mask]
            
            if len(clean_factor) > 20:
                asymmetric_ic = clean_factor.corr(clean_labels)
                
                # 分别计算做多和做空的IC
                long_mask = clean_labels == 1
                short_mask = clean_labels == -1
                
                long_ic = clean_factor[long_mask].corr(returns.loc[common_idx][valid_mask][long_mask]) if long_mask.sum() > 10 else 0
                short_ic = clean_factor[short_mask].corr(returns.loc[common_idx][valid_mask][short_mask]) if short_mask.sum() > 10 else 0
                
                return {
                    'asymmetric_ic': asymmetric_ic,
                    'long_ic': long_ic,
                    'short_ic': short_ic,
                    'long_short_balance': long_mask.sum() / short_mask.sum() if short_mask.sum() > 0 else float('inf')
                }
            else:
                return {'asymmetric_ic': 0, 'long_ic': 0, 'short_ic': 0}
                
        except Exception as e:
            return {'error': str(e)}
    
    def generate_ic_analysis_report(self, analysis_results: Dict) -> str:
        """生成IC分析报告"""
        report = ["# 🔍 高级IC分析报告\n"]
        
        for factor_name, factor_results in analysis_results.items():
            report.append(f"## 📊 因子: {factor_name}\n")
            
            # 基础IC
            basic_ic = factor_results.get('basic_ic', {})
            report.append(f"### 基础IC分析")
            report.append(f"- **IC值**: {basic_ic.get('ic', 0):.4f}")
            report.append(f"- **IC IR**: {basic_ic.get('ic_ir', 0):.2f}")
            report.append(f"- **样本量**: {basic_ic.get('sample_size', 0)}")
            
            # 衰减曲线
            decay_curve = factor_results.get('decay_curve', {})
            if 'best_lag' in decay_curve and decay_curve['best_lag']:
                best_lag = decay_curve['best_lag']
                report.append(f"\n### 衰减曲线分析")
                report.append(f"- **最佳持有期**: {best_lag['lag']}期")
                report.append(f"- **最佳IC**: {best_lag['ic']:.4f}")
                report.append(f"- **半衰期**: {decay_curve.get('half_life', 'N/A')}期")
            
            # 成本调整
            cost_adjusted = factor_results.get('cost_adjusted', {})
            if 'cost_adjusted_ic' in cost_adjusted:
                report.append(f"\n### 成本调整分析")
                report.append(f"- **原始IC**: {cost_adjusted.get('original_ic', 0):.4f}")
                report.append(f"- **成本调整IC**: {cost_adjusted.get('cost_adjusted_ic', 0):.4f}")
                report.append(f"- **IC衰减**: {cost_adjusted.get('ic_degradation', 0):.4f}")
            
            report.append("\n---\n")
        
        return "\n".join(report)


def test_advanced_ic_analysis():
    """测试高级IC分析"""
    print("🧪 测试高级IC分析...")
    
    # 创建模拟数据
    np.random.seed(42)
    symbols = ['A', 'B', 'C', 'D', 'E']
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    
    # MultiIndex
    index = pd.MultiIndex.from_product([symbols, dates], names=['symbol', 'timestamp'])
    
    # 模拟因子数据
    factor_data = pd.DataFrame({
        'rsi_14': np.random.normal(50, 15, len(index)),
        'macd_enhanced': np.random.normal(0, 1, len(index))
    }, index=index)
    
    # 模拟价格数据（有一定的因子相关性）
    base_returns = np.random.normal(0, 0.01, len(index))
    factor_effect = factor_data['rsi_14'] * 0.0001  # 添加因子效应
    
    price_data = pd.DataFrame({
        'close': 100 * (1 + base_returns + factor_effect).groupby(level='symbol').cumprod()
    }, index=index)
    
    # 运行高级IC分析
    analyzer = AdvancedICAnalyzer()
    results = analyzer.comprehensive_ic_analysis(
        factor_data, price_data, ['rsi_14', 'macd_enhanced']
    )
    
    # 生成报告
    report = analyzer.generate_ic_analysis_report(results)
    
    print("✅ 测试完成")
    print(f"   分析因子数: {len(results)}")
    print(f"   报告长度: {len(report)} 字符")
    
    return results, report


if __name__ == "__main__":
    results, report = test_advanced_ic_analysis()
    print("\n" + "="*50)
    print(report[:500])  # 显示报告前500字符
