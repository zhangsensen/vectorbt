#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终工作版VectorBT系统
简化逻辑，确保IC计算正常工作
基于已验证的单股票调试结果
"""

import os
import sys
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import psutil

# VectorBT和技术指标
try:
    import vectorbt as vbt
    import talib
    print("✅ VectorBT和TA-Lib加载成功")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# 导入自定义模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors.factor_pool import AdvancedFactorPool
from utils.dtype_fixer import CategoricalDtypeFixer
from strategies.cta_evaluator import CTAEvaluator

# 统计学库
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac
    print("✅ Statsmodels加载成功")
except ImportError:
    print("⚠️ Statsmodels未安装，将使用简化版IC_IR计算")

warnings.filterwarnings('ignore')

class FinalWorkingVectorBT:
    """
    最终工作版VectorBT系统
    基于调试结果，确保IC计算正常
    """
    
    def __init__(self, data_dir: str = "../vectorbt_workspace/data", capital: float = 300000):
        self.data_dir = data_dir
        self.capital = capital
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🔥 最终工作配置 - 基于实际测试结果
        self.working_config = {
            'test_timeframes': ['1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h', '4h', '1d'],  # 🔥 全时间框架对比测试
            'max_symbols': 54,  # 🔥 全股票对比测试: 54只股票
            'evaluation_mode': 'cta',  # 🔥 新增: CTA回测模式 vs 'ic'模式
            
            # 🔥 基于调试结果的宽松阈值
            'min_ic_threshold': 0.005,   # 基于单股票测试，IC=0.02-0.11是合理的
            'min_ir_threshold': 0.01,    # 非常宽松的IR要求
            'min_sample_size': 10,       # 最小样本量
            'min_supporting_stocks': 2,  # 至少2只股票支持
            
            # 启用功能
            'full_factor_pool': True,
            'debug_mode': True
        }
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.factor_pool = AdvancedFactorPool()
        self.categorical_fixer = CategoricalDtypeFixer()
        
        # 获取测试股票
        self.test_symbols = self._get_test_symbols()
        self.logger.info(f"✅ 测试股票: {self.test_symbols}")
        
    def _setup_logging(self):
        """设置简单日志"""
        log_dir = f"logs/final_working_{self.timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('FinalWorking')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            f"{log_dir}/final_working.log", 
            encoding='utf-8'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        
        # 格式器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _get_test_symbols(self) -> List[str]:
        """获取测试股票"""
        # 优先选择有代表性的股票
        priority_symbols = ['0700.HK', '0005.HK', '0388.HK', '1211.HK', '0981.HK']
        
        available_symbols = []
        
        # 检查1d数据确保股票可用
        d1_dir = os.path.join(self.data_dir, '1d')
        if os.path.exists(d1_dir):
            for symbol in priority_symbols:
                file_path = os.path.join(d1_dir, f'{symbol}.parquet')
                if os.path.exists(file_path):
                    available_symbols.append(symbol)
        
        # 补充到指定数量
        if len(available_symbols) < self.working_config['max_symbols']:
            for file in os.listdir(d1_dir):
                if file.endswith('.parquet'):
                    symbol = file.replace('.parquet', '')
                    if symbol not in available_symbols:
                        available_symbols.append(symbol)
                        if len(available_symbols) >= self.working_config['max_symbols']:
                            break
        
        return available_symbols[:self.working_config['max_symbols']]
    
    def _load_symbol_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加载单个股票数据"""
        try:
            file_path = os.path.join(self.data_dir, timeframe, f'{symbol}.parquet')
            if not os.path.exists(file_path):
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            
            # 标准化索引和列
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 列名标准化
            column_mapping = {
                'Close': 'close', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Volume': 'volume', 'Turnover': 'turnover'
            }
            df = df.rename(columns=column_mapping)
            
            # 确保基础列存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) >= 4:
                return df[available_cols].dropna()
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.debug(f"加载{symbol}-{timeframe}失败: {e}")
            return pd.DataFrame()
    
    def _safe_divide(self, numerator, denominator, fill_value=0):
        """安全除法，避免除零和NaN"""
        # 处理Series和numpy数组
        if hasattr(numerator, 'values'):
            num_vals = numerator.values
        else:
            num_vals = np.asarray(numerator)
            
        if hasattr(denominator, 'values'):
            den_vals = denominator.values
        else:
            den_vals = np.asarray(denominator)
        
        # 避免除零
        den_vals = np.where(np.abs(den_vals) < 1e-10, 1e-10, den_vals)
        
        result = num_vals / den_vals
        result = np.where(np.isinf(result), fill_value, result)
        result = np.where(np.isnan(result), fill_value, result)

        # 返回pandas Series如果输入是Series
        if hasattr(numerator, 'index'):
            return pd.Series(result, index=numerator.index)
        else:
            return result

    def _select_sharpe_column(self, df: pd.DataFrame) -> Optional[str]:
        """选择最佳夏普率列，优先使用成本现实化后的夏普率"""
        if df is None or df.empty:
            return None

        for candidate in ('sharpe_cost', 'sharpe_mean', 'sharpe'):
            if candidate in df.columns:
                return candidate

        for col in df.columns:
            if isinstance(col, str) and 'sharpe' in col.lower():
                return col

        return None
    
    def _align_multi_stock_data(self, symbol_data: Dict[str, pd.DataFrame], symbol_factors: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, pd.DatetimeIndex]:
        """🔥 核心修复：多股票数据对齐到共同时间范围"""
        if not symbol_data:
            return {}, {}, pd.DatetimeIndex([])
        
        # 1. 找到所有股票的共同时间范围
        all_indices = [df.index for df in symbol_data.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        
        if len(common_index) == 0:
            self.logger.warning("❌ 无共同时间范围，跳过对齐")
            return {}, {}, pd.DatetimeIndex([])
        
        # 2. 防御式日志：打印共同样本覆盖  
        start_dt = pd.to_datetime(common_index[0])
        end_dt = pd.to_datetime(common_index[-1])
        self.logger.debug("🔍 共同样本区间: %s 至 %s (共 %d 根K线)", 
                         start_dt.strftime('%Y-%m-%d %H:%M:%S'), 
                         end_dt.strftime('%Y-%m-%d %H:%M:%S'), 
                         len(common_index))
        
        # 3. 截断所有数据到共同范围
        aligned_data = {}
        aligned_factors = {}
        
        for symbol in symbol_data.keys():
            if symbol in symbol_factors:
                # 对齐价格数据
                aligned_data[symbol] = symbol_data[symbol].reindex(common_index)
                # 对齐因子数据
                aligned_factors[symbol] = symbol_factors[symbol].reindex(common_index)
                
                self.logger.debug(f"    ✅ {symbol}: 对齐到{len(common_index)}条数据")
        
        return aligned_data, aligned_factors, common_index
    
    def _filter_low_sample_factors(self, symbol_factors: Dict[str, pd.DataFrame], common_index: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """🔥 核心修复：过滤低样本因子"""
        if len(common_index) == 0:
            return symbol_factors
        
        min_sample_threshold = int(len(common_index) * 0.8)  # 80%阈值
        invalid_factors = {}
        filtered_factors = {}
        
        for symbol, factors_df in symbol_factors.items():
            valid_factors = {}
            
            for factor_name in factors_df.columns:
                if factor_name in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'timestamp']:
                    continue
                
                valid_count = factors_df[factor_name].notna().sum()
                
                if valid_count >= min_sample_threshold:
                    valid_factors[factor_name] = factors_df[factor_name]
                else:
                    # 记录无效因子
                    if factor_name not in invalid_factors:
                        invalid_factors[factor_name] = []
                    invalid_factors[factor_name].append(symbol)
                    self.logger.debug(f"    ⚠️ {symbol}-{factor_name}: 样本不足({valid_count}/{len(common_index)})")
            
            if valid_factors:
                # 保留价格列
                price_cols = {col: factors_df[col] for col in ['open', 'high', 'low', 'close', 'volume'] if col in factors_df.columns}
                filtered_factors[symbol] = pd.DataFrame({**price_cols, **valid_factors}, index=factors_df.index)
            
        # 防御式日志：打印无效因子清单
        if invalid_factors:
            self.logger.warning(f"⚠️ 以下因子因样本不足被剔除: {dict(invalid_factors)}")
        
        return filtered_factors
    
    def _calculate_newey_west_ic_ir(self, ic_series: pd.Series, lags: int = 3) -> float:
        """🔥 核心修复：Newey-West自相关修正的IC_IR"""
        try:
            if len(ic_series) < 10:  # 样本太少
                return 0.0
            
            # 移除NaN
            clean_ic = ic_series.dropna()
            if len(clean_ic) < 10:
                return 0.0
            
            mean_ic = clean_ic.mean()
            
            # 使用statsmodels计算Newey-West标准误
            try:
                from statsmodels.regression.linear_model import OLS
                from statsmodels.stats.sandwich_covariance import cov_hac
                
                # 构造简单线性模型 (IC对常数项回归)
                X = np.ones((len(clean_ic), 1))
                y = clean_ic.values
                
                model = OLS(y, X).fit()
                nw_cov = cov_hac(model, nlags=lags)
                nw_std = np.sqrt(nw_cov[0, 0])
                
                ic_ir_adj = mean_ic / nw_std if nw_std > 0 else 0.0
                
                self.logger.debug(f"      Newey-West IC_IR: {ic_ir_adj:.4f} (原始: {mean_ic/clean_ic.std():.4f})")
                return ic_ir_adj
            except ImportError:
                # 降级到普通IC_IR
                return mean_ic / clean_ic.std() if clean_ic.std() > 0 else 0.0
                
        except Exception as e:
            self.logger.debug(f"      Newey-West计算失败: {e}, 使用原始IC_IR")
            return ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0.0
    
    def _calculate_fixed_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算修复版因子，解决NaN问题"""
        print(f"🔧 计算修复版因子: {df.shape}")
        
        # 确保数值类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        high, low, close, volume = df.high, df.low, df.close, df.volume
        
        # 使用原有的因子池，但添加关键修复
        df = self.factor_pool.calculate_all_factors(df)
        
        # 修复keltner_position
        try:
            import talib as ta  # 确保本地导入
            atr = ta.ATR(high.values, low.values, close.values, timeperiod=14)
            atr_series = pd.Series(atr, index=df.index)
            keltner_ma = close.rolling(20, min_periods=1).mean()
            keltner_atr = atr_series.rolling(20, min_periods=1).mean()
            keltner_upper = keltner_ma + 2 * keltner_atr
            keltner_lower = keltner_ma - 2 * keltner_atr
            
            # 安全计算位置
            band_width = keltner_upper - keltner_lower
            position_numerator = close - keltner_lower
            
            keltner_position = self._safe_divide(position_numerator, band_width, fill_value=0.5)
            keltner_position = np.clip(keltner_position, -1, 2)
            df['keltner_position'] = keltner_position
            
            valid_count = keltner_position.notna().sum()
            print(f"  ✅ Keltner Position修复: {valid_count}/{len(keltner_position)} 有效值")
            
        except Exception as e:
            print(f"  ❌ Keltner修复失败: {e}")
            df['keltner_position'] = 0.5
        
        # 修复VWAP偏离度
        try:
            typical_price = (high + low + close) / 3
            window = min(20, len(df) // 2) if len(df) >= 40 else max(1, len(df) // 4)
            
            vwap = (typical_price * volume).rolling(window, min_periods=1).sum() / volume.rolling(window, min_periods=1).sum()
            vwap_deviation = self._safe_divide(close - vwap, vwap, fill_value=0.0)
            df['vwap_deviation'] = vwap_deviation
            
            valid_count = vwap_deviation.notna().sum()
            print(f"  ✅ VWAP偏离度修复: {valid_count}/{len(vwap_deviation)} 有效值")
            
        except Exception as e:
            print(f"  ❌ VWAP修复失败: {e}")
            df['vwap_deviation'] = 0.0
        
        # 修复动态ranking
        try:
            # 动态确定窗口大小
            ranking_window = 288 * 2  # ✅补丁5: 2天窗口(5m数据)
            print(f"  🔧 动态ranking窗口: {ranking_window}")
            
            factors_to_rank = ['rsi_14', 'macd_enhanced', 'atrp', 'vwap_deviation']
            for factor in factors_to_rank:
                if factor in df.columns and not df[factor].isna().all():
                    try:
                        min_periods = max(1, ranking_window // 4)
                        factor_rank = df[factor].rolling(ranking_window, min_periods=min_periods).rank(pct=True)
                        df[f'{factor}_rank'] = factor_rank
                        valid_count = factor_rank.notna().sum()
                        print(f"    ✅ {factor}_rank: {valid_count}/{len(factor_rank)} 有效值")
                    except Exception as rank_error:
                        print(f"    ❌ {factor}_rank失败: {rank_error}")
                        df[f'{factor}_rank'] = 0.5
                else:
                    df[f'{factor}_rank'] = 0.5
                    
        except Exception as e:
            print(f"  ❌ Ranking修复失败: {e}")
        
        return df
    
    def _calculate_symbol_factors(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """计算单个股票的因子 - NaN问题修复版"""
        if data.empty:
            return pd.DataFrame()
        
        try:
            # 计算因子 (使用修复版方法)
            factors_df = self._calculate_fixed_factors(data.copy())
            
            # 只保留因子列
            base_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            factor_cols = [col for col in factors_df.columns if col not in base_cols]
            
            if factor_cols:
                factor_only_df = factors_df[factor_cols].copy()
                
                # Categorical修复
                factor_only_df, _ = self.categorical_fixer.comprehensive_fix(factor_only_df)
                
                return factor_only_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.debug(f"{symbol} 因子计算失败: {e}")
            return pd.DataFrame()
    
    def _calculate_symbol_ic(self, symbol: str, factors_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict[str, Any]:
        """🔥 计算单个股票的IC - 基于调试验证的逻辑"""
        self.logger.debug(f"    🔍 开始计算{symbol}的IC")
        self.logger.debug(f"      因子数据: {factors_df.shape}, 价格数据: {price_df.shape}")
        
        if factors_df.empty or price_df.empty:
            self.logger.debug(f"      ❌ {symbol}: 数据为空，跳过IC计算")
            return {}
        
        # 计算未来收益率
        returns = price_df['close'].pct_change(1).shift(-1)
        self.logger.debug(f"      收益率计算: 非空数量={returns.notna().sum()}, 总数={len(returns)}")
        
        # 确保数据长度一致
        min_len = min(len(factors_df), len(returns))
        if min_len < 20:  # 至少20个样本
            return {}
        
        # 截取相同长度
        aligned_factors = factors_df.iloc[:min_len]
        aligned_returns = returns.iloc[:min_len]
        
        # 移除缺失值
        valid_returns = aligned_returns.dropna()
        if len(valid_returns) < self.working_config['min_sample_size']:
            return {}
        
        symbol_ic_results = {}
        
        # 🔥 为每个因子计算IC
        self.logger.debug(f"      待计算因子: {len(aligned_factors.columns)}个")
        for i, factor_name in enumerate(aligned_factors.columns):
            try:
                factor_values = aligned_factors[factor_name].iloc[:len(valid_returns)]
                
                # 确保factor和return对齐
                valid_idx = factor_values.notna() & valid_returns.notna()
                
                if valid_idx.sum() < self.working_config['min_sample_size']:
                    self.logger.debug(f"        ⚠️ {factor_name}: 有效样本不足({valid_idx.sum()})")
                    continue
                
                clean_factor = factor_values[valid_idx]
                clean_returns = valid_returns[valid_idx]
                
                # 计算IC (基于调试验证的方法)
                ic_spearman = clean_factor.corr(clean_returns, method='spearman')
                ic_pearson = clean_factor.corr(clean_returns, method='pearson')
                self.logger.debug(f"        📊 {factor_name}: Spearman={ic_spearman:.4f}, Pearson={ic_pearson:.4f}, 样本={len(clean_factor)}")
                
                # 选择有效的IC
                ic = ic_spearman if not pd.isna(ic_spearman) else ic_pearson
                
                if not pd.isna(ic):
                    symbol_ic_results[factor_name] = {
                        'ic': float(ic),
                        'sample_size': len(clean_factor),
                        'method': 'spearman' if not pd.isna(ic_spearman) else 'pearson'
                    }
                    
                    if self.working_config['debug_mode']:
                        self.logger.debug(f"  {symbol}-{factor_name}: IC={ic:.4f}, 样本={len(clean_factor)}")
                
            except Exception as e:
                self.logger.debug(f"  {symbol}-{factor_name} IC计算失败: {e}")
                continue
        
        return symbol_ic_results
    
    def _aggregate_multi_stock_ic(self, all_symbol_ics: Dict[str, Dict]) -> Dict[str, Any]:
        """🔥 汇总多股票IC结果"""
        self.logger.debug(f"  🎯 开始聚合{len(all_symbol_ics)}只股票的IC结果")
        factor_ic_summary = {}
        
        # 按因子名汇总
        all_factor_names = set()
        for symbol_ics in all_symbol_ics.values():
            all_factor_names.update(symbol_ics.keys())
        
        self.logger.debug(f"    发现{len(all_factor_names)}个不同因子")
        
        for factor_name in all_factor_names:
            ic_values = []
            sample_sizes = []
            supporting_symbols = []
            
            # 收集该因子在各股票的IC
            for symbol, symbol_ics in all_symbol_ics.items():
                if factor_name in symbol_ics:
                    ic_data = symbol_ics[factor_name]
                    ic_values.append(ic_data['ic'])
                    sample_sizes.append(ic_data['sample_size'])
                    supporting_symbols.append(symbol)
            
            # 检查是否有足够支持
            if len(ic_values) < self.working_config['min_supporting_stocks']:
                continue
            
            # 汇总统计
            mean_ic = np.mean(ic_values)
            std_ic = np.std(ic_values) if len(ic_values) > 1 else 0
            ic_ir = mean_ic / std_ic if std_ic > 1e-6 else 0
            
            # 🔥 新增：Newey-West修正IC_IR
            ic_series = pd.Series(ic_values)
            ic_ir_adj = self._calculate_newey_west_ic_ir(ic_series)
            self.logger.debug(f"    📊 {factor_name}: 原始IC_IR={ic_ir:.4f}, 修正IC_IR={ic_ir_adj:.4f}")
            
            positive_ic_count = sum(1 for ic in ic_values if ic > 0)
            positive_ic_ratio = positive_ic_count / len(ic_values)
            
            total_samples = sum(sample_sizes)
            
            # 🔥 升级：使用修正后的IC_IR进行筛选，提高质量标准
            if (abs(mean_ic) >= self.working_config['min_ic_threshold'] and 
                abs(ic_ir_adj) >= 0.05):  # 修正后IC_IR最小阈值
                
                factor_ic_summary[factor_name] = {
                    'ic': float(mean_ic),
                    'ic_ir': float(ic_ir),
                    'ic_ir_adj': float(ic_ir_adj),  # 新增修正IC_IR
                    'ic_std': float(std_ic),
                    'positive_ic_ratio': float(positive_ic_ratio),
                    'total_sample_size': int(total_samples),
                    'supporting_stocks': len(supporting_symbols),
                    'stock_list': supporting_symbols,
                    'ic_values': ic_values
                }
            else:
                self.logger.debug(f"    ❌ {factor_name}: 未通过质量筛选 (IC={mean_ic:.4f}, IC_IR_adj={ic_ir_adj:.4f})")
        
        return factor_ic_summary
    
    def _analyze_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """分析单个时间框架"""
        self.logger.info(f"🔍 分析{timeframe}时间框架...")
        self.logger.debug(f"  目标股票数: {len(self.test_symbols)}")
        
        # 1. 加载所有股票数据
        symbol_data = {}
        symbol_factors = {}
        
        for symbol in self.test_symbols:
            data = self._load_symbol_data(symbol, timeframe)
            if not data.empty:
                symbol_data[symbol] = data
                self.logger.debug(f"    ✅ {symbol}: 加载{len(data)}条数据")
                
                # 计算因子
                factors = self._calculate_symbol_factors(symbol, data)
                if not factors.empty:
                    symbol_factors[symbol] = factors
                    self.logger.debug(f"    📊 {symbol}: 计算{len(factors.columns)}个因子")
                else:
                    self.logger.debug(f"    ❌ {symbol}: 因子计算失败")
            else:
                self.logger.debug(f"    ❌ {symbol}: 数据加载失败")
        
        if not symbol_factors:
            self.logger.warning(f"❌ {timeframe} 无有效因子数据")
            return {}
        
        self.logger.info(f"  ✅ {len(symbol_factors)}只股票有效，平均{np.mean([len(f.columns) for f in symbol_factors.values()]):.0f}个因子")
        
        # 🔥 新增：数据对齐和质量过滤
        aligned_data, aligned_factors, common_index = self._align_multi_stock_data(symbol_data, symbol_factors)
        if not aligned_data:
            self.logger.warning(f"❌ {timeframe} 数据对齐失败")
            return {}
        
        # 🔥 新增：过滤低样本因子
        filtered_factors = self._filter_low_sample_factors(aligned_factors, common_index)
        if not filtered_factors:
            self.logger.warning(f"❌ {timeframe} 因子过滤后无剩余")
            return {}
        
        self.logger.info(f"  ✅ 对齐后{len(filtered_factors)}只股票，共同样本{len(common_index)}条")
        
        # 2. 计算各股票IC (使用对齐后的数据)
        all_symbol_ics = {}
        self.logger.debug(f"  🔢 开始计算{len(filtered_factors)}只股票的IC")
        
        for symbol in filtered_factors.keys():
            if symbol in aligned_data:
                symbol_ic = self._calculate_symbol_ic(symbol, filtered_factors[symbol], aligned_data[symbol])
                if symbol_ic:
                    all_symbol_ics[symbol] = symbol_ic
                    self.logger.debug(f"    ✅ {symbol}: 计算{len(symbol_ic)}个因子IC")
                else:
                    self.logger.debug(f"    ❌ {symbol}: IC计算失败")
        
        if not all_symbol_ics:
            self.logger.warning(f"❌ {timeframe} 无有效IC数据")
            return {}
        
        self.logger.info(f"  ✅ {len(all_symbol_ics)}只股票有IC数据")
        
        # 3. 汇总IC结果
        ic_summary = self._aggregate_multi_stock_ic(all_symbol_ics)
        
        self.logger.info(f"  ✅ {timeframe} 最终有效因子: {len(ic_summary)}个")
        
        return ic_summary
    
    def _run_cta_analysis(self, timeframe: str) -> Dict[str, Any]:
        """🔥 CTA回测模式分析单个时间框架"""
        self.logger.info(f"🎯 CTA回测分析{timeframe}时间框架...")
        self.logger.debug(f"  目标股票数: {len(self.test_symbols)}")
        
        # 1. 加载所有股票数据和因子
        symbol_data = {}
        symbol_factors = {}
        
        for symbol in self.test_symbols:
            data = self._load_symbol_data(symbol, timeframe)
            if not data.empty:
                symbol_data[symbol] = data
                self.logger.debug(f"    ✅ {symbol}: 加载{len(data)}条数据")
                
                # 计算因子
                factors = self._calculate_symbol_factors(symbol, data)
                if not factors.empty:
                    symbol_factors[symbol] = factors
                    self.logger.debug(f"    📊 {symbol}: 计算{len(factors.columns)}个因子")
        
        if not symbol_factors:
            self.logger.warning(f"❌ {timeframe} 无有效因子数据")
            return {}
        
        self.logger.info(f"  ✅ {len(symbol_factors)}只股票有效，平均{np.mean([len(f.columns) for f in symbol_factors.values()]):.0f}个因子")
        
        # ✅修复5: 添加内存监控
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"    💾 数据加载后内存使用: {memory_mb:.1f} MB")
        except:
            pass
        
        # 2. 🔥 使用原版cta_eval_v3修复回测问题
        from strategies.cta_eval_v3 import CTAEvaluatorV3
        cta_evaluator = CTAEvaluatorV3(
            look_ahead=6,                 # ✅补丁1: 拿6根K线(30分钟)让利润奔跑
            entry_percentile=0.90,        # ✅补丁2: 极端信号 90%分位
            exit_percentile=0.10,         # ✅补丁2: 极端信号 10%分位
            sl_stop=0.02,
            tp_stop=0.03,
            direction='both',
            slippage=0.001,               # ✅补丁3: 港股差异化 0.1%滑点
            fees=0.0005,                  # ✅补丁3: 港股差异化 0.05%手续费(合计0.15%)
            min_trades=30                 # ✅补丁4: 先用30次看效果，后续可调整
        )
        
        # 获取所有因子名称
        all_factors = set()
        for factors_df in symbol_factors.values():
            all_factors.update(factors_df.columns)
        factor_names = list(all_factors)
        
        self.logger.info(f"  🔢 开始CTA评估{len(symbol_factors)}只股票 × {len(factor_names)}个因子")
        
        # 3. 批量CTA评估 (传入时间框架)
        cta_results = cta_evaluator.batch_evaluate(
            symbols=list(symbol_factors.keys()),
            factor_data=symbol_factors,
            price_data=symbol_data,
            factor_names=factor_names,
            timeframe=timeframe  # 🔥 传入时间框架以便正确计算窗口
        )
        
        if cta_results.empty:
            self.logger.warning(f"❌ {timeframe} CTA评估无结果")
            return {}
        
        # ✅修复5: 评估后内存监控
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"    💾 CTA评估后内存使用: {memory_mb:.1f} MB")
        except:
            pass
        
        # 4. 因子排名 (修复版评估器有内置过滤)
        # ✅修复2: 验证CTA评估结果有效性
        self.logger.debug(f"    📊 CTA评估结果验证:")
        self.logger.debug(f"      结果形状: {cta_results.shape}")
        self.logger.debug(f"      列名: {list(cta_results.columns)}")
        if 'sharpe' in cta_results.columns:
            sharpe_stats = cta_results['sharpe'].describe()
            self.logger.debug(f"      夏普率统计: 均值={sharpe_stats['mean']:.4f}, 最大值={sharpe_stats['max']:.4f}, 中位数={sharpe_stats['50%']:.4f}")
            valid_sharpe_count = (cta_results['sharpe'] > 0.01).sum()
            self.logger.debug(f"      有效夏普率(>0.01): {valid_sharpe_count}/{len(cta_results)} ({valid_sharpe_count/len(cta_results)*100:.1f}%)")
        if 'trades' in cta_results.columns:
            trade_stats = cta_results['trades'].describe()
            self.logger.debug(f"      交易次数统计: 均值={trade_stats['mean']:.1f}, 最大值={trade_stats['max']:.0f}, 中位数={trade_stats['50%']:.0f}")
        
        factor_ranking = cta_evaluator.rank_factors(
            cta_results,
            rank_by='sharpe'  # 使用修复版的内置过滤条件
        )

        sharpe_col = self._select_sharpe_column(factor_ranking)

        # 5. 统计有效因子
        if factor_ranking.empty:
            valid_factors = pd.DataFrame()
        elif sharpe_col:
            valid_factors = factor_ranking[factor_ranking[sharpe_col] >= 0.05]
        else:
            valid_factors = factor_ranking.head(10)  # 取前10个因子作为有效因子
        
        self.logger.info(f"  ✅ {timeframe} 发现{len(valid_factors)}个优质因子 (夏普≥0.05，修复阈值)")
        
        # 🔍 上线前最后体检：Top5因子人工抽查
        if not factor_ranking.empty and len(factor_ranking) >= 1:
            top5 = factor_ranking.head(5)
            trades_col = 'trades_sum' if 'trades_sum' in top5.columns else 'trades' if 'trades' in top5.columns else None
            winrate_col = 'win_rate_mean' if 'win_rate_mean' in top5.columns else 'win_rate' if 'win_rate' in top5.columns else None

            display_cols = ['factor']
            for col in (sharpe_col, trades_col, winrate_col):
                if col and col in top5.columns and col not in display_cols:
                    display_cols.append(col)
            
            self.logger.info(f"\n🔍 {timeframe} Top5因子人工抽查:")
            self.logger.info("\n" + top5[display_cols].to_string(index=False))
            
            # 异常值警告
            if sharpe_col and sharpe_col in top5.columns:
                max_sharpe = top5[sharpe_col].max()
                min_sharpe = top5[sharpe_col].min()
                if max_sharpe > 0.8:
                    self.logger.warning(f"⚠️ 发现超高夏普率{max_sharpe:.3f}>0.8，可能过拟合！")
                elif min_sharpe < 0.02:
                    self.logger.warning(f"⚠️ 发现超低夏普率{min_sharpe:.3f}<0.02，可能是噪音！")

            if trades_col and trades_col in top5.columns:
                min_trades = top5[trades_col].min()
                max_trades = top5[trades_col].max()
                if min_trades < 20:
                    self.logger.warning(f"⚠️ 发现超低交易次数{min_trades}<20，样本不足！")
                elif max_trades > 2000:
                    self.logger.warning(f"⚠️ 发现超高交易次数{max_trades}>2000，信号过密！")

            if winrate_col and winrate_col in top5.columns:
                max_winrate = top5[winrate_col].max()
                if max_winrate > 0.6:
                    self.logger.warning(f"⚠️ 发现超高胜率{max_winrate:.1%}>60%，复查是否偷价！")
        
        return {
            'timeframe': timeframe,
            'cta_results': cta_results,
            'factor_ranking': factor_ranking,
            'valid_factors': valid_factors,
            'summary': {
                'tested_symbols': len(symbol_factors),
                'tested_factors': len(factor_names),
                'total_evaluations': len(cta_results),
                'valid_factors_count': len(valid_factors),
                'best_factor': factor_ranking.iloc[0]['factor'] if not factor_ranking.empty else None,
                'best_sharpe': float(factor_ranking.iloc[0][sharpe_col]) if (not factor_ranking.empty and sharpe_col and sharpe_col in factor_ranking.columns and pd.notna(factor_ranking.iloc[0][sharpe_col])) else 0
            }
        }
    
    def run_final_test(self):
        """运行最终测试"""
        print("🚀 启动最终工作版VectorBT测试...")
        start_time = time.time()
        
        self.logger.info("最终测试开始")
        self.logger.info(f"  测试时间框架: {self.working_config['test_timeframes']}")
        self.logger.info(f"  测试股票: {self.test_symbols}")
        
        # 结果初始化
        results = {
            'execution_time': 0,
            'analysis_approach': 'final_working_vectorbt',
            'tested_symbols': self.test_symbols,
            'tested_timeframes': self.working_config['test_timeframes'],
            'working_config': self.working_config,
            'timeframe_results': {}
        }
        
        # 🔥 按时间框架分析 - 支持CTA和IC两种模式
        total_factors = 0
        evaluation_mode = self.working_config.get('evaluation_mode', 'ic')
        self.logger.info(f"  评估模式: {evaluation_mode.upper()}")
        
        for timeframe in self.working_config['test_timeframes']:
            if evaluation_mode == 'cta':
                analysis_results = self._run_cta_analysis(timeframe)
                if analysis_results:
                    results['timeframe_results'][timeframe] = analysis_results
                    total_factors += analysis_results['summary']['valid_factors_count']
            else:
                ic_results = self._analyze_timeframe(timeframe)
                if ic_results:
                    results['timeframe_results'][timeframe] = ic_results
                    total_factors += len(ic_results)
            
        # 完成统计
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        
        self.logger.info(f"最终测试完成，耗时{execution_time:.1f}秒")
        self.logger.info(f"总有效因子: {total_factors}个")
        
        # 保存结果
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """保存结果"""
        result_dir = f"results/final_working_{self.timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存JSON结果
        json_file = f"{result_dir}/final_working_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成简要报告
        self._generate_summary_report(results, result_dir)
        
        self.logger.info(f"结果已保存到: {result_dir}")
    
    def _generate_summary_report(self, results: Dict, result_dir: str):
        """生成总结报告"""
        report = [
            "# 🎯 最终工作版VectorBT测试报告",
            "",
            f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**执行时间**: {results['execution_time']:.1f}秒",
            f"**测试股票**: {len(results['tested_symbols'])}只",
            f"**测试时间框架**: {len(results['tested_timeframes'])}个",
            "",
            "## 📊 结果统计",
            ""
        ]
        
        timeframe_results = results.get('timeframe_results', {})
        
        if timeframe_results:
            total_factors = 0
            # 🔥 根据评估模式生成不同的统计表
            evaluation_mode = results.get('working_config', {}).get('evaluation_mode', 'ic')

            cta_uses_cost = False
            if evaluation_mode == 'cta':
                for result_data in timeframe_results.values():
                    factor_ranking = result_data.get('factor_ranking', pd.DataFrame())
                    if self._select_sharpe_column(factor_ranking) == 'sharpe_cost':
                        cta_uses_cost = True
                        break

                sharpe_header = "最佳夏普(扣费)" if cta_uses_cost else "最佳夏普"
                report.append(f"| 时间框架 | 有效因子数 | 优秀因子 | {sharpe_header} |")
                report.append("|----------|------------|----------|----------|")
            else:
                report.append("| 时间框架 | 有效因子数 | 优秀因子 | 最佳IC |")
                report.append("|----------|------------|----------|--------|")

            for tf, result_data in timeframe_results.items():
                if evaluation_mode == 'cta':
                    # CTA模式: 处理CTA结果
                    factor_count = result_data['summary']['valid_factors_count']
                    total_factors += factor_count

                    # 统计优秀因子 (夏普>0.5)
                    valid_factors = result_data.get('valid_factors', pd.DataFrame())
                    sharpe_col = self._select_sharpe_column(valid_factors)
                    if not valid_factors.empty and sharpe_col:
                        excellent_factors = len(valid_factors[valid_factors[sharpe_col] > 0.5])
                    else:
                        excellent_factors = 0

                    best_sharpe = result_data.get('summary', {}).get('best_sharpe', 0)

                    report.append(f"| {tf} | {factor_count} | {excellent_factors} | {best_sharpe:.3f} |")
                else:
                    # IC模式: 原逻辑
                    factor_count = len(result_data)
                    total_factors += factor_count
                    
                    # 统计优秀因子 (|IC| > 0.02)
                    excellent_factors = sum(1 for f in result_data.values() if abs(f['ic']) > 0.02)
                    
                    # 找最佳IC
                    best_ic = max(result_data.values(), key=lambda x: abs(x['ic']))['ic'] if result_data else 0
                    
                    report.append(f"| {tf} | {factor_count} | {excellent_factors} | {best_ic:.3f} |")
            
            report.extend([
                "",
                f"**总计**: {total_factors}个有效因子",
                ""
            ])
            
            # 🔥 显示最佳因子 - 根据评估模式
            all_factors = []
            
            if evaluation_mode == 'cta':
                # CTA模式: 提取因子排名数据
                sharpe_label = "夏普率(扣费)" if cta_uses_cost else "夏普率"
                for tf, result_data in timeframe_results.items():
                    factor_ranking = result_data.get('factor_ranking', pd.DataFrame())
                    sharpe_col = self._select_sharpe_column(factor_ranking)
                    if not factor_ranking.empty:
                        for _, row in factor_ranking.head(10).iterrows():
                            sharpe_value = row.get(sharpe_col) if sharpe_col else row.get('sharpe_mean')
                            if pd.isna(sharpe_value):
                                sharpe_value = 0
                            all_factors.append({
                                'name': row.get('factor', 'unknown'),
                                'timeframe': tf,
                                'sharpe': float(sharpe_value) if sharpe_value is not None else 0,
                                'win_rate': row.get('win_rate_mean', 0),
                                'trades': row.get('trades_sum', 0)
                            })

                if all_factors:
                    all_factors.sort(key=lambda x: x['sharpe'], reverse=True)

                    report.extend([
                        "## 🏆 最佳因子 (Top 10)",
                        "",
                        f"| 排名 | 因子名称 | 时间框架 | {sharpe_label} | 胜率 | 交易次数 |",
                        "|------|----------|----------|--------|------|----------|"
                    ])

                    for i, factor in enumerate(all_factors[:10], 1):
                        report.append(
                            f"| {i} | {factor['name']} | {factor['timeframe']} | "
                            f"{factor['sharpe']:.3f} | {factor['win_rate']:.1%} | {factor['trades']:.0f} |"
                        )
            else:
                # IC模式: 原逻辑
                for tf, ic_data in timeframe_results.items():
                    for factor_name, factor_data in ic_data.items():
                        all_factors.append({
                            'name': factor_name,
                            'timeframe': tf,
                            'ic': factor_data['ic'],
                            'ic_ir': factor_data['ic_ir'],
                            'supporting_stocks': factor_data['supporting_stocks']
                        })
                
                if all_factors:
                    all_factors.sort(key=lambda x: abs(x['ic']), reverse=True)
                    
                    report.extend([
                        "## 🏆 最佳因子 (Top 10)",
                        "",
                        "| 排名 | 因子名称 | 时间框架 | IC | IC_IR | 支持股票数 |",
                        "|------|----------|----------|-----|-------|------------|"
                    ])
                    
                    for i, factor in enumerate(all_factors[:10], 1):
                        report.append(
                            f"| {i} | {factor['name']} | {factor['timeframe']} | "
                            f"{factor['ic']:.3f} | {factor['ic_ir']:.3f} | {factor['supporting_stocks']} |"
                        )
        else:
            report.append("❌ 未发现有效因子")
        
        # 写入报告
        report_content = "\n".join(report)
        report_file = f"{result_dir}/final_working_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

def main():
    """主函数"""
    # ✅修复4: 设置随机种子确保结果可复现
    np.random.seed(42)
    
    final_analyzer = FinalWorkingVectorBT()
    results = final_analyzer.run_final_test()
    
    if results:
        print("🎉 最终工作版测试完成！")
        
        timeframe_results = results.get('timeframe_results', {})
        evaluation_mode = results.get('working_config', {}).get('evaluation_mode', 'ic')
        
        # 计算总因子数 - 根据评估模式
        if evaluation_mode == 'cta':
            total_factors = sum(result_data['summary']['valid_factors_count'] for result_data in timeframe_results.values())
        else:
            total_factors = sum(len(ic_data) for ic_data in timeframe_results.values())
        
        if evaluation_mode == 'cta':
            print(f"🎯 CTA回测模式: 发现{total_factors}个优质因子 (夏普≥0.05，修复完成)")
            print(f"📊 评估维度: 夏普率、胜率、盈亏比、交易次数")
        else:
            print(f"📊 IC分析模式: 发现{total_factors}个有效因子")
            print(f"📊 评估维度: IC、IC_IR、正向比例")
        print(f"⚡ 覆盖{len(timeframe_results)}个时间框架")
        print(f"📈 测试{len(results['tested_symbols'])}只股票")
        
        if total_factors > 0:
            print("✅ 评估系统正常工作！")
        else:
            print("❌ 仍需进一步调试...")

if __name__ == "__main__":
    main()
