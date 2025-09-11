#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 优化版最终工作系统 V2.0
===============================

基于原版final_working_vectorbt.py的优化版本
保持原版的稳定逻辑，但集成向量化技术提升性能

关键特性：
- 🔥 保持原版V3的核心逻辑和参数
- ⚡ 集成向量化批量回测提升性能  
- 🛡️ 完整的错误处理和日志系统
- 📊 与原版完全一致的输出格式

Author: Optimized VectorBT System V2
Date: 2025-09-11
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
from factors.factor_pool import AdvancedFactorPool
from utils.dtype_fixer import CategoricalDtypeFixer
from strategies.cta_eval_v3 import CTAEvaluatorV3

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

class OptimizedFinalWorking:
    """
    🚀 优化版最终工作系统
    基于原版逻辑，集成向量化优化
    """
    
    def __init__(self, data_dir: str = "/Users/zhangshenshen/longport/vectorized_factor_analyzer_v2/data", capital: float = 300000):
        self.data_dir = data_dir
        self.capital = capital
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🔥 继承原版的稳定配置
        self.working_config = {
            'test_timeframes': ['1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h', '4h', '1d'],
            'max_symbols': 54,
            'evaluation_mode': 'cta',  # 使用稳定的CTA模式
            
            # 原版V3的宽松阈值
            'min_ic_threshold': 0.005,
            'min_ir_threshold': 0.01,
            'min_sample_size': 10,
            'min_supporting_stocks': 2,
            
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
        """设置日志系统"""
        log_dir = f"logs/optimized_final_{self.timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('OptimizedFinal')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            f"{log_dir}/optimized_final.log", 
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
    
    def _run_cta_analysis(self, timeframe: str) -> Dict[str, Any]:
        """🚀 优化版CTA回测分析"""
        self.logger.info(f"🎯 优化版CTA回测分析{timeframe}时间框架...")
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
        
        # 添加内存监控
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"    💾 数据加载后内存使用: {memory_mb:.1f} MB")
        except:
            pass
        
        # 2. 🔥 使用原版V3评估器确保逻辑一致性
        cta_evaluator = CTAEvaluatorV3(
            look_ahead=6,                 # ✅ 原版V3参数
            entry_percentile=0.80,        # ✅ 原版V3参数
            exit_percentile=0.20,         # ✅ 原版V3参数
            sl_stop=0.02,
            tp_stop=0.03,
            direction='both',
            slippage=0.002,               # ✅ 原版V3参数
            fees=0.001,                   # ✅ 原版V3参数
            min_trades=10                 # ✅ 原版V3参数
        )
        
        # 获取所有因子名称
        all_factors = set()
        for factors_df in symbol_factors.values():
            all_factors.update(factors_df.columns)
        factor_names = list(all_factors)
        
        self.logger.info(f"  🔢 开始CTA评估{len(symbol_factors)}只股票 × {len(factor_names)}个因子")
        
        # 3. 批量CTA评估
        cta_results = cta_evaluator.batch_evaluate(
            symbols=list(symbol_factors.keys()),
            factor_data=symbol_factors,
            price_data=symbol_data,
            factor_names=factor_names,
            timeframe=timeframe
        )
        
        if cta_results.empty:
            self.logger.warning(f"❌ {timeframe} CTA评估无结果")
            return {}
        
        # 评估后内存监控
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"    💾 CTA评估后内存使用: {memory_mb:.1f} MB")
        except:
            pass
        
        # 4. 结果验证
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
        
        # 5. 因子排名
        factor_ranking = cta_evaluator.rank_factors(
            cta_results, 
            rank_by='sharpe'
        )
        
        # 6. 统计有效因子
        if factor_ranking.empty:
            valid_factors = pd.DataFrame()
        else:
            # 检查列名是否存在
            sharpe_col = 'sharpe_mean' if 'sharpe_mean' in factor_ranking.columns else 'sharpe'
            if sharpe_col in factor_ranking.columns:
                valid_factors = factor_ranking[factor_ranking[sharpe_col] >= 0.05]
            else:
                valid_factors = factor_ranking.head(10)
        
        self.logger.info(f"  ✅ {timeframe} 发现{len(valid_factors)}个优质因子 (夏普≥0.05)")
        
        # 🔍 Top5因子人工抽查
        if not factor_ranking.empty and len(factor_ranking) >= 1:
            top5 = factor_ranking.head(5)
            required_cols = ['factor', 'sharpe_mean', 'trades_sum', 'win_rate_mean']
            available_cols = [col for col in required_cols if col in top5.columns]
            
            # 如果列名不同，尝试找到对应的列
            col_mapping = {
                'sharpe_mean': 'sharpe' if 'sharpe' in top5.columns else 'sharpe_mean',
                'trades_sum': 'trades' if 'trades' in top5.columns else 'trades_sum', 
                'win_rate_mean': 'win_rate' if 'win_rate' in top5.columns else 'win_rate_mean'
            }
            
            display_cols = ['factor'] + [col_mapping.get(col, col) for col in required_cols[1:] if col_mapping.get(col, col) in top5.columns]
            
            self.logger.info(f"\n🔍 {timeframe} Top5因子人工抽查:")
            self.logger.info("\n" + top5[display_cols].to_string(index=False))
            
            # 异常值警告
            sharpe_col = col_mapping.get('sharpe_mean', 'sharpe_mean')
            trades_col = col_mapping.get('trades_sum', 'trades_sum')
            winrate_col = col_mapping.get('win_rate_mean', 'win_rate_mean')
            
            if sharpe_col in top5.columns:
                max_sharpe = top5[sharpe_col].max()
                min_sharpe = top5[sharpe_col].min()
                if max_sharpe > 0.8:
                    self.logger.warning(f"⚠️ 发现超高夏普率{max_sharpe:.3f}>0.8，可能过拟合！")
                elif min_sharpe < 0.02:
                    self.logger.warning(f"⚠️ 发现超低夏普率{min_sharpe:.3f}<0.02，可能是噪音！")
                    
            if trades_col in top5.columns:
                min_trades = top5[trades_col].min()
                max_trades = top5[trades_col].max()
                if min_trades < 20:
                    self.logger.warning(f"⚠️ 发现超低交易次数{min_trades}<20，样本不足！")
                elif max_trades > 2000:
                    self.logger.warning(f"⚠️ 发现超高交易次数{max_trades}>2000，信号过密！")
                    
            if winrate_col in top5.columns:
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
                'best_sharpe': factor_ranking.iloc[0]['sharpe_mean'] if not factor_ranking.empty else 0
            }
        }
    
    def run_optimized_test(self):
        """运行优化版测试"""
        print("🚀 启动优化版最终工作系统...")
        start_time = time.time()
        
        self.logger.info("优化版测试开始")
        self.logger.info(f"  测试时间框架: {self.working_config['test_timeframes']}")
        self.logger.info(f"  测试股票: {self.test_symbols}")
        self.logger.info(f"  评估模式: {self.working_config['evaluation_mode'].upper()}")
        
        # 结果初始化
        results = {
            'execution_time': 0,
            'analysis_approach': 'optimized_final_working_v2',
            'tested_symbols': self.test_symbols,
            'tested_timeframes': self.working_config['test_timeframes'],
            'working_config': self.working_config,
            'timeframe_results': {}
        }
        
        # 按时间框架分析
        total_factors = 0
        evaluation_mode = self.working_config.get('evaluation_mode', 'cta')
        
        for timeframe in self.working_config['test_timeframes']:
            if evaluation_mode == 'cta':
                analysis_results = self._run_cta_analysis(timeframe)
                if analysis_results:
                    results['timeframe_results'][timeframe] = analysis_results
                    total_factors += analysis_results['summary']['valid_factors_count']
            
        # 完成统计
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        
        self.logger.info(f"优化版测试完成，耗时{execution_time:.1f}秒")
        self.logger.info(f"总有效因子: {total_factors}个")
        
        # 保存结果
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """保存结果"""
        result_dir = f"results/optimized_final_{self.timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存JSON结果
        json_file = f"{result_dir}/optimized_final_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成简要报告
        self._generate_summary_report(results, result_dir)
        
        self.logger.info(f"结果已保存到: {result_dir}")
    
    def _generate_summary_report(self, results: Dict, result_dir: str):
        """生成总结报告"""
        report = [
            "# 🚀 优化版最终工作系统测试报告",
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
            report.append("| 时间框架 | 有效因子数 | 优秀因子 | 最佳夏普率 |")
            report.append("|----------|------------|----------|------------|")
            
            total_factors = 0
            
            for tf, result_data in timeframe_results.items():
                factor_count = result_data['summary']['valid_factors_count']
                total_factors += factor_count
                
                # 统计优秀因子 (夏普>0.5)
                valid_factors = result_data.get('valid_factors', pd.DataFrame())
                if not valid_factors.empty and 'sharpe_mean' in valid_factors.columns:
                    excellent_factors = len(valid_factors[valid_factors['sharpe_mean'] > 0.5])
                    best_sharpe = valid_factors['sharpe_mean'].max() if not valid_factors.empty else 0
                else:
                    excellent_factors = 0
                    best_sharpe = 0
                
                report.append(f"| {tf} | {factor_count} | {excellent_factors} | {best_sharpe:.3f} |")
            
            report.extend([
                "",
                f"**总计**: {total_factors}个有效因子",
                ""
            ])
            
            # 显示最佳因子
            all_factors = []
            
            for tf, result_data in timeframe_results.items():
                factor_ranking = result_data.get('factor_ranking', pd.DataFrame())
                if not factor_ranking.empty:
                    for _, row in factor_ranking.head(10).iterrows():
                        all_factors.append({
                            'name': row.get('factor', 'unknown'),
                            'timeframe': tf,
                            'sharpe': row.get('sharpe_mean', 0),
                            'win_rate': row.get('win_rate_mean', 0),
                            'trades': row.get('trades_sum', 0)
                        })
            
            if all_factors:
                all_factors.sort(key=lambda x: x['sharpe'], reverse=True)
                
                report.extend([
                    "## 🏆 最佳因子 (Top 10)",
                    "",
                    "| 排名 | 因子名称 | 时间框架 | 夏普率 | 胜率 | 交易次数 |",
                    "|------|----------|----------|--------|------|----------|"
                ])
                
                for i, factor in enumerate(all_factors[:10], 1):
                    report.append(
                        f"| {i} | {factor['name']} | {factor['timeframe']} | "
                        f"{factor['sharpe']:.3f} | {factor['win_rate']:.1%} | {factor['trades']:.0f} |"
                    )
        else:
            report.append("❌ 未发现有效因子")
        
        # 写入报告
        report_content = "\n".join(report)
        report_file = f"{result_dir}/optimized_final_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

def main():
    """主函数"""
    # 设置随机种子确保结果可复现
    np.random.seed(42)
    
    print("🚀 优化版最终工作系统 V2.0")
    print("📝 基于原版逻辑，集成向量化优化")
    print("=" * 50)
    
    optimized_analyzer = OptimizedFinalWorking()
    results = optimized_analyzer.run_optimized_test()
    
    if results:
        print("\n🎉 优化版测试完成！")
        
        timeframe_results = results.get('timeframe_results', {})
        total_factors = sum(result_data['summary']['valid_factors_count'] for result_data in timeframe_results.values())
        
        print(f"🎯 CTA回测模式: 发现{total_factors}个优质因子 (夏普≥0.05)")
        print(f"📊 评估维度: 夏普率、胜率、盈亏比、交易次数")
        print(f"⚡ 覆盖{len(timeframe_results)}个时间框架")
        print(f"📈 测试{len(results['tested_symbols'])}只股票")
        
        if total_factors > 0:
            print("✅ 优化版系统正常工作！")
            print(f"⚡ 性能提升：集成向量化技术")
            print(f"🛡️ 稳定性：保持原版V3逻辑")
        else:
            print("❌ 仍需进一步调试...")

if __name__ == "__main__":
    main()
