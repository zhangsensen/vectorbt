#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全股票向量化分析器 - 探查所有54只港股的向量化分析
支持多股票、多时间框架、多因子的批量处理
实现真正的vectorbt优势和内存优化
🎯 重点：个股独立IC分析 → 横截面平均统计
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import pandas as pd
import vectorbt as vbt
import talib as ta
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vectorized_multi_stock_analyzer import VectorizedMultiStockAnalyzer
from advanced_factor_pool import AdvancedFactorPool


class FullStockAnalyzer:
    """全股票向量化分析器 - 处理所有54只港股"""
    
    def __init__(self):
        """初始化全股票分析器"""
        self.analyzer = VectorizedMultiStockAnalyzer()
        self.logger = self._setup_logger()
        
        # 🚀 集成高级因子池
        self.advanced_factor_pool = AdvancedFactorPool()
        
        # 分析配置
        self.batch_sizes = {
            '1m': 5,   # 1分钟数据量大，批量处理5只股票
            '2m': 8,   # 2分钟数据，批量处理8只股票
            '3m': 10,  # 3分钟数据，批量处理10只股票
            '5m': 15,  # 5分钟数据，批量处理15只股票
            '10m': 20, # 10分钟数据，批量处理20只股票
            '15m': 25, # 15分钟数据，批量处理25只股票
            '30m': 30, # 30分钟数据，批量处理30只股票
            '1h': 40,  # 1小时数据，批量处理40只股票
            '4h': 54,  # 4小时数据，可以处理全部54只
            '1d': 54   # 1天数据，可以处理全部54只
        }
        
        # 内存监控阈值
        self.memory_threshold = 0.85  # 85%内存使用率触发清理
        
        # 🎯 高级因子池配置
        self.factor_pool = self.advanced_factor_pool.get_factor_descriptions()
        
        # 分层测试策略（按重要性和计算复杂度）
        self.factor_tiers = {
            'tier_1_core': ['rsi_14', 'macd_enhanced', 'atrp', 'vwap_deviation'],  # 核心4因子
            'tier_2_momentum': ['rsi_2', 'stoch_rsi', 'cci_14', 'roc_12'],        # 动量扩展
            'tier_3_trend': ['dema_14', 'tema_14', 'adx_14', 'aroon_up'],         # 趋势扩展
            'tier_4_volume': ['volume_rsi', 'cmf', 'ad_line', 'volume_intensity'], # 成交量
            'tier_5_micro': ['hl_spread', 'price_efficiency', 'bb_squeeze']       # 微观结构
        }
        
        # 当前测试配置（可动态调整）
        self.current_tier = 'tier_1_core'  # 从核心因子开始
        self.test_factors = self.factor_tiers[self.current_tier]
        
        print("=" * 80)
        print("🚀 全股票向量化分析器 - 54只港股全面探查")
        print("=" * 80)
        print(f"📊 分析配置:")
        print(f"   总股票数: {len(self.analyzer.all_symbols)}")
        print(f"   测试因子: {self.test_factors}")
        print(f"   时间框架: {len(self.analyzer.all_timeframes)}个")
        print(f"   内存限制: {self.analyzer.memory_limit_gb}GB")
        print("")
        
    def _setup_logger(self) -> logging.Logger:
        """设置时间戳日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/full_stock_analysis_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志
        logger = logging.getLogger(f"{__name__}.FullStockAnalyzer")
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(f"{log_dir}/full_stock_analysis.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def _get_memory_usage(self) -> float:
        """获取当前内存使用率"""
        return psutil.virtual_memory().percent / 100.0
        
    def _should_clear_cache(self) -> bool:
        """判断是否需要清理缓存"""
        return self._get_memory_usage() > self.memory_threshold
        
    def _create_stock_batches(self, timeframe: str) -> List[List[str]]:
        """根据时间框架创建股票批次"""
        batch_size = self.batch_sizes.get(timeframe, 10)
        stocks = self.analyzer.all_symbols
        
        batches = []
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            batches.append(batch)
            
        self.logger.info(f"时间框架 {timeframe}: 创建了 {len(batches)} 个批次，每批最多 {batch_size} 只股票")
        return batches
    
    def make_factor_pool(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🚀 升级版因子池 - 30+高级因子一键计算
        替代原来的6个基础因子，使用产业级指标体系
        """
        try:
            self.logger.debug(f"开始计算高级因子池，输入数据形状: {df.shape}")
            
            # 使用高级因子池计算所有因子
            df_with_factors = self.advanced_factor_pool.calculate_all_factors(df)
            
            self.logger.debug(f"高级因子计算完成，输出数据形状: {df_with_factors.shape}")
            
            return df_with_factors
            
        except Exception as e:
            self.logger.error(f"高级因子池计算失败: {e}")
            # 降级到基础因子
            return self._fallback_basic_factors(df)
    
    def _fallback_basic_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """降级到基础因子（当高级因子计算失败时）"""
        self.logger.warning("使用基础因子作为降级方案")
        
        try:
            close, high, low, vol = df['close'], df['high'], df['low'], df['volume']
            
            # 基础因子
            df['rsi_14'] = ta.RSI(close.values, timeperiod=14)
            
            # MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            df['macd_enhanced'] = macd - signal
            
            # 相对ATR
            atr = ta.ATR(high.values, low.values, close.values, timeperiod=14)
            df['atrp'] = atr / close
            
            # VWAP偏离
            vwap = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
            df['vwap_deviation'] = (close - vwap) / close
            
            self.logger.info("基础因子计算完成")
            
        except Exception as e:
            self.logger.error(f"基础因子计算也失败: {e}")
            
        return df
    
    def calculate_individual_stock_ic(self, 
                                    data: pd.DataFrame, 
                                    factors: List[str], 
                                    forward_periods: int = 1) -> Dict[str, Dict[str, float]]:
        """
        核心方法：个股独立IC计算
        先计算每只股票的IC，再进行横截面统计
        """
        self.logger.info(f"开始个股独立IC计算，涉及因子: {factors}")
        
        ic_results = {}
        symbols = data.index.get_level_values('symbol').unique()
        
        # 为每个因子初始化结果容器
        for factor in factors:
            ic_results[factor] = {
                'individual_ics': [],
                'symbol_details': [],
                'statistics': {}
            }
        
        # 逐股票计算IC
        for symbol in symbols:
            try:
                # 提取单只股票数据
                symbol_data = data.loc[symbol].copy()
                
                # 添加因子池
                symbol_data = self.make_factor_pool(symbol_data)
                
                # 计算未来收益率
                fwd_returns = symbol_data['close'].pct_change(periods=forward_periods).shift(-forward_periods)
                
                # 计算每个因子的IC
                for factor in factors:
                    if factor in symbol_data.columns:
                        factor_values = symbol_data[factor]
                        
                        # 数据对齐和清洗
                        common_idx = factor_values.index.intersection(fwd_returns.index)
                        if len(common_idx) > 30:  # 至少30个数据点
                            aligned_factor = factor_values.loc[common_idx]
                            aligned_returns = fwd_returns.loc[common_idx]
                            
                            # 去除NaN
                            valid_mask = aligned_factor.notna() & aligned_returns.notna()
                            clean_factor = aligned_factor[valid_mask]
                            clean_returns = aligned_returns[valid_mask]
                            
                            if len(clean_factor) > 20:
                                # 计算皮尔逊相关系数作为IC
                                ic = clean_factor.corr(clean_returns)
                                
                                if not np.isnan(ic):
                                    ic_results[factor]['individual_ics'].append(ic)
                                    ic_results[factor]['symbol_details'].append({
                                        'symbol': symbol,
                                        'ic': ic,
                                        'sample_size': len(clean_factor),
                                        'factor_mean': float(clean_factor.mean()),
                                        'factor_std': float(clean_factor.std()),
                                        'returns_mean': float(clean_returns.mean()),
                                        'returns_std': float(clean_returns.std())
                                    })
                                    
                                    self.logger.debug(f"{symbol} {factor}: IC={ic:.4f}")
                            else:
                                self.logger.debug(f"{symbol} {factor}: 有效数据不足({len(clean_factor)})")
                        else:
                            self.logger.debug(f"{symbol} {factor}: 数据长度不足({len(common_idx)})")
                    else:
                        self.logger.warning(f"{symbol}: 缺少因子 {factor}")
                        
            except Exception as e:
                self.logger.warning(f"{symbol} IC计算失败: {e}")
        
        # 横截面统计
        for factor in factors:
            individual_ics = ic_results[factor]['individual_ics']
            
            if len(individual_ics) > 0:
                ic_array = np.array(individual_ics)
                
                ic_results[factor]['statistics'] = {
                    'mean_ic': float(np.mean(ic_array)),
                    'median_ic': float(np.median(ic_array)),
                    'std_ic': float(np.std(ic_array)),
                    'min_ic': float(np.min(ic_array)),
                    'max_ic': float(np.max(ic_array)),
                    'ic_ir': float(np.mean(ic_array) / np.std(ic_array)) if np.std(ic_array) > 0 else 0.0,
                    'positive_ic_ratio': float(np.mean(ic_array > 0)),
                    'total_stocks': len(ic_array),
                    'abs_mean_ic': float(np.mean(np.abs(ic_array))),
                    't_stat': float(np.mean(ic_array) / (np.std(ic_array) / np.sqrt(len(ic_array)))) if len(ic_array) > 1 and np.std(ic_array) > 0 else 0.0
                }
                
                self.logger.info(f"✅ {factor}: 平均IC={np.mean(ic_array):.4f}, IC_IR={ic_results[factor]['statistics']['ic_ir']:.2f}, 正IC比例={np.mean(ic_array > 0):.1%}, 覆盖{len(ic_array)}只股票")
            else:
                ic_results[factor]['statistics'] = {
                    'mean_ic': 0.0,
                    'total_stocks': 0
                }
                self.logger.warning(f"❌ {factor}: 无有效IC计算结果")
        
        return ic_results
    
    def _generate_cross_sectional_summary(self, timeframe_results: Dict) -> Dict[str, Any]:
        """
        生成时间框架级别的横截面因子汇总
        将所有批次的IC结果合并，计算因子的整体表现
        """
        cross_sectional_ic = {}
        
        # 初始化因子容器
        for factor in self.test_factors:
            cross_sectional_ic[factor] = {
                'all_individual_ics': [],
                'all_symbol_details': []
            }
        
        # 合并所有批次的IC结果
        for batch in timeframe_results.get('batches', []):
            if batch.get('status') == 'success':
                ic_statistics = batch.get('ic_statistics', {})
                
                for factor in self.test_factors:
                    if factor in ic_statistics:
                        factor_result = ic_statistics[factor]
                        
                        # 合并个股IC
                        individual_ics = factor_result.get('individual_ics', [])
                        cross_sectional_ic[factor]['all_individual_ics'].extend(individual_ics)
                        
                        # 合并详细信息
                        symbol_details = factor_result.get('symbol_details', [])
                        cross_sectional_ic[factor]['all_symbol_details'].extend(symbol_details)
        
        # 计算跨时间框架的横截面统计
        factor_ranking = []
        
        for factor in self.test_factors:
            all_ics = cross_sectional_ic[factor]['all_individual_ics']
            
            if len(all_ics) > 0:
                ic_array = np.array(all_ics)
                
                stats = {
                    'factor': factor,
                    'factor_name': self.factor_pool.get(factor, factor),
                    'mean_ic': float(np.mean(ic_array)),
                    'median_ic': float(np.median(ic_array)),
                    'std_ic': float(np.std(ic_array)),
                    'ic_ir': float(np.mean(ic_array) / np.std(ic_array)) if np.std(ic_array) > 0 else 0.0,
                    'positive_ic_ratio': float(np.mean(ic_array > 0)),
                    'abs_mean_ic': float(np.mean(np.abs(ic_array))),
                    't_stat': float(np.mean(ic_array) / (np.std(ic_array) / np.sqrt(len(ic_array)))) if len(ic_array) > 1 and np.std(ic_array) > 0 else 0.0,
                    'total_stocks': len(all_ics),
                    'top_stocks': sorted(cross_sectional_ic[factor]['all_symbol_details'], key=lambda x: abs(x['ic']), reverse=True)[:5]
                }
                
                factor_ranking.append(stats)
            else:
                factor_ranking.append({
                    'factor': factor,
                    'factor_name': self.factor_pool.get(factor, factor),
                    'mean_ic': 0.0,
                    'total_stocks': 0
                })
        
        # 按|IC_IR|排序
        factor_ranking.sort(key=lambda x: abs(x.get('ic_ir', 0)), reverse=True)
        
        return {
            'factor_ranking': factor_ranking,
            'best_factor': factor_ranking[0] if factor_ranking else None,
            'worst_factor': factor_ranking[-1] if factor_ranking else None,
            'summary_stats': {
                'total_factors_tested': len(factor_ranking),
                'factors_with_positive_mean_ic': sum(1 for f in factor_ranking if f.get('mean_ic', 0) > 0),
                'avg_abs_ic_ir': np.mean([abs(f.get('ic_ir', 0)) for f in factor_ranking]) if factor_ranking else 0
            }
        }
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行全股票综合分析"""
        self.logger.info("🚀 开始全股票综合分析")
        
        start_time = time.time()
        results = {
            'analysis_start_time': datetime.now().isoformat(),
            'timeframe_results': {},
            'overall_statistics': {},
            'error_summary': {},
            'performance_metrics': {}
        }
        
        total_combinations = 0
        successful_combinations = 0
        failed_combinations = 0
        
        # 逐个时间框架分析
        for timeframe_idx, timeframe in enumerate(self.analyzer.all_timeframes, 1):
            print(f"\n📈 [{timeframe_idx}/{len(self.analyzer.all_timeframes)}] 分析时间框架: {timeframe}")
            self.logger.info(f"开始分析时间框架: {timeframe}")
            
            timeframe_start = time.time()
            timeframe_results = {
                'batches': [],
                'successful_stocks': [],
                'failed_stocks': [],
                'total_stocks_processed': 0,
                'execution_time': 0
            }
            
            # 创建股票批次
            stock_batches = self._create_stock_batches(timeframe)
            
            # 逐批次处理
            for batch_idx, stock_batch in enumerate(stock_batches, 1):
                print(f"   🔄 批次 [{batch_idx}/{len(stock_batches)}]: 处理 {len(stock_batch)} 只股票")
                self.logger.info(f"开始处理批次 {batch_idx}: {stock_batch}")
                
                batch_start = time.time()
                
                try:
                    # 🎯 使用个股独立IC分析 (替代原来的向量化分析)
                    # 1. 加载数据
                    batch_data = self.analyzer.load_timeframe_data_vectorized(timeframe, stock_batch)
                    
                    # 2. 计算个股独立IC
                    ic_results = self.calculate_individual_stock_ic(batch_data, self.test_factors)
                    
                    batch_time = time.time() - batch_start
                    
                    # 3. 统计批次结果
                    if ic_results and any(ic_results[factor]['statistics'].get('total_stocks', 0) > 0 for factor in self.test_factors):
                        batch_successful = len(stock_batch)
                        batch_failed = 0
                        timeframe_results['successful_stocks'].extend(stock_batch)
                        
                        # 记录详细IC统计
                        timeframe_results['batches'].append({
                            'batch_id': batch_idx,
                            'stocks': stock_batch,
                            'execution_time': batch_time,
                            'ic_statistics': ic_results,
                            'status': 'success'
                        })
                        
                        # 显示因子表现亮点
                        best_factor = None
                        best_ic_ir = -999
                        for factor in self.test_factors:
                            ic_ir = ic_results[factor]['statistics'].get('ic_ir', 0)
                            if abs(ic_ir) > abs(best_ic_ir):
                                best_ic_ir = ic_ir
                                best_factor = factor
                        
                        print(f"   ✅ 批次成功 - 耗时: {batch_time:.3f}秒 | 最佳因子: {best_factor}(IC_IR={best_ic_ir:.2f})")
                        
                    else:
                        batch_successful = 0
                        batch_failed = len(stock_batch)
                        timeframe_results['failed_stocks'].extend(stock_batch)
                        
                        timeframe_results['batches'].append({
                            'batch_id': batch_idx,
                            'stocks': stock_batch,
                            'execution_time': batch_time,
                            'error': 'No valid IC results',
                            'status': 'failed'
                        })
                        
                        print(f"   ❌ 批次失败 - 无有效IC结果")
                        
                except Exception as e:
                    batch_time = time.time() - batch_start
                    batch_successful = 0
                    batch_failed = len(stock_batch)
                    timeframe_results['failed_stocks'].extend(stock_batch)
                    
                    error_msg = str(e)
                    timeframe_results['batches'].append({
                        'batch_id': batch_idx,
                        'stocks': stock_batch,
                        'execution_time': batch_time,
                        'error': error_msg,
                        'status': 'exception'
                    })
                    
                    self.logger.error(f"批次 {batch_idx} 异常: {error_msg}")
                    print(f"   ❌ 批次异常 - {error_msg}")
                
                # 更新统计
                successful_combinations += batch_successful * len(self.test_factors)
                failed_combinations += batch_failed * len(self.test_factors)
                total_combinations += len(stock_batch) * len(self.test_factors)
                timeframe_results['total_stocks_processed'] += len(stock_batch)
                
                # 内存管理
                if self._should_clear_cache():
                    self.logger.info(f"内存使用率: {self._get_memory_usage():.1%}, 清理缓存")
                    self.analyzer._clear_cache()
                    
                # 批次间暂停（避免过载）
                time.sleep(0.1)
            
            # 时间框架统计
            timeframe_time = time.time() - timeframe_start
            timeframe_results['execution_time'] = timeframe_time
            
            success_rate = len(timeframe_results['successful_stocks']) / len(self.analyzer.all_symbols) * 100
            
            print(f"   📊 {timeframe} 汇总:")
            print(f"       成功股票: {len(timeframe_results['successful_stocks'])}")
            print(f"       失败股票: {len(timeframe_results['failed_stocks'])}")
            print(f"       成功率: {success_rate:.1f}%")
            print(f"       执行时间: {timeframe_time:.3f}秒")
            
            # 🎯 时间框架级别的横截面因子汇总
            timeframe_results['cross_sectional_summary'] = self._generate_cross_sectional_summary(timeframe_results)
            
            results['timeframe_results'][timeframe] = timeframe_results
            
            # 强制缓存清理（每个时间框架之间）
            self.analyzer._clear_cache()
            
        # 总体统计
        total_time = time.time() - start_time
        overall_success_rate = successful_combinations / total_combinations * 100 if total_combinations > 0 else 0
        
        results['overall_statistics'] = {
            'total_combinations': total_combinations,
            'successful_combinations': successful_combinations,
            'failed_combinations': failed_combinations,
            'overall_success_rate': overall_success_rate,
            'total_execution_time': total_time,
            'average_time_per_timeframe': total_time / len(self.analyzer.all_timeframes),
            'memory_peak_usage': self._get_memory_usage()
        }
        
        results['analysis_end_time'] = datetime.now().isoformat()
        
        return results
        
    def save_results(self, results: Dict[str, Any]) -> str:
        """保存分析结果到时间戳目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/full_stock_analysis_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(results_dir, "full_stock_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # 生成汇总报告
        summary_file = os.path.join(results_dir, "analysis_summary.txt")
        self._generate_summary_report(results, summary_file)
        
        # 🎯 生成因子探查专用报告
        factor_report_file = os.path.join(results_dir, "factor_exploration_report.md")
        self._generate_factor_exploration_report(results, factor_report_file)
        
        self.logger.info(f"结果已保存到: {results_dir}")
        return results_dir
        
    def _generate_summary_report(self, results: Dict[str, Any], summary_file: str):
        """生成可读的汇总报告"""
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🎯 全股票向量化分析 - 汇总报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体统计
            stats = results['overall_statistics']
            f.write("📊 总体统计:\n")
            f.write(f"总组合数: {stats['total_combinations']}\n")
            f.write(f"成功组合: {stats['successful_combinations']}\n")
            f.write(f"失败组合: {stats['failed_combinations']}\n")
            f.write(f"总成功率: {stats['overall_success_rate']:.1f}%\n")
            f.write(f"总执行时间: {stats['total_execution_time']:.2f}秒\n")
            f.write(f"平均每时间框架: {stats['average_time_per_timeframe']:.2f}秒\n\n")
            
            # 各时间框架详情
            f.write("📈 各时间框架表现:\n")
            for timeframe, tf_results in results['timeframe_results'].items():
                success_count = len(tf_results['successful_stocks'])
                total_count = tf_results['total_stocks_processed']
                success_rate = success_count / total_count * 100 if total_count > 0 else 0
                
                f.write(f"\n{timeframe}:\n")
                f.write(f"  成功股票数: {success_count}/{total_count}\n")
                f.write(f"  成功率: {success_rate:.1f}%\n")
                f.write(f"  执行时间: {tf_results['execution_time']:.3f}秒\n")
                f.write(f"  批次数: {len(tf_results['batches'])}\n")
                
                if tf_results['failed_stocks']:
                    f.write(f"  失败股票: {tf_results['failed_stocks']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"分析完成时间: {results['analysis_end_time']}\n")
    
    def _generate_factor_exploration_report(self, results: Dict[str, Any], report_file: str):
        """生成因子探查专用报告 - 按|IC_IR|排序显示最佳因子"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🎯 因子探查专用报告\n\n")
            f.write(f"**分析时间**: {results['analysis_end_time']}\n")
            f.write(f"**测试因子数**: {len(self.test_factors)}\n")
            f.write(f"**涵盖时间框架**: {len(self.analyzer.all_timeframes)}个\n")
            f.write(f"**分析股票数**: {len(self.analyzer.all_symbols)}只\n\n")
            
            f.write("## 📊 各时间框架因子表现汇总\n\n")
            
            # 生成跨时间框架的因子排行榜
            all_timeframe_rankings = {}
            
            for timeframe, tf_results in results['timeframe_results'].items():
                cross_sectional = tf_results.get('cross_sectional_summary', {})
                factor_ranking = cross_sectional.get('factor_ranking', [])
                
                if factor_ranking:
                    f.write(f"### {timeframe} 时间框架\n\n")
                    f.write("| 排名 | 因子名称 | 平均IC | IC_IR | 正IC比例 | 覆盖股票 | t统计量 |\n")
                    f.write("|------|----------|--------|-------|----------|----------|----------|\n")
                    
                    for rank, factor_stats in enumerate(factor_ranking, 1):
                        factor_name = factor_stats.get('factor_name', factor_stats.get('factor', 'Unknown'))
                        mean_ic = factor_stats.get('mean_ic', 0)
                        ic_ir = factor_stats.get('ic_ir', 0)
                        pos_ratio = factor_stats.get('positive_ic_ratio', 0)
                        total_stocks = factor_stats.get('total_stocks', 0)
                        t_stat = factor_stats.get('t_stat', 0)
                        
                        # 添加表现标识
                        performance_icon = "🔥" if abs(ic_ir) > 1.5 else "✅" if abs(ic_ir) > 0.5 else "⚠️"
                        
                        f.write(f"| {rank} | {performance_icon} {factor_name} | {mean_ic:.4f} | {ic_ir:.2f} | {pos_ratio:.1%} | {total_stocks} | {t_stat:.2f} |\n")
                    
                    # 最佳因子详情
                    best_factor = cross_sectional.get('best_factor', {})
                    if best_factor:
                        f.write(f"\n**🏆 {timeframe}最佳因子**: {best_factor.get('factor_name', 'Unknown')}\n")
                        f.write(f"- 平均IC: {best_factor.get('mean_ic', 0):.4f}\n")
                        f.write(f"- IC_IR: {best_factor.get('ic_ir', 0):.2f}\n")
                        f.write(f"- 正IC比例: {best_factor.get('positive_ic_ratio', 0):.1%}\n")
                        
                        # 展示该因子的top股票
                        top_stocks = best_factor.get('top_stocks', [])
                        if top_stocks:
                            f.write(f"- Top 5 股票:\n")
                            for stock in top_stocks[:5]:
                                f.write(f"  - {stock['symbol']}: IC={stock['ic']:.4f}\n")
                    
                    f.write("\n")
                    
                    # 收集跨时间框架排名数据
                    for factor_stats in factor_ranking:
                        factor_code = factor_stats.get('factor', '')
                        if factor_code not in all_timeframe_rankings:
                            all_timeframe_rankings[factor_code] = []
                        all_timeframe_rankings[factor_code].append({
                            'timeframe': timeframe,
                            'ic_ir': factor_stats.get('ic_ir', 0),
                            'mean_ic': factor_stats.get('mean_ic', 0),
                            'rank': rank
                        })
            
            # 生成跨时间框架因子稳定性排行
            f.write("## 🎖️ 跨时间框架因子稳定性排行\n\n")
            f.write("*基于各时间框架IC_IR的平均绝对值排序*\n\n")
            
            stability_ranking = []
            for factor_code, timeframe_data in all_timeframe_rankings.items():
                if len(timeframe_data) > 0:
                    avg_abs_ic_ir = np.mean([abs(d['ic_ir']) for d in timeframe_data])
                    avg_ic = np.mean([d['mean_ic'] for d in timeframe_data])
                    consistency = 1 - np.std([d['ic_ir'] for d in timeframe_data]) / (np.mean([abs(d['ic_ir']) for d in timeframe_data]) + 1e-8)
                    
                    stability_ranking.append({
                        'factor': factor_code,
                        'factor_name': self.factor_pool.get(factor_code, factor_code),
                        'avg_abs_ic_ir': avg_abs_ic_ir,
                        'avg_ic': avg_ic,
                        'consistency': consistency,
                        'timeframe_data': timeframe_data
                    })
            
            stability_ranking.sort(key=lambda x: x['avg_abs_ic_ir'], reverse=True)
            
            f.write("| 排名 | 因子名称 | 平均|IC_IR| | 平均IC | 一致性 | 详细表现 |\n")
            f.write("|------|----------|-------------|--------|--------|----------|\n")
            
            for rank, factor_data in enumerate(stability_ranking[:10], 1):  # 只显示前10名
                factor_name = factor_data['factor_name']
                avg_abs_ic_ir = factor_data['avg_abs_ic_ir']
                avg_ic = factor_data['avg_ic']
                consistency = factor_data['consistency']
                
                # 生成各时间框架表现摘要
                tf_summary = ", ".join([f"{d['timeframe']}({d['ic_ir']:.2f})" for d in factor_data['timeframe_data']])
                
                # 表现等级
                if avg_abs_ic_ir > 1.0:
                    grade = "🏆 优秀"
                elif avg_abs_ic_ir > 0.5:
                    grade = "🥈 良好"
                elif avg_abs_ic_ir > 0.2:
                    grade = "🥉 一般"
                else:
                    grade = "❌ 较差"
                
                f.write(f"| {rank} | {grade} {factor_name} | {avg_abs_ic_ir:.3f} | {avg_ic:.4f} | {consistency:.2f} | {tf_summary} |\n")
            
            f.write("\n## 💡 因子探查建议\n\n")
            
            if stability_ranking:
                best_overall = stability_ranking[0]
                f.write(f"### 🎯 重点推荐因子\n")
                f.write(f"**{best_overall['factor_name']}** (代码: `{best_overall['factor']}`)  \n")
                f.write(f"- 平均|IC_IR|: {best_overall['avg_abs_ic_ir']:.3f}\n")
                f.write(f"- 平均IC: {best_overall['avg_ic']:.4f}\n")
                f.write(f"- 跨时间框架一致性: {best_overall['consistency']:.2f}\n\n")
                
                f.write("### 📈 使用建议\n")
                if best_overall['avg_abs_ic_ir'] > 1.0:
                    f.write("- **策略建议**: 该因子表现优异，建议作为主要交易信号\n")
                    f.write("- **风险控制**: 建议设置止损，IC>0.15时加仓，IC<-0.15时减仓\n")
                elif best_overall['avg_abs_ic_ir'] > 0.5:
                    f.write("- **策略建议**: 该因子表现良好，可与其他因子组合使用\n")
                    f.write("- **风险控制**: 建议保守仓位，密切监控因子失效\n")
                else:
                    f.write("- **策略建议**: 因子表现一般，建议进一步优化参数或寻找替代因子\n")
                    f.write("- **风险控制**: 严格限制仓位，或仅作为辅助信号\n")
                
                f.write("\n### 🔬 进一步探索\n")
                f.write("1. **参数优化**: 对表现最佳的因子进行参数调优\n")
                f.write("2. **因子组合**: 尝试多因子组合，降低单一因子风险\n")
                f.write("3. **行业中性**: 考虑行业中性化处理，消除行业偏差\n")
                f.write("4. **动态调整**: 根据市场环境动态调整因子权重\n")
            
            f.write(f"\n---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    """主函数 - 执行全股票分析"""
    analyzer = FullStockAnalyzer()
    
    try:
        # 运行全面分析
        results = analyzer.run_comprehensive_analysis()
        
        # 保存结果
        results_dir = analyzer.save_results(results)
        
        # 打印最终汇总
        stats = results['overall_statistics']
        print("\n" + "=" * 80)
        print("📊 全股票分析汇总")
        print("=" * 80)
        print(f"总组合数: {stats['total_combinations']}")
        print(f"成功组合: {stats['successful_combinations']}")
        print(f"失败组合: {stats['failed_combinations']}")
        print(f"总成功率: {stats['overall_success_rate']:.1f}%")
        print(f"总执行时间: {stats['total_execution_time']:.2f}秒")
        print(f"平均每时间框架: {stats['average_time_per_timeframe']:.2f}秒")
        print(f"\n📄 详细结果已保存: {results_dir}")
        print("\n🎉 全股票分析完成!")
        
        if stats['overall_success_rate'] >= 90:
            print("✅ 系统表现优秀 (成功率 ≥ 90%)")
        elif stats['overall_success_rate'] >= 70:
            print("⚠️ 系统表现良好 (成功率 ≥ 70%)")
        else:
            print("❌ 系统需要优化 (成功率 < 70%)")
            
    except Exception as e:
        analyzer.logger.error(f"全股票分析失败: {str(e)}")
        print(f"❌ 分析失败: {str(e)}")
        return False
        
    return True


if __name__ == "__main__":
    main()
