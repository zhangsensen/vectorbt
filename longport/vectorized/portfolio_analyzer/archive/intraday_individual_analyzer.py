#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内交易专用 - 个股独立IC分析器
针对30万港币资金的日内交易优化设计
每只股票独立计算IC，专注高频时间框架
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import gc
from pathlib import Path

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vectorized_multi_stock_analyzer import VectorizedMultiStockAnalyzer


class IntradayIndividualAnalyzer(VectorizedMultiStockAnalyzer):
    """日内交易专用 - 个股独立分析器"""
    
    def __init__(self, data_directory: str = None, capital: float = 300000):
        """
        初始化日内交易分析器
        
        Args:
            data_directory: 数据目录路径
            capital: 交易资金（港币）
        """
        # 继承基础分析器
        if data_directory is None:
            data_directory = "/Users/zhangshenshen/longport/vectorbt_workspace/data"
        super().__init__(data_directory)
        
        # 日内交易专用配置
        self.capital = capital  # 30万港币
        self.intraday_timeframes = ['1m', '2m', '3m', '5m', '10m', '15m', '30m']  # 重点关注高频框架
        self.intraday_factors = ['RSI', 'MACD', 'Momentum_ROC']  # 适合日内的因子
        
        # 仓位管理参数
        self.max_position_per_stock = 0.1  # 单股最大仓位10%
        self.max_total_positions = 5  # 最多持仓5只股票
        self.min_trade_amount = 5000  # 最小交易金额5000港币
        
        # 日内交易时间窗口（港股交易时间）
        self.trading_sessions = {
            'morning': ('09:30', '12:00'),  # 早市
            'afternoon': ('13:00', '16:00')  # 午市
        }
        
        self.logger.info(f"日内交易分析器初始化完成:")
        self.logger.info(f"  资金规模: {self.capital:,.0f} 港币")
        self.logger.info(f"  关注时间框架: {self.intraday_timeframes}")
        self.logger.info(f"  单股最大仓位: {self.max_position_per_stock:.1%}")
        self.logger.info(f"  最多持仓数: {self.max_total_positions}只")
    
    def calculate_individual_ic(self, 
                              data: pd.DataFrame, 
                              factors_dict: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """
        核心改进：每只股票独立计算IC
        """
        self.logger.info("开始个股独立IC计算")
        
        ic_results = {}
        
        # 获取所有股票列表
        symbols = data.index.get_level_values('symbol').unique()
        
        # 对每个因子分别处理
        for factor_name in factors_dict.keys():
            self.logger.info(f"  处理因子: {factor_name}")
            
            individual_ics = []
            individual_stats = []
            
            # 对每只股票独立计算IC
            for symbol in symbols:
                try:
                    # 提取单只股票的数据
                    symbol_data = data.loc[symbol]
                    symbol_factor = factors_dict[factor_name].loc[symbol]
                    
                    # 计算未来收益率
                    symbol_returns = symbol_data['close'].pct_change(periods=1).shift(-1)
                    
                    # 确保数据对齐
                    common_index = symbol_factor.index.intersection(symbol_returns.index)
                    
                    if len(common_index) > 20:  # 至少需要20个数据点
                        aligned_factor = symbol_factor.loc[common_index]
                        aligned_returns = symbol_returns.loc[common_index]
                        
                        # 去除NaN值
                        valid_mask = aligned_factor.notna() & aligned_returns.notna()
                        clean_factor = aligned_factor[valid_mask]
                        clean_returns = aligned_returns[valid_mask]
                        
                        if len(clean_factor) > 15:
                            # 计算个股IC
                            ic = clean_factor.corr(clean_returns)
                            
                            if not np.isnan(ic):
                                individual_ics.append(ic)
                                individual_stats.append({
                                    'symbol': symbol,
                                    'ic': ic,
                                    'sample_size': len(clean_factor),
                                    'factor_mean': float(clean_factor.mean()),
                                    'factor_std': float(clean_factor.std()),
                                    'returns_mean': float(clean_returns.mean()),
                                    'returns_std': float(clean_returns.std()),
                                    'abs_ic': abs(ic)  # 绝对IC值，用于排序
                                })
                                
                                self.logger.debug(f"    {symbol}: IC={ic:.4f}, 样本={len(clean_factor)}")
                            else:
                                self.logger.warning(f"    {symbol}: IC计算为NaN")
                        else:
                            self.logger.warning(f"    {symbol}: 有效数据不足({len(clean_factor)})")
                    else:
                        self.logger.warning(f"    {symbol}: 数据点不足({len(common_index)})")
                        
                except Exception as e:
                    self.logger.warning(f"    {symbol}: IC计算失败 - {e}")
            
            # 汇总统计
            if individual_ics:
                ic_results[factor_name] = {
                    'mean_ic': float(np.mean(individual_ics)),
                    'median_ic': float(np.median(individual_ics)),
                    'std_ic': float(np.std(individual_ics)),
                    'min_ic': float(np.min(individual_ics)),
                    'max_ic': float(np.max(individual_ics)),
                    'positive_ic_ratio': float(np.mean([ic > 0 for ic in individual_ics])),
                    'total_stocks': len(individual_ics),
                    'individual_stats': individual_stats,
                    'ic_ir': float(np.mean(individual_ics) / np.std(individual_ics)) if np.std(individual_ics) > 0 else 0.0
                }
                
                self.logger.info(f"  ✅ {factor_name}: 平均IC={np.mean(individual_ics):.4f}, "
                               f"正IC比例={np.mean([ic > 0 for ic in individual_ics]):.1%}, "
                               f"覆盖股票={len(individual_ics)}只")
            else:
                ic_results[factor_name] = {
                    'mean_ic': 0.0,
                    'total_stocks': 0,
                    'individual_stats': []
                }
                self.logger.warning(f"  ❌ {factor_name}: 无有效IC计算结果")
        
        return ic_results
    
    def calculate_intraday_factors(self, 
                                 data: pd.DataFrame, 
                                 factors: List[str] = None) -> Dict[str, pd.Series]:
        """
        优化的日内因子计算 - 考虑港股交易特性
        """
        if factors is None:
            factors = self.intraday_factors
        
        self.logger.info(f"开始日内因子计算: {factors}")
        
        factors_dict = {}
        grouped = data.groupby('symbol')
        
        for factor_name in factors:
            self.logger.info(f"  计算因子: {factor_name}")
            
            factor_results = []
            
            for symbol, group_data in grouped:
                try:
                    # 根据因子类型调整参数
                    factor_values = self._calculate_intraday_factor(group_data, factor_name)
                    
                    if factor_values is not None and len(factor_values) > 0:
                        factor_series = pd.Series(
                            factor_values, 
                            index=group_data.index.get_level_values('timestamp'),
                            name=f"{symbol}_{factor_name}"
                        )
                        
                        factor_df = factor_series.to_frame(factor_name)
                        factor_df['symbol'] = symbol
                        factor_results.append(factor_df)
                        
                except Exception as e:
                    self.logger.warning(f"{symbol}计算{factor_name}失败: {e}")
            
            if factor_results:
                factor_combined = pd.concat(factor_results)
                factor_combined.reset_index(inplace=True)
                factor_combined.set_index(['symbol', 'timestamp'], inplace=True)
                factor_combined.sort_index(inplace=True)
                
                factors_dict[factor_name] = factor_combined[factor_name]
                
                self.logger.info(f"  ✅ {factor_name}计算完成: {len(factor_results)}只股票")
            else:
                self.logger.warning(f"  ❌ {factor_name}计算失败：无有效数据")
        
        return factors_dict
    
    def _calculate_intraday_factor(self, data: pd.DataFrame, factor_name: str) -> np.ndarray:
        """
        日内因子计算 - 针对高频数据优化参数
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            if factor_name == "RSI":
                # 日内RSI使用较短周期
                return talib.RSI(close, timeperiod=9)
            
            elif factor_name == "MACD":
                # 日内MACD使用快速参数
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=4)
                return macd
            
            elif factor_name == "Momentum_ROC":
                # 日内动量使用短周期
                return talib.ROC(close, timeperiod=5)
            
            elif factor_name == "Price_Position":
                # 日内价格位置
                period = 10  # 缩短周期
                rolling_high = pd.Series(high).rolling(period).max()
                rolling_low = pd.Series(low).rolling(period).min()
                price_position = (close - rolling_low) / (rolling_high - rolling_low)
                return price_position.values
            
            elif factor_name == "Volume_Ratio":
                # 日内成交量比率
                period = 10  # 缩短周期
                avg_volume = pd.Series(volume).rolling(period).mean()
                volume_ratio = volume / avg_volume
                return volume_ratio.values
            
            elif factor_name == "Volatility":
                # 日内波动率
                period = 10  # 缩短周期
                returns = pd.Series(close).pct_change()
                volatility = returns.rolling(period).std()
                return volatility.values
            
            else:
                self.logger.warning(f"未知日内因子: {factor_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"计算日内因子{factor_name}失败: {e}")
            return None
    
    def calculate_position_sizing(self, ic_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        基于IC结果计算仓位配置
        
        Returns:
            Dict[symbol, position_weight]: 股票代码到仓位权重的映射
        """
        self.logger.info("开始计算仓位配置")
        
        # 收集所有股票的综合评分
        stock_scores = {}
        
        for factor_name, factor_results in ic_results.items():
            if 'individual_stats' in factor_results:
                for stock_stat in factor_results['individual_stats']:
                    symbol = stock_stat['symbol']
                    abs_ic = stock_stat['abs_ic']
                    sample_size = stock_stat['sample_size']
                    
                    # 计算权重评分：绝对IC值 * 样本数权重
                    score = abs_ic * min(sample_size / 100, 1.0)  # 样本数归一化
                    
                    if symbol not in stock_scores:
                        stock_scores[symbol] = []
                    stock_scores[symbol].append(score)
        
        # 计算每只股票的平均评分
        final_scores = {}
        for symbol, scores in stock_scores.items():
            final_scores[symbol] = np.mean(scores)
        
        # 选择top股票并分配仓位
        sorted_stocks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = sorted_stocks[:self.max_total_positions]
        
        if not selected_stocks:
            self.logger.warning("没有找到合适的股票")
            return {}
        
        # 计算仓位权重
        total_score = sum(score for _, score in selected_stocks)
        position_weights = {}
        
        for symbol, score in selected_stocks:
            # 基础权重 + 最小/最大仓位限制
            base_weight = score / total_score
            adjusted_weight = min(base_weight, self.max_position_per_stock)
            adjusted_weight = max(adjusted_weight, self.min_trade_amount / self.capital)
            
            position_weights[symbol] = adjusted_weight
        
        # 归一化权重
        total_weight = sum(position_weights.values())
        if total_weight > 0:
            position_weights = {k: v/total_weight for k, v in position_weights.items()}
        
        self.logger.info(f"仓位配置完成:")
        for symbol, weight in position_weights.items():
            amount = weight * self.capital
            self.logger.info(f"  {symbol}: {weight:.1%} (约{amount:,.0f}港币)")
        
        return position_weights
    
    def run_intraday_analysis(self, 
                            symbols: List[str] = None, 
                            timeframes: List[str] = None) -> Dict:
        """
        运行日内交易分析 - 个股独立IC计算
        """
        if symbols is None:
            symbols = self.all_symbols[:10]  # 测试前10只股票
        if timeframes is None:
            timeframes = self.intraday_timeframes[:4]  # 测试前4个高频框架
        
        self.logger.info(f"🚀 开始日内交易分析:")
        self.logger.info(f"  分析股票: {len(symbols)}只")
        self.logger.info(f"  时间框架: {timeframes}")
        self.logger.info(f"  目标因子: {self.intraday_factors}")
        
        start_time = time.time()
        results = {
            'metadata': {
                'analysis_type': 'intraday_individual',
                'capital': self.capital,
                'symbols': symbols,
                'timeframes': timeframes,
                'factors': self.intraday_factors,
                'start_time': datetime.now().isoformat()
            },
            'timeframe_results': {},
            'position_recommendations': {}
        }
        
        # 分析每个时间框架
        for timeframe in timeframes:
            self.logger.info(f"\n📈 分析时间框架: {timeframe}")
            
            try:
                # 1. 加载数据
                data = self.load_timeframe_data_vectorized(timeframe, symbols)
                
                # 2. 计算日内因子
                factors_dict = self.calculate_intraday_factors(data, self.intraday_factors)
                
                # 3. 计算个股独立IC
                ic_results = self.calculate_individual_ic(data, factors_dict)
                
                # 4. 计算仓位建议
                position_weights = self.calculate_position_sizing(ic_results)
                
                # 5. 存储结果
                results['timeframe_results'][timeframe] = {
                    'data_shape': data.shape,
                    'symbols_analyzed': len(data.index.get_level_values('symbol').unique()),
                    'factors_calculated': list(factors_dict.keys()),
                    'ic_results': ic_results,
                    'analysis_status': 'success'
                }
                
                results['position_recommendations'][timeframe] = position_weights
                
                self.logger.info(f"✅ {timeframe}分析完成")
                
            except Exception as e:
                error_msg = f"{timeframe}分析失败: {str(e)}"
                self.logger.error(error_msg)
                
                results['timeframe_results'][timeframe] = {
                    'analysis_status': 'failed',
                    'error': str(e)
                }
        
        # 计算总体统计
        execution_time = time.time() - start_time
        results['metadata']['end_time'] = datetime.now().isoformat()
        results['metadata']['execution_time'] = execution_time
        
        self.logger.info(f"\n🎉 日内分析完成! 总耗时: {execution_time:.2f}秒")
        
        return results
    
    def save_intraday_results(self, results: Dict, output_dir: str = None) -> str:
        """保存日内交易分析结果"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/intraday_analysis_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(output_dir, "intraday_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成仓位建议报告
        self._generate_position_report(results, output_dir)
        
        # 生成因子表现报告
        self._generate_factor_report(results, output_dir)
        
        self.logger.info(f"日内分析结果已保存到: {output_dir}")
        return output_dir
    
    def _generate_position_report(self, results: Dict, output_dir: str):
        """生成仓位建议报告"""
        report_file = os.path.join(output_dir, "position_recommendations.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 日内交易仓位建议报告\n\n")
            f.write(f"**分析时间**: {results['metadata']['start_time']}\n")
            f.write(f"**资金规模**: {results['metadata']['capital']:,.0f} 港币\n")
            f.write(f"**分析股票**: {len(results['metadata']['symbols'])}只\n\n")
            
            f.write("## 各时间框架仓位建议\n\n")
            
            for timeframe, positions in results.get('position_recommendations', {}).items():
                f.write(f"### {timeframe} 时间框架\n\n")
                
                if positions:
                    f.write("| 股票代码 | 建议仓位 | 资金分配 |\n")
                    f.write("|---------|---------|----------|\n")
                    
                    for symbol, weight in positions.items():
                        amount = weight * results['metadata']['capital']
                        f.write(f"| {symbol} | {weight:.1%} | {amount:,.0f} 港币 |\n")
                    
                    f.write(f"\n**总仓位使用**: {sum(positions.values()):.1%}\n")
                    f.write(f"**剩余现金**: {(1-sum(positions.values()))*results['metadata']['capital']:,.0f} 港币\n\n")
                else:
                    f.write("暂无仓位建议\n\n")
    
    def _generate_factor_report(self, results: Dict, output_dir: str):
        """生成因子表现报告"""
        report_file = os.path.join(output_dir, "factor_performance.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 日内因子表现报告\n\n")
            f.write(f"**分析时间**: {results['metadata']['start_time']}\n")
            f.write(f"**分析因子**: {', '.join(results['metadata']['factors'])}\n\n")
            
            for timeframe, timeframe_results in results.get('timeframe_results', {}).items():
                f.write(f"## {timeframe} 时间框架因子表现\n\n")
                
                ic_results = timeframe_results.get('ic_results', {})
                
                if ic_results:
                    f.write("| 因子 | 平均IC | IC标准差 | IC_IR | 正IC比例 | 覆盖股票 |\n")
                    f.write("|-----|--------|---------|-------|----------|----------|\n")
                    
                    for factor_name, factor_stats in ic_results.items():
                        mean_ic = factor_stats.get('mean_ic', 0)
                        std_ic = factor_stats.get('std_ic', 0)
                        ic_ir = factor_stats.get('ic_ir', 0)
                        pos_ratio = factor_stats.get('positive_ic_ratio', 0)
                        total_stocks = factor_stats.get('total_stocks', 0)
                        
                        f.write(f"| {factor_name} | {mean_ic:.4f} | {std_ic:.4f} | {ic_ir:.2f} | {pos_ratio:.1%} | {total_stocks} |\n")
                    
                    f.write("\n")
                else:
                    f.write("无因子表现数据\n\n")


if __name__ == "__main__":
    # 创建日内交易分析器
    analyzer = IntradayIndividualAnalyzer()
    
    # 运行日内分析测试
    print("🚀 开始日内交易分析测试...")
    
    # 测试参数
    test_symbols = ['0700.HK', '0005.HK', '0388.HK', '0981.HK', '1211.HK']  # 5只活跃股票
    test_timeframes = ['1m', '5m', '15m']  # 3个高频时间框架
    
    # 运行分析
    results = analyzer.run_intraday_analysis(
        symbols=test_symbols,
        timeframes=test_timeframes
    )
    
    # 保存结果
    output_dir = analyzer.save_intraday_results(results)
    
    print(f"✅ 日内交易分析完成! 结果保存在: {output_dir}")
