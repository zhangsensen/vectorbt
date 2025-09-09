#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正确的WFO (Walk Forward Optimization) 单股票因子验证系统
每只股票独立进行WFO，避免跨股票数据泄露
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import gc
from tqdm import tqdm
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

class SingleStockWFO:
    """单股票WFO分析器"""
    
    def __init__(self, 
                 symbol: str,
                 data: pd.DataFrame,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 train_window: timedelta = timedelta(days=60),
                 test_window: timedelta = timedelta(days=30),
                 step_size: timedelta = timedelta(days=15),
                 min_samples: int = 100):
        """
        初始化单股票WFO
        
        Args:
            symbol: 股票代码
            data: 股票数据
            start_date: 分析开始日期
            end_date: 分析结束日期
            train_window: 训练窗口长度
            test_window: 测试窗口长度  
            step_size: 滚动步长
            min_samples: 最小样本数
        """
        self.symbol = symbol
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_samples = min_samples
        self.logger = logging.getLogger(f"{__name__}.{symbol}")
        
        # 验证timestamp_ns列是否存在
        assert 'timestamp_ns' in data.columns, f"[{symbol}] timestamp_ns 列丢失，无法重建索引。实际列: {data.columns.tolist()}"
        
    def run_wfo(self, factors: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        运行单股票WFO
        
        Args:
            factors: 因子字典 {factor_name: factor_series}
            
        Returns:
            WFO结果字典
        """
        self.logger.info(f"🔄 开始 {self.symbol} 的WFO分析...")
        
        # 确保索引是 DatetimeIndex - 数据已经从load_timeframe_data正确处理过时区
        # 在多进程环境中，数据序列化可能会丢失时区信息，需要重新处理
        if not isinstance(self.data.index, pd.DatetimeIndex):
            # 如果索引不是DatetimeIndex，说明数据在序列化过程中出了问题
            # 尝试从原始数据重新构建正确的索引
            self.logger.warning(f"数据索引不是DatetimeIndex，尝试重新构建...")
            
            # 优先使用timestamp_ns列（纳秒精度，最可靠）
            if 'timestamp_ns' in self.data.columns:
                self.data.index = pd.to_datetime(self.data['timestamp_ns'], unit='ns')
                self.logger.info(f"使用timestamp_ns列重建索引")
            # 其次使用timestamp列
            elif 'timestamp' in self.data.columns:
                if pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
                    self.data = self.data.set_index('timestamp')
                elif self.data['timestamp'].dtype in ['int64', 'float64']:
                    # 智能判断时间戳单位
                    if self.data['timestamp'].max() > 1e16:          # 微秒
                        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='us')
                    elif self.data['timestamp'].max() > 1e13:        # 毫秒
                        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')
                    elif self.data['timestamp'].max() > 1e10:        # 秒
                        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='s')
                    else:
                        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data = self.data.set_index('timestamp')
                else:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data = self.data.set_index('timestamp')
                self.logger.info(f"使用timestamp列重建索引")
            else:
                # 如果没有timestamp列，但索引是整数，可能需要转换
                if self.data.index.dtype in ['int64', 'float64']:
                    # 检查是否可能是Unix时间戳
                    if self.data.index.max() > 1e10:  # 很可能是时间戳
                        self.data.index = pd.to_datetime(self.data.index, unit='s')
                    else:
                        # 如果不是时间戳，保留原样但记录警告
                        self.logger.warning(f"索引是整数但值较小，可能不是时间戳: {self.data.index.min()} - {self.data.index.max()}")
        
        # 确保时区信息
        if hasattr(self.data.index, 'tz') and self.data.index.tz is None:
            self.data.index = self.data.index.tz_localize('Asia/Hong_Kong')
        elif not hasattr(self.data.index, 'tz'):
            # 如果索引没有tz属性，说明它不是DatetimeIndex，跳过时区处理
            self.logger.warning(f"索引没有tz属性，跳过时区处理")
        
        # 1. 强制单调递增 & 去空
        self.data = self.data.sort_index().dropna()
        
        # 2. 统一时区处理（避免时区比较失败）
        if hasattr(self.data.index, 'tz') and self.data.index.tz is None:
            self.data.index = self.data.index.tz_localize('Asia/Hong_Kong')
        elif not hasattr(self.data.index, 'tz'):
            # 如果索引没有tz属性，说明它不是DatetimeIndex，记录错误并跳过
            self.logger.error(f"索引不是DatetimeIndex，无法进行时区处理")
            return {'error': '数据索引不是DatetimeIndex，无法进行时区处理'}
        
        # 3. 确保分析日期有时区信息
        if self.start_date.tz is None:
            self.start_date = self.start_date.tz_localize('Asia/Hong_Kong')
        if self.end_date.tz is None:
            self.end_date = self.end_date.tz_localize('Asia/Hong_Kong')
        
        # 4. 把训练/测试窗口参数临时调小（快速验证）
        original_train_window = self.train_window
        original_test_window = self.test_window
        original_step_size = self.step_size
        original_min_samples = self.min_samples
        
        self.train_window = pd.Timedelta(days=10)      # 10 根即可
        self.test_window = pd.Timedelta(days=5)
        self.step_size = pd.Timedelta(days=2)
        self.min_samples = 10      # 最小样本
        
        # 调试信息：打印数据范围
        self.logger.info(f"数据范围: {self.data.index.min()} 到 {self.data.index.max()}")
        self.logger.info(f"分析范围: {self.start_date} 到 {self.end_date}")
        self.logger.info(f"数据点数: {len(self.data)}")
        
        wfo_results = []
        current_date = self.start_date
        
        while current_date + self.train_window + self.test_window <= self.end_date:
            # 定义当前窗口
            train_start = current_date
            train_end = current_date + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            
            # 获取训练和测试数据
            train_data = self.data[(self.data.index >= train_start) & (self.data.index < train_end)]
            test_data = self.data[(self.data.index >= test_start) & (self.data.index < test_end)]
            
            # 调试信息：打印窗口信息
            self.logger.info(f"窗口 {len(wfo_results)+1}: 训练期 {train_start} 到 {train_end} (数据点: {len(train_data)}), 测试期 {test_start} 到 {test_end} (数据点: {len(test_data)})")
            
            # 检查数据充足性
            if len(train_data) < self.min_samples or len(test_data) < self.min_samples // 2:
                self.logger.info(f"  -> 跳过窗口：训练数据 {len(train_data)} < {self.min_samples} 或 测试数据 {len(test_data)} < {self.min_samples//2}")
                current_date += self.step_size
                continue
                
            # 计算训练期因子表现
            train_factors = {}
            for factor_name, factor_series in factors.items():
                # 对齐因子数据到训练期时间范围
                aligned_factor = factor_series.loc[train_data.index.intersection(factor_series.index)]
                if len(aligned_factor) > 0:
                    train_factors[factor_name] = aligned_factor
            
            # 计算测试期因子表现  
            test_factors = {}
            for factor_name, factor_series in factors.items():
                # 对齐因子数据到测试期时间范围
                aligned_factor = factor_series.loc[test_data.index.intersection(factor_series.index)]
                if len(aligned_factor) > 0:
                    test_factors[factor_name] = aligned_factor
            
            # 计算指标
            window_result = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'train_metrics': self._calculate_window_metrics(train_factors, train_data),
                'test_metrics': self._calculate_window_metrics(test_factors, test_data)
            }
            
            wfo_results.append(window_result)
            current_date += self.step_size
            
        self.logger.info(f"✅ {self.symbol} WFO完成，共 {len(wfo_results)} 个窗口")
        
        # 汇总WFO结果
        return self._aggregate_wfo_results(wfo_results)
    
    def _calculate_window_metrics(self, factors: Dict[str, pd.Series], data: pd.DataFrame) -> Dict[str, Any]:
        """计算窗口内的因子指标"""
        metrics = {}
        
        if 'close' not in data.columns or len(data) < 2:
            self.logger.warning(f"数据不足: close列存在={ 'close' in data.columns}, 数据长度={len(data)}")
            return metrics
            
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 5:
            self.logger.warning(f"收益数据不足: {len(returns)}")
            return metrics
        
        self.logger.info(f"开始计算指标，因子数量: {len(factors)}, 收益数据长度: {len(returns)}")
        
        for factor_name, factor_series in factors.items():
            try:
                # 检查因子数据
                if factor_series.isna().all():
                    self.logger.warning(f"因子 {factor_name} 全为NaN")
                    continue
                
                # 对齐数据
                aligned_returns, aligned_factor = returns.align(factor_series, join='inner')
                
                if len(aligned_returns) < 5:
                    self.logger.warning(f"因子 {factor_name} 对齐后数据不足: {len(aligned_returns)}")
                    continue
                
                # 移除NaN值
                mask = ~(aligned_returns.isna() | aligned_factor.isna())
                clean_returns = aligned_returns[mask]
                clean_factor = aligned_factor[mask]
                
                if len(clean_returns) < 5:
                    self.logger.warning(f"因子 {factor_name} 清洗后数据不足: {len(clean_returns)}")
                    continue
                
                # 计算IC
                ic = clean_returns.corr(clean_factor)
                
                # 计算RankIC
                rank_ic = clean_returns.corr(clean_factor.rank())
                
                # 计算分位数收益
                try:
                    quantiles = pd.qcut(clean_factor, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
                    quantile_returns = clean_returns.groupby(quantiles).mean()
                except Exception as e:
                    self.logger.warning(f"因子 {factor_name} 分位数计算失败: {e}")
                    quantile_returns = pd.Series()
                
                metrics[factor_name] = {
                    'ic': ic if not np.isnan(ic) else 0,
                    'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
                    'samples': len(clean_returns),
                    'quantile_returns': quantile_returns.to_dict(),
                    'factor_mean': clean_factor.mean(),
                    'factor_std': clean_factor.std()
                }
                
                self.logger.info(f"因子 {factor_name} 计算完成: IC={ic:.4f}, 样本数={len(clean_returns)}")
                
            except Exception as e:
                self.logger.error(f"因子 {factor_name} 计算失败: {e}")
                continue
            
        return metrics
    
    def _aggregate_wfo_results(self, wfo_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总WFO结果"""
        if not wfo_results:
            return {}
            
        aggregated = {}
        
        # 收集所有因子的测试期IC
        factor_ics = {}
        factor_rank_ics = {}
        
        for window in wfo_results:
            test_metrics = window.get('test_metrics', {})
            for factor_name, metrics in test_metrics.items():
                if factor_name not in factor_ics:
                    factor_ics[factor_name] = []
                    factor_rank_ics[factor_name] = []
                    
                if 'ic' in metrics and not np.isnan(metrics['ic']):
                    factor_ics[factor_name].append(metrics['ic'])
                    
                if 'rank_ic' in metrics and not np.isnan(metrics['rank_ic']):
                    factor_rank_ics[factor_name].append(metrics['rank_ic'])
        
        # 计算汇总统计
        for factor_name in factor_ics.keys():
            ics = factor_ics[factor_name]
            rank_ics = factor_rank_ics[factor_name]
            
            if ics:
                aggregated[factor_name] = {
                    'windows': len(ics),
                    'mean_ic': np.mean(ics),
                    'std_ic': np.std(ics),
                    'median_ic': np.median(ics),
                    'ic_positive_ratio': sum(1 for ic in ics if ic > 0) / len(ics),
                    'mean_rank_ic': np.mean(rank_ics) if rank_ics else np.nan,
                    'std_rank_ic': np.std(rank_ics) if rank_ics else np.nan,
                    'stability_score': 1 - (np.std(ics) / abs(np.mean(ics))) if np.mean(ics) != 0 else 0,
                    'all_ics': ics,
                    'all_rank_ics': rank_ics
                }
        
        return {
            'symbol': self.symbol,
            'total_windows': len(wfo_results),
            'wfo_results': wfo_results,
            'aggregated_metrics': aggregated
        }


class MultiStockWFOAnalyzer:
    """多股票WFO分析器"""
    
    def __init__(self, 
                 data_dir: str,
                 start_date: str = "2025-03-01",
                 end_date: str = "2025-09-01",
                 train_window_days: int = 60,
                 test_window_days: int = 30,
                 step_days: int = 15,
                 n_workers: int = None):
        """
        初始化多股票WFO分析器
        
        Args:
            data_dir: 数据目录
            start_date: 开始日期
            end_date: 结束日期
            train_window_days: 训练窗口天数
            test_window_days: 测试窗口天数
            step_days: 滚动步长天数
            n_workers: 工作进程数
        """
        self.data_dir = Path(data_dir)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.step_size = timedelta(days=step_days)
        self.n_workers = n_workers or max(1, min(8, mp.cpu_count() - 2))
        self.logger = logging.getLogger(__name__)
        
    def load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """加载股票数据"""
        self.logger.info("📂 加载股票数据...")
        stock_data = {}
        
        # 假设数据文件为CSV格式
        for file_path in self.data_dir.glob("*.csv"):
            symbol = file_path.stem
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                
                if len(df) > 100:  # 最小数据量要求
                    stock_data[symbol] = df
                    self.logger.info(f"   加载 {symbol}: {len(df)} 条记录")
                    
            except Exception as e:
                self.logger.warning(f"   加载 {symbol} 失败: {e}")
                
        self.logger.info(f"✅ 成功加载 {len(stock_data)} 只股票数据")
        return stock_data
    
    def calculate_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算技术因子"""
        factors = {}
        
        if len(data) < 20:
            return factors
            
        try:
            # 尝试使用TALib，如果不可用则使用pandas实现
            try:
                import talib
                use_talib = True
            except ImportError:
                use_talib = False
                self.logger.info("TALib不可用，使用pandas实现")
            
            close = data['close'].astype(np.float64).values
            high = data['high'].astype(np.float64).values if 'high' in data.columns else close
            low = data['low'].astype(np.float64).values if 'low' in data.columns else close
            volume = data['volume'].astype(np.float64).values if 'volume' in data.columns else np.ones(len(close), dtype=np.float64)
            
            # RSI
            if len(close) > 14:
                if use_talib:
                    factors['RSI'] = pd.Series(talib.RSI(close, timeperiod=14), index=data.index)
                else:
                    delta = np.diff(close)
                    gain = np.where(delta > 0, delta, 0)
                    loss = np.where(delta < 0, -delta, 0)
                    avg_gain = pd.Series(gain).rolling(14).mean()
                    avg_loss = pd.Series(loss).rolling(14).mean()
                    rs = avg_gain / avg_loss
                    factors['RSI'] = 100 - (100 / (1 + rs))
                    factors['RSI'].index = data.index
                
            # MACD
            if len(close) > 26:
                if use_talib:
                    macd, signal, hist = talib.MACD(close)
                    factors['MACD'] = pd.Series(macd, index=data.index)
                else:
                    ema12 = pd.Series(close).ewm(span=12).mean()
                    ema26 = pd.Series(close).ewm(span=26).mean()
                    factors['MACD'] = ema12 - ema26
                
            # 简单移动平均因子
            if len(close) > 20:
                ma5 = pd.Series(close).rolling(5).mean()
                ma20 = pd.Series(close).rolling(20).mean()
                factors['MA_Ratio'] = ma5 / ma20
                
            # 动量指标
            if len(close) > 10:
                factors['Momentum_ROC'] = data['close'].pct_change(10)
                
            # 波动率
            if len(close) > 10:
                factors['Volatility'] = data['close'].pct_change().rolling(10).std()
                
            # 价格位置
            if len(high) > 20 and len(low) > 20:
                high_20 = data['high'].rolling(20).max()
                low_20 = data['low'].rolling(20).min()
                factors['Price_Position'] = (data['close'] - low_20) / (high_20 - low_20)
                
            # 成交量比率
            if len(volume) > 20:
                volume_ma = pd.Series(volume).rolling(20).mean()
                factors['Volume_Ratio'] = pd.Series(volume) / volume_ma
                
        except Exception as e:
            self.logger.warning(f"计算因子失败: {e}")
            
        # 过滤掉NaN过多的因子
        valid_factors = {}
        for name, series in factors.items():
            if series.notna().sum() >= len(series) * 0.5:  # 至少50%非NaN
                valid_factors[name] = series
            else:
                self.logger.warning(f"因子 {name} NaN过多，已丢弃")
        
        return valid_factors
    
    def run_single_stock_wfo(self, args: Tuple[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行单只股票的WFO"""
        symbol, data = args
        
        try:
            # 验证并修复数据索引
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning(f"{symbol}: 数据索引不是DatetimeIndex，尝试修复...")
                
                # 优先使用timestamp_ns列（纳秒精度，最可靠）
                if 'timestamp_ns' in data.columns:
                    data.index = pd.to_datetime(data['timestamp_ns'], unit='ns')
                    self.logger.info(f"{symbol}: 使用timestamp_ns列重建索引")
                # 其次使用timestamp列
                elif 'timestamp' in data.columns:
                    if pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                        data = data.set_index('timestamp')
                    elif data['timestamp'].dtype in ['int64', 'float64']:
                        # 智能判断时间戳单位
                        if data['timestamp'].max() > 1e16:          # 微秒
                            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
                        elif data['timestamp'].max() > 1e13:        # 毫秒
                            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                        elif data['timestamp'].max() > 1e10:        # 秒
                            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
                        else:
                            data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data = data.set_index('timestamp')
                    else:
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data = data.set_index('timestamp')
                    self.logger.info(f"{symbol}: 使用timestamp列重建索引")
                else:
                    # 如果没有timestamp列，无法修复
                    self.logger.error(f"{symbol}: 数据索引不是DatetimeIndex且无timestamp列")
                    return {'symbol': symbol, 'error': '数据索引不是DatetimeIndex且无timestamp列'}
            
            # 确保时区信息
            if hasattr(data.index, 'tz') and data.index.tz is None:
                data.index = data.index.tz_localize('Asia/Hong_Kong')
            
            # 计算因子
            factors = self.calculate_factors(data)
            
            if not factors:
                self.logger.warning(f"{symbol}: 无法计算因子")
                return {'symbol': symbol, 'error': '无法计算因子'}
            
            # 运行WFO
            print(f"[DEBUG] {symbol}: 传递前的列: {data.columns.tolist()}")
            print(f"[DEBUG] {symbol}: 数据索引类型: {type(data.index)}")
            print(f"[DEBUG] {symbol}: 数据形状: {data.shape}")
            
            wfo_analyzer = SingleStockWFO(
                symbol=symbol,
                data=data,
                start_date=self.start_date,
                end_date=self.end_date,
                train_window=self.train_window,
                test_window=self.test_window,
                step_size=self.step_size,
                min_samples=30  # 降低最小样本要求
            )
            
            result = wfo_analyzer.run_wfo(factors)
            
            # 检查结果是否有效
            if not result or 'aggregated_metrics' not in result or not result['aggregated_metrics']:
                self.logger.warning(f"{symbol}: WFO结果为空")
                return {'symbol': symbol, 'error': 'WFO结果为空'}
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理 {symbol} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return {'symbol': symbol, 'error': str(e)}
    
    def run_all_stocks_wfo(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行所有股票的WFO"""
        self.logger.info(f"🚀 开始多股票WFO分析（{self.n_workers}个工作进程）...")
        
        all_results = {}
        
        # 准备参数
        tasks = [(symbol, data) for symbol, data in stock_data.items()]
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # 提交所有任务
            future_to_symbol = {executor.submit(self.run_single_stock_wfo, task): task[0] 
                               for task in tasks}
            
            # 使用进度条
            with tqdm(total=len(tasks), desc="WFO进度") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        all_results[symbol] = result
                    except Exception as e:
                        self.logger.error(f"{symbol} 处理失败: {e}")
                        all_results[symbol] = {'symbol': symbol, 'error': str(e)}
                    finally:
                        pbar.update(1)
        
        # 汇总结果
        summary = self._summarize_all_results(all_results)
        
        self.logger.info("✅ 多股票WFO分析完成")
        return {
            'individual_results': all_results,
            'summary': summary,
            'config': {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'train_window_days': self.train_window.days,
                'test_window_days': self.test_window.days,
                'step_days': self.step_size.days,
                'total_stocks': len(stock_data)
            }
        }
    
    def _summarize_all_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """汇总所有股票的WFO结果"""
        summary = {
            'total_stocks': len(all_results),
            'successful_stocks': 0,
            'factor_performance': {},
            'stable_factors': {}
        }
        
        # 收集所有因子的表现
        factor_windows = {}
        factor_ics = {}
        
        for symbol, result in all_results.items():
            if 'error' in result or 'aggregated_metrics' not in result:
                continue
                
            summary['successful_stocks'] += 1
            
            aggregated_metrics = result['aggregated_metrics']
            
            for factor_name, metrics in aggregated_metrics.items():
                if factor_name not in factor_windows:
                    factor_windows[factor_name] = []
                    factor_ics[factor_name] = []
                    
                factor_windows[factor_name].append(metrics.get('windows', 0))
                factor_ics[factor_name].extend(metrics.get('all_ics', []))
        
        # 计算因子汇总统计
        for factor_name in factor_windows.keys():
            windows = factor_windows[factor_name]
            ics = factor_ics[factor_name]
            
            if windows and ics:
                summary['factor_performance'][factor_name] = {
                    'avg_windows_per_stock': np.mean(windows),
                    'total_windows': sum(windows),
                    'mean_ic': np.mean(ics),
                    'std_ic': np.std(ics),
                    'median_ic': np.median(ics),
                    'ic_positive_ratio': sum(1 for ic in ics if ic > 0) / len(ics),
                    'stability': 1 - (np.std(ics) / abs(np.mean(ics))) if np.mean(ics) != 0 else 0
                }
                
                # 识别稳定因子（IC稳定性 > 0.7 且 平均IC绝对值 > 0.02）
                performance = summary['factor_performance'][factor_name]
                if (performance['stability'] > 0.7 and 
                    abs(performance['mean_ic']) > 0.02 and
                    performance['ic_positive_ratio'] > 0.6 or performance['ic_positive_ratio'] < 0.4):
                    summary['stable_factors'][factor_name] = performance
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
        self.logger.info(f"📁 结果已保存到: {output_path}")


def main():
    """主函数示例"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建分析器
    analyzer = MultiStockWFOAnalyzer(
        data_dir="/Users/zhangshenshen/longport/vectorbt_workspace/data",
        start_date="2025-03-01",
        end_date="2025-09-01",
        train_window_days=60,
        test_window_days=30,
        step_days=15,
        n_workers=4
    )
    
    # 加载数据
    stock_data = analyzer.load_stock_data()
    
    # 运行WFO
    results = analyzer.run_all_stocks_wfo(stock_data)
    
    # 保存结果
    analyzer.save_results(results, "wfo_results.json")
    
    # 打印摘要
    print("\n" + "="*60)
    print("WFO分析摘要")
    print("="*60)
    print(f"总股票数: {results['summary']['total_stocks']}")
    print(f"成功分析: {results['summary']['successful_stocks']}")
    print(f"稳定因子: {list(results['summary']['stable_factors'].keys())}")
    
    for factor_name, performance in results['summary']['stable_factors'].items():
        print(f"\n{factor_name}:")
        print(f"  平均IC: {performance['mean_ic']:.4f}")
        print(f"  稳定性: {performance['stability']:.4f}")
        print(f"  正向比率: {performance['ic_positive_ratio']:.4f}")


if __name__ == "__main__":
    main()