#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于vectorbt的港股技术因子探索与WFO验证系统
集成大规模因子探索和Walk Forward Optimization验证
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import talib
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import os
import sys
import gc
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

def _calculate_price_position_safe(high, low, close, timeperiod=20):
    """
    安全的价格位置计算，避免前视偏差
    使用滞后一个周期的SMA值确保只用历史数据
    """
    # 计算SMA
    sma_high = talib.SMA(high, timeperiod=timeperiod)
    sma_low = talib.SMA(low, timeperiod=timeperiod)
    
    # 创建滞后版本避免前视偏差
    sma_high_lagged = np.full_like(sma_high, np.nan)
    sma_low_lagged = np.full_like(sma_low, np.nan)
    
    # 从第timeperiod+1个位置开始填充滞后值
    sma_high_lagged[timeperiod:] = sma_high[timeperiod-1:-1]
    sma_low_lagged[timeperiod:] = sma_low[timeperiod-1:-1]
    
    # 计算价格位置
    position = (close - sma_low_lagged) / (sma_high_lagged - sma_low_lagged)
    
    # 处理除零和边界情况
    position = np.where(
        (sma_high_lagged - sma_low_lagged) > 1e-8,
        position,
        0.5  # 当范围太小时，设为中间值
    )
    
    # 限制在合理范围内
    position = np.clip(position, 0, 1)
    
    return position

# 导入WFO系统
from core.single_stock_wfo import SingleStockWFO, MultiStockWFOAnalyzer

# 设置日志
def setup_logging():
    """设置日志系统"""
    current_dir = Path(__file__).parent
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "vectorbt_wfo_analysis_{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_file

logger, log_file = setup_logging()

class VectorbtWFOAnalyzer:
    """基于vectorbt的因子探索与WFO验证分析器"""
    
    def __init__(self, 
                 data_dir="/Users/zhangshenshen/longport/vectorbt_workspace/data",
                 output_dir=None):
        """
        初始化分析器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        
        # 港股交易成本设置（详细拆分）
        self.hk_transaction_cost = {
            'commission': 0.0003,     # 佣金0.03%（双向）
            'slippage': 0.002,       # 滑点0.2%（双向）
            'stamp_duty': 0.0013,    # 印花税0.13%（仅卖出）
            'trading_fee': 0.00005,  # 交易费0.005%（双向）
            'ccass_fee': 0.0002,     # 结算费0.02%（双向）
            'levy': 0.000027,        # 征费0.0027%（双向）
        }
        
        # 计算总交易成本
        self.total_buy_cost = (self.hk_transaction_cost['commission'] + 
                              self.hk_transaction_cost['slippage'] + 
                              self.hk_transaction_cost['trading_fee'] + 
                              self.hk_transaction_cost['ccass_fee'] + 
                              self.hk_transaction_cost['levy'])
        
        self.total_sell_cost = (self.total_buy_cost + 
                               self.hk_transaction_cost['stamp_duty'])
        
        self.total_round_trip_cost = self.total_buy_cost + self.total_sell_cost  # 约0.60%
        
        # 分析时间范围
        self.start_date = pd.to_datetime("2025-03-01")
        self.end_date = pd.to_datetime("2025-09-01")
        
        # 时间框架
        self.timeframes = ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '3h', '4h', '1d']
        
        # 因子定义 - 每个因子独立探测和验证
        self.factor_definitions = self._create_factor_definitions()
        
        # WFO配置
        self.wfo_config = {
            'train_window_days': 60,
            'test_window_days': 30,
            'step_days': 15,
            'min_samples': 50,
            'n_workers': max(1, min(8, cpu_count() - 2))
        }
        
        # vectorbt回测配置
        self.vbt_config = {
            'init_cash': 1_000_000,
            'freq': None,  # 自动根据时间框架设置
            'direction': 'longonly',  # 先测试多头
            'slippage': self.hk_transaction_cost['slippage'],
            'commission': self.hk_transaction_cost['commission'],
        }
        
        self.logger = logger
    
    def _create_factor_definitions(self) -> Dict[str, Dict]:
        """创建因子定义"""
        return {
            # 趋势因子
            'RSI': {
                'function': lambda close: talib.RSI(close, timeperiod=14),
                'direction': 'negative',  # 超买信号
                'params': {'timeperiod': [7, 14, 21]},
                'category': 'momentum'
            },
            'MACD': {
                'function': lambda close: talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0],
                'direction': 'neutral',
                'params': {'fastperiod': [8, 12, 16], 'slowperiod': [20, 26, 32]},
                'category': 'trend'
            },
            'Stochastic': {
                'function': lambda high, low, close: talib.STOCH(high, low, close, fastk_period=14)[0],
                'direction': 'mean_reverting',
                'params': {'fastk_period': [9, 14, 19]},
                'category': 'momentum'
            },
            'Williams_R': {
                'function': lambda high, low, close: talib.WILLR(high, low, close, timeperiod=14),
                'direction': 'negative',
                'params': {'timeperiod': [10, 14, 18]},
                'category': 'momentum'
            },
            
            # 均值回归因子
            'Bollinger_Position': {
                'function': lambda close: (close - talib.BBANDS(close, timeperiod=20)[0]) / (talib.BBANDS(close, timeperiod=20)[2] - talib.BBANDS(close, timeperiod=20)[0]),
                'direction': 'mean_reverting',
                'params': {'timeperiod': [15, 20, 25]},
                'category': 'mean_reverting'
            },
            'Z_Score': {
                'function': lambda close: (close - talib.SMA(close, timeperiod=20)) / talib.STDDEV(close, timeperiod=20, nbdev=1),
                'direction': 'mean_reverting',
                'params': {'timeperiod': [15, 20, 25]},
                'category': 'mean_reverting'
            },
            'CCI': {
                'function': lambda high, low, close: talib.CCI(high, low, close, timeperiod=14),
                'direction': 'mean_reverting',
                'params': {'timeperiod': [10, 14, 18]},
                'category': 'mean_reverting'
            },
            
            # 波动率因子
            'ADX': {
                'function': lambda high, low, close: talib.ADX(high, low, close, timeperiod=14),
                'direction': 'neutral',
                'params': {'timeperiod': [10, 14, 18]},
                'category': 'volatility'
            },
            'ATR': {
                'function': lambda high, low, close: talib.ATR(high, low, close, timeperiod=14),
                'direction': 'neutral',
                'params': {'timeperiod': [10, 14, 18]},
                'category': 'volatility'
            },
            
            # 成交量因子
            'Volume_Ratio': {
                'function': lambda close, volume: volume / talib.SMA(volume.astype(float), timeperiod=20),
                'direction': 'neutral',
                'params': {'timeperiod': [15, 20, 25]},
                'category': 'volume'
            },
            'Volume_OSC': {
                'function': lambda volume: (talib.SMA(volume.astype(float), timeperiod=5) - talib.SMA(volume.astype(float), timeperiod=20)) / talib.SMA(volume.astype(float), timeperiod=20),
                'direction': 'neutral',
                'params': {'short_period': [3, 5, 7], 'long_period': [15, 20, 25]},
                'category': 'volume'
            },
            
            # 动量因子
            'Momentum_ROC': {
                'function': lambda close: talib.ROC(close, timeperiod=10),
                'direction': 'neutral',
                'params': {'timeperiod': [5, 10, 15]},
                'category': 'momentum'
            },
            'Momentum': {
                'function': lambda close: close / talib.SMA(close, timeperiod=10) - 1,
                'direction': 'neutral',
                'params': {'timeperiod': [5, 10, 15]},
                'category': 'momentum'
            },
            
            # 价格位置因子
            'Price_Position': {
                'function': lambda high, low, close, timeperiod=20: _calculate_price_position_safe(high, low, close, timeperiod),
                'direction': 'neutral',
                'params': {'timeperiod': [15, 20, 25]},
                'category': 'position'
            }
        }
    
    def load_timeframe_data(self, timeframe: str, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """加载时间框架数据"""
        timeframe_dir = self.data_dir / timeframe
        data = {}
        
        if not timeframe_dir.exists():
            self.logger.warning(f"时间框架目录不存在: {timeframe}")
            return data
        
        # 如果没有指定股票，加载所有股票
        if symbols is None:
            symbol_files = list(timeframe_dir.glob("*.parquet"))
            symbols = [f.stem for f in symbol_files if f.stem.endswith('.HK')]
        
        loaded_count = 0
        for symbol in symbols:
            file_path = timeframe_dir / f"{symbol}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    
                    # 修复Unix时间戳转换问题
                    # 数据格式：整数索引 + timestamp列
                    if 'timestamp' in df.columns:
                        # 如果timestamp列已经是datetime类型，直接使用
                        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                            df = df.set_index('timestamp')
                        # 如果timestamp列是整数类型，需要转换
                        elif df['timestamp'].dtype in ['int64', 'float64']:
                            # 自动判断是秒、毫秒还是微秒
                            if df['timestamp'].max() > 1e16:          # 微秒 (>= 2286-11-20)
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
                            elif df['timestamp'].max() > 1e13:        # 毫秒 (>= 1973-09-27)
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            elif df['timestamp'].max() > 1e10:        # 秒 (>= 2286-11-20)
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                            else:
                                # 如果值很小，尝试直接转换
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.set_index('timestamp')
                        else:
                            # 其他类型，尝试直接转换
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.set_index('timestamp')
                    else:
                        # 如果没有timestamp列，检查索引是否为datetime
                        if pd.api.types.is_datetime64_any_dtype(df.index):
                            pass  # 索引已经是datetime
                        elif df.index.dtype in ['int64', 'float64']:
                            # 只有在没有timestamp列且索引为整数时才尝试转换索引
                            # 自动判断是秒、毫秒还是微秒
                            if df.index.max() > 1e16:          # 微秒 (>= 2286-11-20)
                                df.index = pd.to_datetime(df.index, unit='us')
                            elif df.index.max() > 1e13:        # 毫秒 (>= 1973-09-27)
                                df.index = pd.to_datetime(df.index, unit='ms')
                            elif df.index.max() > 1e10:        # 秒 (>= 2286-11-20)
                                df.index = pd.to_datetime(df.index, unit='s')
                            else:
                                # 如果值很小，保留原样但记录警告
                                self.logger.warning(f"{symbol}: 索引值很小 ({df.index.min()} - {df.index.max()}), 可能不是Unix时间戳")
                                pass
                        else:
                            # 尝试直接转换索引
                            try:
                                df.index = pd.to_datetime(df.index)
                            except Exception as e:
                                self.logger.warning(f"{symbol}: 无法转换索引为datetime: {e}")
                                continue
                    
                    # 保留timestamp列以防止多进程传递中索引丢失
                    if isinstance(df.index, pd.DatetimeIndex) and 'timestamp' not in df.columns:
                        df['timestamp'] = df.index.copy()
                    
                    # 3. 确保时区设置
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('Asia/Hong_Kong')
                    
                    # 4. 确保比较的日期有时区信息
                    start_date = self.start_date
                    end_date = self.end_date
                    
                    # 统一到香港时区进行比较
                    if start_date.tz is None:
                        start_date = start_date.tz_localize('Asia/Hong_Kong')
                    else:
                        start_date = start_date.tz_convert('Asia/Hong_Kong')
                    
                    if end_date.tz is None:
                        end_date = end_date.tz_localize('Asia/Hong_Kong')
                    else:
                        end_date = end_date.tz_convert('Asia/Hong_Kong')
                    
                    # 过滤时间范围
                    df = df.loc[start_date:end_date]
                    
                    # 数据质量检查
                    if self._check_data_quality(df, timeframe):
                        data[symbol] = df
                        loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"加载{symbol}失败: {e}")
        
        self.logger.info(f"加载{timeframe}数据: {loaded_count}/{len(symbols)}只股票")
        return data
    
    def _check_data_quality(self, df: pd.DataFrame, timeframe: str) -> bool:
        """检查数据质量"""
        if df.empty:
            return False
        
        # 检查必要列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # 检查最小数据量
        min_samples = {
            '1m': 1000, '3m': 500, '5m': 300, '10m': 200, '15m': 150,
            '30m': 100, '1h': 80, '2h': 60, '3h': 50, '4h': 40, '1d': 30
        }
        
        if len(df) < min_samples.get(timeframe, 50):
            return False
        
        # 检查数据合理性
        if (df['close'] <= 0).any() or (df['volume'] < 0).any():
            return False
        
        return True
    
    def calculate_factor(self, factor_name: str, factor_def: Dict, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """计算单个因子"""
        factor_data = []
        
        for symbol, df in data.items():
            try:
                # 准备数据 - 确保数据类型为float64
                close = df['close'].astype(np.float64).values
                high = df['high'].astype(np.float64).values
                low = df['low'].astype(np.float64).values
                volume = df['volume'].astype(np.float64).values
                
                # 计算因子值
                if factor_name in ['Volume_Ratio', 'Volume_OSC']:
                    factor_values = factor_def['function'](close, volume)
                elif factor_name in ['RSI', 'MACD', 'Momentum_ROC', 'Momentum']:
                    factor_values = factor_def['function'](close)
                else:
                    factor_values = factor_def['function'](high, low, close)
                
                # 创建MultiIndex Series
                timestamps = df.index[:len(factor_values)]
                multi_index = pd.MultiIndex.from_product([[symbol], timestamps], names=['symbol', 'timestamp'])
                factor_series = pd.Series(factor_values, index=multi_index, name=factor_name)
                
                # 清理数据
                factor_series = factor_series.dropna()
                if len(factor_series) > 0:
                    factor_data.append(factor_series)
                    
            except Exception as e:
                self.logger.warning(f"计算{symbol}的{factor_name}因子失败: {e}")
                continue
        
        if factor_data:
            return pd.concat(factor_data)
        else:
            return pd.Series(dtype=float)
    
    def vectorbt_backtest_factor(self, factor_name: str, factor_values: pd.Series, 
                                data: Dict[str, pd.DataFrame], timeframe: str) -> Dict[str, Any]:
        """使用vectorbt对单个因子进行回测"""
        try:
            # 设置频率
            freq_map = {
                '1m': '1T', '3m': '3T', '5m': '5T', '10m': '10T', '15m': '15T',
                '30m': '30T', '1h': '1H', '2h': '2H', '3h': '3H', '4h': '4H', '1d': '1D'
            }
            freq = freq_map.get(timeframe, '1D')
            
            # 为每只股票创建回测
            portfolio_results = {}
            
            for symbol in factor_values.index.get_level_values(0).unique():
                try:
                    # 获取该股票的因子值和价格数据
                    symbol_factor = factor_values.loc[symbol]
                    symbol_data = data[symbol]
                    
                    # 对齐数据
                    common_index = symbol_factor.index.intersection(symbol_data.index)
                    if len(common_index) < 10:
                        continue
                    
                    aligned_factor = symbol_factor.loc[common_index]
                    aligned_close = symbol_data.loc[common_index, 'close']
                    
                    # 生成信号（简单的阈值策略）
                    factor_def = self.factor_definitions[factor_name]
                    direction = factor_def['direction']
                    
                    if direction == 'negative':
                        # 负向因子：低值买入，高值卖出
                        signals = pd.Series(0, index=aligned_factor.index)
                        signals[aligned_factor < aligned_factor.quantile(0.3)] = 1  # 买入
                        signals[aligned_factor > aligned_factor.quantile(0.7)] = -1  # 卖出
                    elif direction == 'mean_reverting':
                        # 均值回归：极值买入/卖出
                        signals = pd.Series(0, index=aligned_factor.index)
                        signals[aligned_factor < aligned_factor.quantile(0.2)] = 1
                        signals[aligned_factor > aligned_factor.quantile(0.8)] = -1
                    else:
                        # 正向因子：高值买入，低值卖出
                        signals = pd.Series(0, index=aligned_factor.index)
                        signals[aligned_factor > aligned_factor.quantile(0.7)] = 1
                        signals[aligned_factor < aligned_factor.quantile(0.3)] = -1
                    
                    # 创建价格数据（使用vectorbt的正确API）
                    price_data = aligned_close.rename('price')
                    
                    # 执行回测
                    try:
                        portfolio = vbt.Portfolio.from_signals(
                            close=price_data,
                            entries=(signals == 1),
                            exits=(signals == -1),
                            init_cash=self.vbt_config['init_cash'],
                            freq=freq,
                            direction=self.vbt_config['direction'],
                            slippage=self.vbt_config['slippage'],
                            fees=self.vbt_config['commission']  # 修正参数名
                        )
                    except Exception as e:
                        self.logger.warning(f"回测{symbol}的{factor_name}失败: {e}")
                        continue
                    
                    # 计算指标
                    try:
                        trades = portfolio.trades  # 修复: 去掉括号，VectorBT ≥0.24 API变化
                        
                        # 防御式取值：防止空交易导致的错误
                        if len(trades) == 0:
                            stats = {
                                'total_return': 0.0,
                                'sharpe_ratio': 0.0,
                                'max_drawdown': 0.0,
                                'win_rate': 0.0,
                                'trades_count': 0,
                                'expectancy': 0.0
                            }
                        else:
                            win_rate = trades.win_rate()  # 修复: 需要调用方法
                            expectancy = trades.expectancy()  # 修复: 从trades对象获取，不是portfolio
                            stats = {
                                'total_return': float(portfolio.total_return()),
                                'sharpe_ratio': float(portfolio.sharpe_ratio()),
                                'max_drawdown': float(portfolio.max_drawdown()),
                                'win_rate': float(win_rate),
                                'trades_count': int(len(trades)),
                                'expectancy': float(expectancy)
                            }
                    except Exception as e:
                        self.logger.warning(f"计算{symbol}的{factor_name}指标失败: {e}")
                        continue
                    
                    portfolio_results[symbol] = stats
                    
                except Exception as e:
                    self.logger.warning(f"回测{symbol}的{factor_name}失败: {e}")
                    continue
            
            # 汇总结果
            if portfolio_results:
                aggregate_stats = {
                    'n_stocks': len(portfolio_results),
                    'avg_total_return': np.mean([s['total_return'] for s in portfolio_results.values()]),
                    'avg_sharpe_ratio': np.mean([s['sharpe_ratio'] for s in portfolio_results.values()]),
                    'avg_max_drawdown': np.mean([s['max_drawdown'] for s in portfolio_results.values()]),
                    'avg_win_rate': np.mean([s['win_rate'] for s in portfolio_results.values()]),
                    'total_trades': sum([s['trades_count'] for s in portfolio_results.values()]),
                    'stock_results': portfolio_results
                }
                return aggregate_stats
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"vectorbt回测{factor_name}失败: {e}")
            return {}
    
    def run_wfo_validation(self, factor_name: str, factor_values: pd.Series, 
                         data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行WFO验证"""
        try:
            # 准备WFO数据格式
            wfo_data = {}
            for symbol in data.keys():
                if symbol in factor_values.index.get_level_values(0):
                    symbol_factor = factor_values.loc[symbol]
                    # 将因子值添加到数据中
                    wfo_df = data[symbol].copy()
                    # 确保索引是datetime类型
                    if not isinstance(wfo_df.index, pd.DatetimeIndex):
                        # 如果原始数据有timestamp列，使用它来重建索引
                        if 'timestamp' in wfo_df.columns:
                            if pd.api.types.is_datetime64_any_dtype(wfo_df['timestamp']):
                                wfo_df = wfo_df.set_index('timestamp')
                            elif wfo_df['timestamp'].dtype in ['int64', 'float64']:
                                # 智能判断时间戳单位
                                if wfo_df['timestamp'].max() > 1e16:          # 微秒
                                    wfo_df['timestamp'] = pd.to_datetime(wfo_df['timestamp'], unit='us')
                                elif wfo_df['timestamp'].max() > 1e13:        # 毫秒
                                    wfo_df['timestamp'] = pd.to_datetime(wfo_df['timestamp'], unit='ms')
                                elif wfo_df['timestamp'].max() > 1e10:        # 秒
                                    wfo_df['timestamp'] = pd.to_datetime(wfo_df['timestamp'], unit='s')
                                else:
                                    wfo_df['timestamp'] = pd.to_datetime(wfo_df['timestamp'])
                                wfo_df = wfo_df.set_index('timestamp')
                            else:
                                wfo_df['timestamp'] = pd.to_datetime(wfo_df['timestamp'])
                                wfo_df = wfo_df.set_index('timestamp')
                        else:
                            # 如果没有timestamp列，假设索引已经是正确的
                            pass
                    
                    # 确保时区信息
                    if hasattr(wfo_df.index, 'tz') and wfo_df.index.tz is None:
                        wfo_df.index = wfo_df.index.tz_localize('Asia/Hong_Kong')
                    
                    # 确保保留timestamp列以防止多进程传递中索引丢失
                    if isinstance(wfo_df.index, pd.DatetimeIndex):
                        # 将datetime索引转换为timestamp列，确保多进程安全
                        wfo_df['timestamp'] = wfo_df.index.copy()
                        # 转换为Unix时间戳（纳秒精度）以确保序列化安全
                        wfo_df['timestamp_ns'] = wfo_df.index.astype('int64')
                    elif 'timestamp' not in wfo_df.columns:
                        # 如果索引不是datetime，创建一个时间戳列
                        wfo_df['timestamp'] = pd.date_range(start=self.start_date, periods=len(wfo_df), freq='D')
                        wfo_df['timestamp_ns'] = wfo_df['timestamp'].astype('int64')
                    
                    # 添加因子值
                    wfo_df[factor_name] = symbol_factor.reindex(wfo_df.index)
                    wfo_data[symbol] = wfo_df
            
            # 创建WFO分析器
            wfo_analyzer = MultiStockWFOAnalyzer(
                data_dir="",  # 我们直接传递数据
                start_date=self.start_date.strftime('%Y-%m-%d'),
                end_date=self.end_date.strftime('%Y-%m-%d'),
                train_window_days=self.wfo_config['train_window_days'],
                test_window_days=self.wfo_config['test_window_days'],
                step_days=self.wfo_config['step_days'],
                n_workers=self.wfo_config['n_workers']
            )
            
            # 直接运行WFO（跳过数据加载）
            wfo_results = wfo_analyzer.run_all_stocks_wfo(wfo_data)
            
            return wfo_results
            
        except Exception as e:
            self.logger.error(f"WFO验证{factor_name}失败: {e}")
            return {}
    
    def calculate_ic(self, factor_values: pd.Series, forward_returns: pd.Series) -> float:
        """
        计算因子IC值（Spearman秩相关系数）
        
        Args:
            factor_values: 因子值 (Series)
            forward_returns: 未来收益 (Series)
        
        Returns:
            float: IC值
        """
        # 对齐数据
        aligned_data = pd.concat([factor_values, forward_returns], axis=1).dropna()
        if len(aligned_data) < 10:
            return np.nan
        
        # 计算Spearman秩相关系数
        try:
            ic = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1], method='spearman')
            return ic if not np.isnan(ic) else np.nan
        except:
            return np.nan
    
    def calculate_factor_ic(self, factor_name: str, factor_values: pd.DataFrame, data: Dict[str, pd.DataFrame], timeframe: str) -> Dict[str, Dict]:
        """
        计算因子在所有股票上的IC值
        
        Args:
            factor_name: 因子名称
            factor_values: 因子值 (MultiIndex DataFrame)
            data: 股票数据字典
            timeframe: 时间框架
        
        Returns:
            Dict: 每个股票的IC结果
        """
        ic_results = {}
        
        # 确定前瞻期（根据时间框架）
        forward_periods = {
            '1m': 5, '3m': 3, '5m': 3, '10m': 2, '15m': 2,
            '30m': 1, '1h': 1, '2h': 1, '3h': 1, '4h': 1, '1d': 1
        }
        forward_period = forward_periods.get(timeframe, 1)
        
        for symbol in factor_values.index.get_level_values(0).unique():
            try:
                # 获取该股票的因子值和价格数据
                symbol_factor = factor_values.loc[symbol]
                symbol_data = data[symbol]
                
                # 对齐数据
                common_index = symbol_factor.index.intersection(symbol_data.index)
                if len(common_index) < 10:
                    continue
                
                aligned_factor = symbol_factor.loc[common_index]
                aligned_close = symbol_data.loc[common_index, 'close']
                
                # 计算未来收益
                forward_returns = aligned_close.pct_change(forward_period).shift(-forward_period)
                
                # 计算IC
                ic = self.calculate_ic(aligned_factor, forward_returns)
                
                if not np.isnan(ic):
                    ic_results[symbol] = {
                        'ic': ic,
                        'n_samples': len(aligned_factor.dropna()),
                        'forward_period': forward_period,
                        'mean_factor': float(aligned_factor.mean()),
                        'std_factor': float(aligned_factor.std())
                    }
                    
            except Exception as e:
                self.logger.warning(f"计算{symbol}的{factor_name}的IC失败: {e}")
                continue
        
        return ic_results
    
    def analyze_single_factor(self, factor_name: str, timeframe: str) -> Dict[str, Any]:
        """分析单个因子：vectorbt回测 + WFO验证"""
        self.logger.info(f"开始分析{timeframe}时间框架的{factor_name}因子...")
        
        # 1. 加载数据
        data = self.load_timeframe_data(timeframe)
        if not data:
            return {'error': '无法加载数据'}
        
        # 1.1 优化内存使用
        self.logger.info(f"优化{timeframe}数据内存使用...")
        data = self.optimize_memory_usage(data)
        
        # 2. 计算因子
        factor_def = self.factor_definitions[factor_name]
        factor_values = self.calculate_factor(factor_name, factor_def, data)
        if factor_values.empty:
            return {'error': '因子计算失败'}
        
        # 3. IC计算
        self.logger.info(f"计算{factor_name}的IC值...")
        ic_results = self.calculate_factor_ic(factor_name, factor_values, data, timeframe)
        
        # 4. vectorbt回测
        self.logger.info(f"执行{factor_name}的vectorbt回测...")
        vbt_results = self.vectorbt_backtest_factor(factor_name, factor_values, data, timeframe)
        
        # 5. 因子信号分析
        self.logger.info(f"分析{factor_name}的信号质量...")
        signal_analysis = self.analyze_factor_signals(factor_values, data)
        
        # 6. WFO验证
        self.logger.info(f"执行{factor_name}的WFO验证...")
        wfo_results = self.run_wfo_validation(factor_name, factor_values, data)
        
        # 7. 整合结果
        # 监控内存使用
        self.monitor_memory_usage(threshold_percent=75.0)
        
        # 计算IC统计
        ic_stats = {}
        if ic_results:
            ic_values = [r['ic'] for r in ic_results.values()]
            ic_stats = {
                'mean_ic': np.mean(ic_values),
                'median_ic': np.median(ic_values),
                'std_ic': np.std(ic_values),
                'min_ic': np.min(ic_values),
                'max_ic': np.max(ic_values),
                'positive_ratio': sum(1 for ic in ic_values if ic > 0) / len(ic_values),
                'significant_ratio': sum(1 for ic in ic_values if abs(ic) > 0.1) / len(ic_values),
                'n_stocks_with_ic': len(ic_results)
            }
        
        factor_analysis = {
            'factor_name': factor_name,
            'timeframe': timeframe,
            'factor_category': factor_def['category'],
            'factor_direction': factor_def['direction'],
            'data_summary': {
                'n_stocks': len(data),
                'factor_data_points': len(factor_values),
                'date_range': {
                    'start': self.start_date.strftime('%Y-%m-%d'),
                    'end': self.end_date.strftime('%Y-%m-%d')
                }
            },
            'ic_results': ic_results,
            'ic_stats': ic_stats,
            'vectorbt_results': vbt_results,
            'signal_analysis': signal_analysis,
            'wfo_results': wfo_results,
            'robustness_score': self._calculate_robustness_score(vbt_results, wfo_results, factor_name, ic_stats, signal_analysis),
            'analysis_time': datetime.now().isoformat()
        }
        
        return factor_analysis
    
    def _calculate_robustness_score(self, vbt_results: Dict, wfo_results: Dict, factor_name: str, ic_stats: Dict = None, signal_analysis: Dict = None) -> float:
        """计算因子稳健性得分"""
        try:
            score = 0.0
            
            # IC得分 (0-30分)
            if ic_stats:
                mean_ic = abs(ic_stats.get('mean_ic', 0))
                positive_ratio = ic_stats.get('positive_ratio', 0.5)
                significant_ratio = ic_stats.get('significant_ratio', 0)
                
                ic_score = min(15, mean_ic * 100)  # IC值得分
                positive_score = positive_ratio * 10  # 正向比率得分
                significant_score = significant_ratio * 5  # 显著性得分
                
                score += ic_score + positive_score + significant_score
            
            # vectorbt表现得分 (0-40分)
            if vbt_results:
                avg_return = vbt_results.get('avg_total_return', 0)
                avg_sharpe = vbt_results.get('avg_sharpe_ratio', 0)
                avg_drawdown = vbt_results.get('avg_max_drawdown', 1)
                
                return_score = max(0, min(20, avg_return * 100))
                sharpe_score = max(0, min(10, avg_sharpe * 10))
                drawdown_score = max(0, min(10, (1 - avg_drawdown) * 10))
                
                score += return_score + sharpe_score + drawdown_score
            
            # WFO稳健性得分 (0-30分)
            if wfo_results and 'summary' in wfo_results:
                stable_factors = wfo_results['summary'].get('stable_factors', {})
                if factor_name in stable_factors:
                    factor_stats = stable_factors[factor_name]
                    stability = factor_stats.get('stability', 0)
                    mean_ic = abs(factor_stats.get('mean_ic', 0))
                    positive_ratio = factor_stats.get('ic_positive_ratio', 0.5)
                    
                    stability_score = stability * 15
                    ic_score = min(10, mean_ic * 100)
                    consistency_score = positive_ratio * 5
                    
                    score += stability_score + ic_score + consistency_score
            
            # 信号质量得分 (0-10分)
            if signal_analysis and 'aggregated_stats' in signal_analysis:
                signal_quality = signal_analysis['aggregated_stats'].get('signal_quality_score', 0)
                signal_score = signal_quality * 0.1  # 转换为0-10分
                score += signal_score
            
            return min(100, max(0, score))
            
        except Exception as e:
            self.logger.warning(f"计算稳健性得分失败: {e}")
            return 0.0
    
    def analyze_all_timeframes(self, factor_name: str) -> Dict[str, Any]:
        """分析所有时间框架的单一因子"""
        self.logger.info(f"开始分析{factor_name}在所有时间框架的表现...")
        
        timeframe_results = {}
        
        for timeframe in self.timeframes:
            try:
                result = self.analyze_single_factor(factor_name, timeframe)
                if 'error' not in result:
                    timeframe_results[timeframe] = result
                    self.logger.info(f"{factor_name} {timeframe}分析完成")
                else:
                    self.logger.warning(f"{factor_name} {timeframe}分析失败: {result['error']}")
            except Exception as e:
                self.logger.error(f"{factor_name} {timeframe}分析异常: {e}")
                continue
            
            # 清理内存
            gc.collect()
        
        # 汇总所有时间框架结果
        summary = self._summarize_timeframe_results(factor_name, timeframe_results)
        
        return {
            'factor_name': factor_name,
            'factor_definition': self.factor_definitions[factor_name],
            'timeframe_results': timeframe_results,
            'summary': summary,
            'analysis_time': datetime.now().isoformat()
        }
    
    def _summarize_timeframe_results(self, factor_name: str, 
                                   timeframe_results: Dict[str, Dict]) -> Dict[str, Any]:
        """汇总时间框架结果"""
        try:
            # 统计表现最好的时间框架
            best_timeframes = []
            for timeframe, result in timeframe_results.items():
                robustness = result.get('robustness_score', 0)
                best_timeframes.append((timeframe, robustness))
            
            best_timeframes.sort(key=lambda x: x[1], reverse=True)
            
            # 计算平均表现
            avg_scores = []
            vbt_performances = []
            wfo_stabilities = []
            
            for result in timeframe_results.values():
                avg_scores.append(result.get('robustness_score', 0))
                
                vbt_res = result.get('vectorbt_results', {})
                if vbt_res:
                    vbt_performances.append(vbt_res.get('avg_total_return', 0))
                
                wfo_res = result.get('wfo_results', {})
                if wfo_res and 'summary' in wfo_res:
                    stable_factors = wfo_res['summary'].get('stable_factors', {})
                    if factor_name in stable_factors:
                        wfo_stabilities.append(stable_factors[factor_name].get('stability', 0))
            
            return {
                'total_timeframes': len(timeframe_results),
                'best_timeframes': best_timeframes[:5],  # 前5名
                'avg_robustness_score': np.mean(avg_scores) if avg_scores else 0,
                'avg_vbt_return': np.mean(vbt_performances) if vbt_performances else 0,
                'avg_wfo_stability': np.mean(wfo_stabilities) if wfo_stabilities else 0,
                'performance_distribution': {
                    'excellent': len([s for s in avg_scores if s >= 80]),
                    'good': len([s for s in avg_scores if 60 <= s < 80]),
                    'moderate': len([s for s in avg_scores if 40 <= s < 60]),
                    'poor': len([s for s in avg_scores if s < 40])
                }
            }
            
        except Exception as e:
            self.logger.error(f"汇总时间框架结果失败: {e}")
            return {}
    
    def run_comprehensive_analysis(self, target_factors: List[str] = None) -> Dict[str, Any]:
        """运行综合分析：所有因子在所有时间框架的探测和验证"""
        self.logger.info("开始综合因子分析...")
        
        if target_factors is None:
            target_factors = list(self.factor_definitions.keys())
        
        comprehensive_results = {}
        
        for i, factor_name in enumerate(target_factors, 1):
            self.logger.info(f"分析进度: {i}/{len(target_factors)} - {factor_name}")
            
            try:
                # 分析所有时间框架
                factor_result = self.analyze_all_timeframes(factor_name)
                comprehensive_results[factor_name] = factor_result
                
                # 保存中间结果
                self._save_intermediate_result(factor_name, factor_result)
                
            except Exception as e:
                self.logger.error(f"分析{factor_name}失败: {e}")
                continue
            
            # 定期清理内存
            if i % 3 == 0:
                gc.collect()
        
        # 生成最终报告
        final_report = self._generate_final_report(comprehensive_results)
        
        return {
            'comprehensive_results': comprehensive_results,
            'final_report': final_report,
            'analysis_config': {
                'target_factors': target_factors,
                'timeframes': self.timeframes,
                'wfo_config': self.wfo_config,
                'transaction_costs': self.hk_transaction_cost
            },
            'analysis_time': datetime.now().isoformat()
        }
    
    def _save_intermediate_result(self, factor_name: str, result: Dict[str, Any]):
        """保存中间结果（已禁用，避免创建不必要的文件夹）"""
        # 此方法已禁用，避免创建不必要的输出文件夹
        pass
    
    def _generate_final_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终报告"""
        try:
            # 因子排名
            factor_rankings = []
            for factor_name, factor_result in results.items():
                summary = factor_result.get('summary', {})
                avg_score = summary.get('avg_robustness_score', 0)
                best_timeframe = summary.get('best_timeframes', [('', 0)])[0][0]
                
                factor_rankings.append({
                    'factor_name': factor_name,
                    'category': self.factor_definitions[factor_name]['category'],
                    'avg_robustness_score': avg_score,
                    'best_timeframe': best_timeframe,
                    'successful_timeframes': summary.get('total_timeframes', 0)
                })
            
            factor_rankings.sort(key=lambda x: x['avg_robustness_score'], reverse=True)
            
            # 按类别统计
            category_stats = {}
            for ranking in factor_rankings:
                category = ranking['category']
                if category not in category_stats:
                    category_stats[category] = {
                        'count': 0,
                        'avg_score': 0,
                        'factors': []
                    }
                
                category_stats[category]['count'] += 1
                category_stats[category]['factors'].append(ranking)
            
            # 计算类别平均分
            for category, stats in category_stats.items():
                scores = [f['avg_robustness_score'] for f in stats['factors']]
                stats['avg_score'] = np.mean(scores)
                stats['factors'].sort(key=lambda x: x['avg_robustness_score'], reverse=True)
            
            return {
                'total_factors': len(results),
                'factor_rankings': factor_rankings,
                'category_statistics': category_stats,
                'top_factors': factor_rankings[:10],
                'robust_factors': [f for f in factor_rankings if f['avg_robustness_score'] >= 70],
                'recommendations': self._generate_recommendations(factor_rankings)
            }
            
        except Exception as e:
            self.logger.error(f"生成最终报告失败: {e}")
            return {}
    
    def _generate_recommendations(self, factor_rankings: List[Dict]) -> List[str]:
        """生成投资建议"""
        recommendations = []
        
        # 分析顶级因子
        top_factors = factor_rankings[:5]
        if top_factors:
            recommendations.append("推荐重点关注以下稳健因子：")
            for factor in top_factors:
                recommendations.append(
                    f"- {factor['factor_name']} ({factor['category']}): "
                    f"平均稳健性得分{factor['avg_robustness_score']:.1f}, "
                    f"最佳时间框架{factor['best_timeframe']}"
                )
        
        # 分析类别表现
        category_performances = {}
        for factor in factor_rankings:
            category = factor['category']
            if category not in category_performances:
                category_performances[category] = []
            category_performances[category].append(factor['avg_robustness_score'])
        
        best_category = max(category_performances.items(), 
                          key=lambda x: np.mean(x[1]))[0]
        
        recommendations.append(f"\n建议重点关注{best_category}类因子，该类别整体表现最佳。")
        
        # 风险提示
        poor_factors = [f for f in factor_rankings if f['avg_robustness_score'] < 40]
        if poor_factors:
            recommendations.append(f"\n注意：{len(poor_factors)}个因子表现不佳，建议谨慎使用或重新优化参数。")
        
        return recommendations

    def calculate_net_performance(self, gross_returns: pd.Series, trade_count: int) -> Dict[str, float]:
        """
        计算扣除交易成本后的净业绩
        
        Args:
            gross_returns: 毛收益率序列
            trade_count: 交易次数
            
        Returns:
            净业绩指标字典
        """
        try:
            # 计算交易成本影响
            total_cost_impact = trade_count * self.total_round_trip_cost
            
            # 计算净收益率
            cumulative_gross_return = (1 + gross_returns).prod() - 1
            net_return = cumulative_gross_return - total_cost_impact
            
            # 计算年化净收益率
            trading_days = len(gross_returns)
            years = trading_days / 252  # 假设252个交易日
            annualized_net_return = (1 + net_return) ** (1 / years) - 1 if years > 0 else 0
            
            # 计算净收益率序列
            cost_per_trade = self.total_round_trip_cost / trade_count if trade_count > 0 else 0
            net_returns = gross_returns - cost_per_trade
            
            # 计算净夏普比率
            risk_free_rate = 0.02  # 假设无风险利率2%
            excess_returns = net_returns - risk_free_rate / 252
            net_sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 1 else 0
            
            # 计算净最大回撤
            cumulative_net_returns = (1 + net_returns).cumprod()
            rolling_max = cumulative_net_returns.expanding().max()
            drawdowns = (cumulative_net_returns / rolling_max) - 1
            net_max_drawdown = drawdowns.min()
            
            # 计算成本比率
            cost_ratio = total_cost_impact / abs(cumulative_gross_return) if cumulative_gross_return != 0 else 0
            
            return {
                'gross_return': cumulative_gross_return,
                'net_return': net_return,
                'annualized_net_return': annualized_net_return,
                'net_sharpe_ratio': net_sharpe,
                'net_max_drawdown': net_max_drawdown,
                'total_cost_impact': total_cost_impact,
                'cost_ratio': cost_ratio,
                'trade_count': trade_count,
                'cost_efficiency': 1 - cost_ratio if cost_ratio < 1 else 0
            }
            
        except Exception as e:
            self.logger.error(f"计算净业绩时出错: {e}")
            return {
                'gross_return': 0,
                'net_return': 0,
                'annualized_net_return': 0,
                'net_sharpe_ratio': 0,
                'net_max_drawdown': 0,
                'total_cost_impact': 0,
                'cost_ratio': 0,
                'trade_count': trade_count,
                'cost_efficiency': 0
            }
    
    def analyze_factor_signals(self, factor_values: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        分析因子信号质量和特征
        
        Args:
            factor_values: 因子值DataFrame
            price_data: 价格数据字典
            
        Returns:
            信号分析结果
        """
        try:
            signal_analysis = {}
            
            # 处理不同类型的factor_values输入
            if isinstance(factor_values, pd.DataFrame):
                symbols = factor_values.columns
            elif isinstance(factor_values, pd.Series):
                # 如果是Series，检查是否有MultiIndex
                if isinstance(factor_values.index, pd.MultiIndex):
                    symbols = factor_values.index.get_level_values(0).unique()
                else:
                    # 单一股票的Series
                    symbols = [factor_values.name or 'single_stock']
            else:
                return {'individual_analysis': {}, 'aggregated_stats': {'signal_quality_score': 0}}
            
            for symbol in symbols:
                if symbol not in price_data:
                    continue
                    
                # 获取因子值
                if isinstance(factor_values, pd.DataFrame):
                    factor_series = factor_values[symbol].dropna()
                elif isinstance(factor_values, pd.Series):
                    if isinstance(factor_values.index, pd.MultiIndex):
                        factor_series = factor_values.loc[symbol].dropna()
                    else:
                        factor_series = factor_values.dropna()
                else:
                    continue
                
                price_series = price_data[symbol]['close']
                
                if len(factor_series) < 10:
                    continue
                
                # 计算信号统计
                signals = pd.Series(index=factor_series.index, dtype=float)
                
                # 根据因子方向生成信号
                # 这里简化处理，实际应该根据具体因子类型优化
                signal_threshold_high = factor_series.quantile(0.7)
                signal_threshold_low = factor_series.quantile(0.3)
                
                signals[factor_series > signal_threshold_high] = 1  # 买入信号
                signals[factor_series < signal_threshold_low] = -1  # 卖出信号
                signals.fillna(0, inplace=True)
                
                # 计算信号变化
                signal_changes = signals.diff().fillna(0)
                trade_signals = signal_changes[signal_changes != 0]
                
                # 计算信号质量指标
                total_signals = len(trade_signals)
                buy_signals = len(trade_signals[trade_signals == 1])
                sell_signals = len(trade_signals[trade_signals == -1])
                
                # 计算信号持续期
                signal_durations = []
                current_duration = 0
                current_signal = 0
                
                for signal in signals:
                    if signal != current_signal and signal != 0:
                        if current_signal != 0:
                            signal_durations.append(current_duration)
                        current_signal = signal
                        current_duration = 1
                    elif signal == current_signal and signal != 0:
                        current_duration += 1
                    elif signal == 0:
                        if current_signal != 0:
                            signal_durations.append(current_duration)
                        current_signal = 0
                        current_duration = 0
                
                if current_signal != 0:
                    signal_durations.append(current_duration)
                
                avg_signal_duration = np.mean(signal_durations) if signal_durations else 0
                
                # 计算信号集中度
                signal_concentration = len(signals[signals != 0]) / len(signals)
                
                signal_analysis[symbol] = {
                    'total_signals': total_signals,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'signal_balance': buy_signals / (buy_signals + sell_signals) if (buy_signals + sell_signals) > 0 else 0,
                    'avg_signal_duration': avg_signal_duration,
                    'signal_concentration': signal_concentration,
                    'signal_frequency': total_signals / len(factor_series) * 252  # 年化信号频率
                }
            
            # 汇总统计
            if signal_analysis:
                aggregated_stats = {
                    'avg_signals_per_stock': np.mean([s['total_signals'] for s in signal_analysis.values()]),
                    'avg_signal_duration': np.mean([s['avg_signal_duration'] for s in signal_analysis.values()]),
                    'avg_signal_concentration': np.mean([s['signal_concentration'] for s in signal_analysis.values()]),
                    'avg_signal_frequency': np.mean([s['signal_frequency'] for s in signal_analysis.values()]),
                    'signal_quality_score': self._calculate_signal_quality_score(signal_analysis)
                }
            else:
                aggregated_stats = {
                    'avg_signals_per_stock': 0,
                    'avg_signal_duration': 0,
                    'avg_signal_concentration': 0,
                    'avg_signal_frequency': 0,
                    'signal_quality_score': 0
                }
            
            return {
                'individual_analysis': signal_analysis,
                'aggregated_stats': aggregated_stats
            }
            
        except Exception as e:
            self.logger.error(f"分析因子信号时出错: {e}")
            return {
                'individual_analysis': {},
                'aggregated_stats': {
                    'avg_signals_per_stock': 0,
                    'avg_signal_duration': 0,
                    'avg_signal_concentration': 0,
                    'avg_signal_frequency': 0,
                    'signal_quality_score': 0
                }
            }
    
    def _calculate_signal_quality_score(self, signal_analysis: Dict) -> float:
        """计算信号质量得分 (0-100)"""
        try:
            if not signal_analysis:
                return 0
            
            scores = []
            
            for symbol, stats in signal_analysis.items():
                score = 0
                
                # 信号平衡性得分 (0-25分)
                balance = stats.get('signal_balance', 0.5)
                balance_score = 25 * (1 - abs(balance - 0.5) * 2)  # 越接近0.5越好
                score += balance_score
                
                # 信号持续期得分 (0-25分)
                duration = stats.get('avg_signal_duration', 0)
                duration_score = min(25, duration * 2)  # 适当长的持续期更好
                score += duration_score
                
                # 信号集中度得分 (0-25分)
                concentration = stats.get('signal_concentration', 0)
                concentration_score = min(25, concentration * 50)  # 适中的集中度
                score += concentration_score
                
                # 信号频率得分 (0-25分)
                frequency = stats.get('signal_frequency', 0)
                frequency_score = min(25, frequency * 5)  # 适中的频率
                score += frequency_score
                
                scores.append(score)
            
            return np.mean(scores) if scores else 0
            
        except Exception as e:
            self.logger.error(f"计算信号质量得分时出错: {e}")
            return 0

    def optimize_memory_usage(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        优化数据内存使用
        
        Args:
            data: 原始数据字典
            
        Returns:
            优化后的数据字典
        """
        try:
            optimized_data = {}
            
            for symbol, df in data.items():
                optimized_df = df.copy()
                
                # 优化数值类型
                for col in optimized_df.columns:
                    if optimized_df[col].dtype == 'float64':
                        # 检查是否可以降级为float32
                        col_min = optimized_df[col].min()
                        col_max = optimized_df[col].max()
                        
                        if pd.notna(col_min) and pd.notna(col_max):
                            if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                                optimized_df[col] = optimized_df[col].astype(np.float32)
                    
                    elif optimized_df[col].dtype == 'int64':
                        # 检查是否可以降级为int32
                        col_min = optimized_df[col].min()
                        col_max = optimized_df[col].max()
                        
                        if pd.notna(col_min) and pd.notna(col_max):
                            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                                optimized_df[col] = optimized_df[col].astype(np.int32)
                
                # 移除完全重复的行
                optimized_df = optimized_df.drop_duplicates()
                
                # 重置索引以节省内存
                optimized_df.reset_index(drop=True, inplace=True)
                
                optimized_data[symbol] = optimized_df
            
            # 计算内存节省
            original_size = sum(df.memory_usage(deep=True).sum() for df in data.values())
            optimized_size = sum(df.memory_usage(deep=True).sum() for df in optimized_data.values())
            
            memory_saved = original_size - optimized_size
            saved_percentage = (memory_saved / original_size * 100) if original_size > 0 else 0
            
            self.logger.info(f"内存优化完成: 节省 {memory_saved / 1024**2:.2f} MB ({saved_percentage:.1f}%)")
            
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"内存优化失败: {e}")
            return data
    
    def clear_cache(self):
        """清理分析器缓存"""
        try:
            # 清理数据缓存
            if hasattr(self, '_data_cache'):
                cache_size = len(self._data_cache)
                self._data_cache.clear()
                self.logger.info(f"清理数据缓存: {cache_size} 项")
            
            # 清理其他可能的缓存属性
            cache_attrs = ['_factor_cache', '_price_cache', '_result_cache']
            for attr in cache_attrs:
                if hasattr(self, attr):
                    cache_size = len(getattr(self, attr))
                    setattr(self, attr, {})
                    self.logger.info(f"清理{attr}缓存: {cache_size} 项")
            
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")

    def cleanup_memory(self, aggressive: bool = False):
        """
        清理内存
        
        Args:
            aggressive: 是否进行激进清理
        """
        try:
            # 清理缓存
            self.clear_cache()
            
            # 强制垃圾回收
            collected = gc.collect()
            
            # 激进清理：清理其他可能的缓存
            if aggressive:
                import sys
                for obj in list(gc.get_objects()):
                    try:
                        if isinstance(obj, (pd.DataFrame, pd.Series)):
                            del obj
                    except:
                        pass
                
                # 再次垃圾回收
                collected += gc.collect()
            
            self.logger.info(f"内存清理完成: 回收 {collected} 个对象")
            
        except Exception as e:
            self.logger.error(f"内存清理失败: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前内存使用情况
        
        Returns:
            内存使用信息字典
        """
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
                'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
                'percent': process.memory_percent(),       # 内存使用百分比
                'available_gb': psutil.virtual_memory().available / 1024**3  # 可用内存
            }
            
        except ImportError:
            self.logger.warning("psutil未安装，无法获取详细内存信息")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_gb': 0}
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_gb': 0}
    
    def monitor_memory_usage(self, threshold_percent: float = 80.0):
        """
        监控内存使用，并在超过阈值时自动清理
        
        Args:
            threshold_percent: 内存使用阈值百分比
        """
        try:
            memory_info = self.get_memory_usage()
            
            if memory_info['percent'] > threshold_percent:
                self.logger.warning(f"内存使用超过阈值: {memory_info['percent']:.1f}% > {threshold_percent}%")
                self.cleanup_memory(aggressive=True)
                
                # 清理后再次检查
                memory_info_after = self.get_memory_usage()
                self.logger.info(f"内存清理后使用率: {memory_info_after['percent']:.1f}%")
                
        except Exception as e:
            self.logger.error(f"内存监控失败: {e}")

    def visualize_factor_results(self, results: Dict[str, Any], save_path: str = None) -> None:
        """
        可视化因子分析结果
        
        Args:
            results: 分析结果字典
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图形
            fig = plt.figure(figsize=(20, 15))
            
            # 1. 因子稳健性得分雷达图
            if 'timeframe_results' in results:
                ax1 = plt.subplot(2, 3, 1, projection='polar')
                timeframe_results = results['timeframe_results']
                
                factors = list(timeframe_results.keys())
                robustness_scores = [timeframe_results[f].get('robustness_score', 0) for f in factors]
                
                angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
                scores = robustness_scores + [robustness_scores[0]]  # 闭合图形
                angles += angles[:1]
                
                ax1.plot(angles, scores, 'o-', linewidth=2)
                ax1.fill(angles, scores, alpha=0.25)
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(factors)
                ax1.set_title('因子稳健性得分', fontsize=14, fontweight='bold')
                ax1.set_ylim(0, 100)
            
            # 2. IC值分布图
            if 'timeframe_results' in results:
                ax2 = plt.subplot(2, 3, 2)
                ic_values = []
                factor_names = []
                
                for factor, result in timeframe_results.items():
                    ic_stats = result.get('ic_stats', {})
                    if ic_stats.get('mean_ic'):
                        ic_values.append(ic_stats['mean_ic'])
                        factor_names.append(factor)
                
                if ic_values:
                    bars = ax2.bar(factor_names, ic_values, color='skyblue', alpha=0.7)
                    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    ax2.set_title('因子IC值对比', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('IC值')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # 添加数值标签
                    for bar, value in zip(bars, ic_values):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 3. 时间框架表现热力图
            if 'timeframe_results' in results:
                ax3 = plt.subplot(2, 3, 3)
                
                # 创建热力图数据
                heatmap_data = []
                timeframes = []
                
                for factor, result in timeframe_results.items():
                    if 'timeframes' in result:
                        factor_data = []
                        for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                            tf_result = result['timeframes'].get(tf, {})
                            score = tf_result.get('robustness_score', 0)
                            factor_data.append(score)
                        heatmap_data.append(factor_data)
                        timeframes.append(factor)
                
                if heatmap_data:
                    sns.heatmap(heatmap_data, 
                              xticklabels=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                              yticklabels=timeframes,
                              annot=True, fmt='.1f', cmap='RdYlGn', ax=ax3)
                    ax3.set_title('时间框架表现热力图', fontsize=14, fontweight='bold')
                    ax3.set_xlabel('时间框架')
                    ax3.set_ylabel('因子')
            
            plt.tight_layout()
            plt.suptitle(f'因子分析综合报告 - {results.get("factor_name", "")}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # 保存或显示
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"可视化结果已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib或seaborn未安装，无法生成可视化")
        except Exception as e:
            self.logger.error(f"生成可视化时出错: {e}")
    
    def generate_performance_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        生成完整的性能报告
        
        Args:
            results: 分析结果
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 生成文本报告
            report_file = output_path / "performance_report.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# 因子性能分析报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 总体摘要
                f.write("## 总体摘要\n\n")
                if 'timeframe_results' in results:
                    timeframe_results = results['timeframe_results']
                    avg_score = np.mean([r.get('robustness_score', 0) for r in timeframe_results.values()])
                    f.write(f"- 平均稳健性得分: {avg_score:.2f}\n")
                    f.write(f"- 分析因子数量: {len(timeframe_results)}\n")
                    
                    # 最佳因子
                    best_factor = max(timeframe_results.items(), 
                                    key=lambda x: x[1].get('robustness_score', 0))
                    f.write(f"- 最佳因子: {best_factor[0]} (得分: {best_factor[1].get('robustness_score', 0):.2f})\n")
                
                f.write("\n## 详细分析\n\n")
                
                # 各因子详细分析
                if 'timeframe_results' in results:
                    for factor_name, result in results['timeframe_results'].items():
                        f.write(f"### {factor_name}\n\n")
                        
                        # 基本信息
                        f.write(f"**稳健性得分**: {result.get('robustness_score', 0):.2f}\n\n")
                        
                        # IC统计
                        ic_stats = result.get('ic_stats', {})
                        if ic_stats:
                            f.write(f"**IC统计**:\n")
                            f.write(f"- 平均IC: {ic_stats.get('mean_ic', 0):.4f}\n")
                            f.write(f"- IC标准差: {ic_stats.get('std_ic', 0):.4f}\n")
                            f.write(f"- 正向比率: {ic_stats.get('positive_ratio', 0):.2%}\n")
                            f.write(f"- 显著比率: {ic_stats.get('significant_ratio', 0):.2%}\n\n")
                        
                        # 信号质量
                        signal_analysis = result.get('signal_analysis', {})
                        if signal_analysis and 'aggregated_stats' in signal_analysis:
                            stats = signal_analysis['aggregated_stats']
                            f.write(f"**信号质量**:\n")
                            f.write(f"- 信号质量得分: {stats.get('signal_quality_score', 0):.1f}\n")
                            f.write(f"- 平均信号频率: {stats.get('avg_signal_frequency', 0):.2f}\n")
                            f.write(f"- 平均信号持续期: {stats.get('avg_signal_duration', 0):.1f}\n\n")
                        
                        # 回测表现
                        vbt_results = result.get('vectorbt_results', {})
                        if vbt_results:
                            f.write(f"**回测表现**:\n")
                            f.write(f"- 平均总收益: {vbt_results.get('avg_total_return', 0):.2%}\n")
                            f.write(f"- 平均夏普比率: {vbt_results.get('avg_sharpe_ratio', 0):.4f}\n")
                            f.write(f"- 平均最大回撤: {vbt_results.get('avg_max_drawdown', 0):.2%}\n")
                            f.write(f"- 胜率: {vbt_results.get('avg_win_rate', 0):.2%}\n\n")
                
                # 投资建议
                f.write("## 投资建议\n\n")
                f.write("基于上述分析，建议:\n")
                f.write("1. 优先选择稳健性得分高于60的因子\n")
                f.write("2. 关注IC值稳定且正向的因子\n")
                f.write("3. 结合多个时间框架的信号\n")
                f.write("4. 严格控制交易成本\n")
                f.write("5. 定期重新评估因子表现\n\n")
                
                # 风险提示
                f.write("## 风险提示\n\n")
                f.write("- 历史表现不代表未来结果\n")
                f.write("- 市场结构变化可能影响因子有效性\n")
                f.write("- 交易成本会显著影响实际收益\n")
                f.write("- 建议在实盘前进行充分测试\n")
            
            # 生成可视化
            viz_file = output_path / "performance_visualization.png"
            self.visualize_factor_results(results, str(viz_file))
            
            self.logger.info(f"性能报告已生成: {report_file}")
            self.logger.info(f"可视化图表已生成: {viz_file}")
            
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"生成性能报告时出错: {e}")
            return None


def main():
    """主函数"""
    print("🚀 启动基于vectorbt的港股因子探索与WFO验证系统...")
    
    # 创建分析器
    analyzer = VectorbtWFOAnalyzer()
    
    # 选择要分析的因子（可以指定部分因子进行测试）
    target_factors = ['RSI', 'MACD', 'Momentum_ROC', 'Bollinger_Position', 'Volume_Ratio']
    
    # 运行综合分析
    results = analyzer.run_comprehensive_analysis(target_factors)
    
    # 保存最终结果
    output_file = analyzer.output_dir / "comprehensive_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 分析完成！结果已保存到: {output_file}")
    
    # 显示摘要
    if 'final_report' in results:
        report = results['final_report']
        print(f"\n📊 分析摘要:")
        print(f"总因子数: {report.get('total_factors', 0)}")
        print(f"稳健因子数: {len(report.get('robust_factors', []))}")
        
        top_factors = report.get('top_factors', [])[:3]
        print(f"\n🏆 顶级因子:")
        for factor in top_factors:
            print(f"  {factor['factor_name']}: {factor['avg_robustness_score']:.1f}分 ({factor['best_timeframe']})")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 建议:")
            for rec in recommendations:
                print(f"  {rec}")


if __name__ == "__main__":
    main()