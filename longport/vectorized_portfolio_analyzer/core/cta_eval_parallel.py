#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Phase 1 - 并行加速CTA评估器
==============================

基于joblib实现的并行回测，预期性能：1008s → 144s (7x加速)
约束：任务函数内禁止磁盘读取和因子重计算
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Any, Tuple
from joblib import Parallel, delayed
import psutil
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.factor_pool import AdvancedFactorPool
from utils.dtype_fixer import CategoricalDtypeFixer

class CTAEvaluatorParallel:
    """🚀 Phase 1: 并行加速CTA评估器"""
    
    def __init__(self, 
                 look_ahead: int = 6,
                 entry_percentile: float = 0.90,
                 exit_percentile: float = 0.10,
                 sl_stop: float = 0.02,
                 tp_stop: float = 0.03,
                 direction: str = 'both',
                 slippage: float = 0.001,
                 fees: float = 0.0005,
                 min_trades: int = 30):
        """初始化并行评估器"""
        self.look_ahead = look_ahead
        self.entry_percentile = entry_percentile
        self.exit_percentile = exit_percentile
        self.sl_stop = sl_stop
        self.tp_stop = tp_stop
        self.direction = direction
        self.slippage = slippage
        self.fees = fees
        self.min_trades = min_trades
        
        # 初始化组件
        self.factor_pool = AdvancedFactorPool()
        self.dtype_fixer = CategoricalDtypeFixer()
        
    def log_memory(self, stage: str):
        """内存监控"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"PeakRAM {memory_mb:.1f} MB ({stage})")
            return memory_mb
        except:
            return 0.0
    
    def preload_data(self, symbols: List[str], data_dir: str = "../vectorbt_workspace/data", timeframe: str = "5m") -> Tuple[Dict, Dict]:
        """🔥 预加载数据 - 在主进程完成，禁止在任务函数中重复加载"""
        print(f"📂 预加载 {len(symbols)} 只股票的价格和因子数据...")
        start_time = time.time()
        
        # 预加载价格数据
        price_data = {}
        timeframe_dir = f"{data_dir}/{timeframe}"
        
        for symbol in symbols:
            try:
                file_path = f"{timeframe_dir}/{symbol}.parquet"
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if not df.empty and 'close' in df.columns:
                        price_data[symbol] = df
                else:
                    print(f"⚠️ 文件不存在: {file_path}")
            except Exception as e:
                print(f"⚠️ 加载 {symbol} 失败: {e}")
                continue
        
        # 预加载因子矩阵
        factor_matrix = {}
        for symbol in price_data.keys():
            try:
                # 计算所有因子
                factors_df = self.factor_pool.calculate_all_factors(price_data[symbol])
                
                # 数据类型修复
                factors_df = self.dtype_fixer.fix_categorical_dataframe(factors_df)
                
                # 转换为float32节省内存
                numeric_cols = factors_df.select_dtypes(include=[np.number]).columns
                factors_df[numeric_cols] = factors_df[numeric_cols].astype('float32')
                
                # 去除NaN和常数列
                factors_df = factors_df.dropna(axis=1, how='all')
                for col in factors_df.columns:
                    if factors_df[col].nunique() <= 1:
                        factors_df = factors_df.drop(columns=[col])
                
                factor_matrix[symbol] = factors_df
                
            except Exception as e:
                print(f"⚠️ 计算 {symbol} 因子失败: {e}")
                continue
        
        elapsed = time.time() - start_time
        print(f"✅ 预加载完成: {len(price_data)} 只股票, 耗时 {elapsed:.1f}s")
        
        return price_data, factor_matrix
    
    def get_annual_factor(self, timeframe: str = '5m') -> float:
        """计算年化因子 - 按持仓周期调整"""
        periods_per_year = {
            '1m': 252 * 240,
            '5m': 252 * 48,
            '15m': 252 * 16,
            '30m': 252 * 8,
            '1h': 252 * 4,
            '4h': 252,
            '1d': 252
        }
        
        base_periods = periods_per_year.get(timeframe, 252 * 48)
        holding_periods = self.look_ahead
        adjusted_periods = base_periods / holding_periods
        
        return np.sqrt(adjusted_periods)
    
    def batch_evaluate(self, symbols: List[str], 
                      factor_data: Dict[str, pd.DataFrame],
                      price_data: Dict[str, pd.DataFrame], 
                      factor_names: List[str],
                      timeframe: str = '5m') -> pd.DataFrame:
        """🚀 并行批量评估 - 保持原接口"""
        
        print(f"🚀 Phase 1 并行CTA评估启动...")
        total_start = time.time()
        
        # 记录初始内存
        start_memory = self.log_memory("开始")
        
        # 🔥 关键：使用传入的数据，不重新加载
        valid_symbols = [s for s in symbols if s in price_data and s in factor_data]
        
        # 构建所有任务
        tasks = []
        for symbol in valid_symbols:
            symbol_factors = factor_data[symbol]
            symbol_prices = price_data[symbol]
            
            # 数据对齐检查
            common_index = symbol_factors.index.intersection(symbol_prices.index)
            if len(common_index) < 50:
                continue
                
            for factor_name in factor_names:
                if factor_name in symbol_factors.columns:
                    tasks.append((symbol, factor_name, symbol_prices, symbol_factors, timeframe))
        
        print(f"📊 生成 {len(tasks)} 个并行任务")
        
        # 🚀 并行执行
        annual_factor = self.get_annual_factor(timeframe)
        
        results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
            delayed(single_cta_task)(
                task[0], task[1], task[2], task[3], task[4],
                self.entry_percentile, self.exit_percentile,
                self.sl_stop, self.tp_stop, self.direction,
                self.slippage, self.fees, self.min_trades, annual_factor
            ) for task in tasks
        )
        
        # 过滤有效结果
        valid_results = [r for r in results if r is not None]
        print(f"🔍 调试信息: 总任务 {len(results)}, 有效结果 {len(valid_results)}")
        
        # 统计失败原因
        none_count = sum(1 for r in results if r is None)
        print(f"   失败任务: {none_count}/{len(results)}")
        
        # 记录结束内存
        end_memory = self.log_memory("结束")
        
        total_time = time.time() - total_start
        print(f"✅ Phase 1 并行评估完成")
        print(f"TotalTime {total_time:.1f} s  PeakRAM {max(start_memory, end_memory):.1f} MB")
        
        if not valid_results:
            return pd.DataFrame()
        
        return pd.DataFrame(valid_results)


def single_cta_task(symbol: str, factor_name: str, 
                   price_data: pd.DataFrame, factor_matrix: pd.DataFrame, 
                   timeframe: str, entry_percentile: float, exit_percentile: float,
                   sl_stop: float, tp_stop: float, direction: str,
                   slippage: float, fees: float, min_trades: int, annual_factor: float) -> Dict[str, Any]:
    """🔥 单任务处理函数 - 严格约束：禁止读取磁盘或重计算因子！"""
    
    try:
        # 🚨 约束检查：禁止在任务函数内进行磁盘读取或因子计算
        if not isinstance(price_data, pd.DataFrame) or not isinstance(factor_matrix, pd.DataFrame):
            return None
        
        # 数据对齐
        common_index = price_data.index.intersection(factor_matrix.index)
        if len(common_index) < 50:
            return None
        
        # 提取价格和因子
        price = price_data.reindex(common_index)['close']
        
        if factor_name not in factor_matrix.columns:
            return None
            
        factor = factor_matrix.reindex(common_index)[factor_name]
        
        # 检查数据有效性
        if factor.isna().all():
            print(f"   {symbol}_{factor_name}: 因子全为NaN")
            return None
        if factor.nunique() <= 1:
            print(f"   {symbol}_{factor_name}: 因子为常数 (unique={factor.nunique()})")
            return None
        
        # 生成信号
        factor_clean = factor.dropna()
        if len(factor_clean) < 30:
            print(f"   {symbol}_{factor_name}: 有效数据不足 ({len(factor_clean)} < 30)")
            return None
        
        # 动态分位数计算 - 使用较短窗口
        window = min(288, len(factor_clean) // 4) if len(factor_clean) > 100 else len(factor_clean)
        
        entries = factor > factor.rolling(window, min_periods=20).quantile(entry_percentile)
        exits = factor < factor.rolling(window, min_periods=20).quantile(exit_percentile)
        
        entries = entries.fillna(False)
        exits = exits.fillna(False)
        
        # 检查信号
        if not entries.any() and not exits.any():
            return None
        
        # VectorBT回测 - 显式指定频率
        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            sl_stop=sl_stop if sl_stop > 0 else None,
            tp_stop=tp_stop if tp_stop > 0 else None,
            direction=direction,
            init_cash=100000,
            fees=fees,
            slippage=slippage,
            freq='5T'  # 🔥 修复: 显式指定5分钟频率
        )
        
        # 提取指标
        total_return = pf.total_return()
        raw_sharpe = pf.sharpe_ratio()
        trades_df = pf.trades.records_readable
        total_trades = len(trades_df) if hasattr(trades_df, '__len__') else 0
        
        # 交易次数过滤 - 暂时放宽要求以便调试
        if total_trades < 5:  # 降低到5次以便观察结果
            print(f"   {symbol}_{factor_name}: 交易次数不足 ({total_trades} < 5)")
            return None
        
        # 年化夏普率
        sharpe = raw_sharpe / annual_factor if annual_factor > 1 else raw_sharpe
        
        # 计算其他指标 - 修复列名问题
        if total_trades > 0:
            
            # 检查正确的列名 - VectorBT可能使用不同的列名
            pnl_col = None
            for col in ['pnl', 'PnL', 'return', 'Return', 'profit', 'Profit']:
                if col in trades_df.columns:
                    pnl_col = col
                    break
            
            if pnl_col:
                win_trades = trades_df[trades_df[pnl_col] > 0]
                win_rate = len(win_trades) / total_trades if total_trades > 0 else 0.0
                
                if len(win_trades) > 0 and len(trades_df[trades_df[pnl_col] < 0]) > 0:
                    avg_win = win_trades[pnl_col].mean()
                    avg_loss = abs(trades_df[trades_df[pnl_col] < 0][pnl_col].mean())
                    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
                else:
                    profit_loss_ratio = 0.0
                    
                avg_trade_return = trades_df[pnl_col].mean() if total_trades > 0 else 0.0
            else:
                print(f"   ⚠️ 找不到PnL列，可用列: {list(trades_df.columns)}")
                win_rate = 0.0
                profit_loss_ratio = 0.0
                avg_trade_return = 0.0
        else:
            win_rate = 0.0
            profit_loss_ratio = 0.0
            avg_trade_return = 0.0
        
        max_drawdown = pf.max_drawdown()
        
        return {
            'symbol': symbol,
            'factor': factor_name,
            'total_return': float(total_return),
            'sharpe': float(sharpe),
            'raw_sharpe': float(raw_sharpe),
            'win_rate': float(win_rate),
            'profit_loss_ratio': float(profit_loss_ratio),
            'max_drawdown': float(max_drawdown),
            'trades': int(total_trades),
            'avg_trade_return': float(avg_trade_return),
            'signal_strength': float(factor_clean.std()),
            'signal_count': int(entries.sum() + exits.sum()),
            'data_quality': 'valid',
            'annual_factor': float(annual_factor)
        }
        
    except Exception as e:
        print(f"   ❌ 异常: {symbol}_{factor_name} - {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Phase 1 验证主函数"""
    print("🚀 Phase 1 - 并行加速CTA评估器验证")
    print("=" * 60)
    
    # 初始化评估器
    evaluator = CTAEvaluatorParallel()
    
    # 测试股票列表  
    test_symbols = ['0005.HK', '0020.HK', '0175.HK', '0291.HK', '0340.HK']
    
    # 预加载数据
    price_data, factor_matrix = evaluator.preload_data(test_symbols)
    
    if not price_data:
        print("❌ 没有加载到有效数据")
        return
    
    # 获取因子名称
    all_factors = set()
    for factors_df in factor_matrix.values():
        all_factors.update(factors_df.columns)
    factor_names = list(all_factors)[:10]  # 测试前10个因子
    
    print(f"📊 测试配置: {len(price_data)}只股票 × {len(factor_names)}个因子")
    
    # 执行并行评估
    results = evaluator.batch_evaluate(
        symbols=list(price_data.keys()),
        factor_data=factor_matrix,
        price_data=price_data,
        factor_names=factor_names,
        timeframe='5m'
    )
    
    # 验证结果
    if not results.empty:
        print(f"✅ 生成 {len(results)} 条评估结果")
        print(f"📈 最佳夏普率: {results['sharpe'].max():.4f}")
        print(f"📊 平均交易次数: {results['trades'].mean():.1f}")
    else:
        print("❌ 评估结果为空")


if __name__ == "__main__":
    main()
