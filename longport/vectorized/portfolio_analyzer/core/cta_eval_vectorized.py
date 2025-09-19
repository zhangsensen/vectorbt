#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Phase 2 - VectorBT原生批量回测
================================

革命性的单次批量回测，预期性能：8.4s → 0.4s (20x加速)
核心：构建三维数据，单次Portfolio.from_signals调用
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt
# import xarray as xr  # 改用pandas实现
from typing import Dict, List, Any, Tuple
import psutil
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.factor_pool import AdvancedFactorPool
from utils.dtype_fixer import CategoricalDtypeFixer

class CTAEvaluatorVectorized:
    """🚀 Phase 2: VectorBT原生批量评估器"""
    
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
        """初始化向量化评估器"""
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
    
    def preload_vectorized_data(self, symbols: List[str], data_dir: str = "../vectorbt_workspace/data", 
                               timeframe: str = "5m") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """🔥 Phase 2核心: 预加载向量化数据结构 (pandas版本)"""
        print(f"📂 Phase 2预加载: 构建向量化数据结构...")
        start_time = time.time()
        
        timeframe_dir = f"{data_dir}/{timeframe}"
        
        # Step 1: 构建价格矩阵 (time × asset)
        price_dfs = []
        factor_data = {}
        
        for symbol in symbols:
            try:
                file_path = f"{timeframe_dir}/{symbol}.parquet"
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if not df.empty and 'close' in df.columns:
                        # 计算因子
                        factors_df = self.factor_pool.calculate_all_factors(df)
                        factors_df = self.dtype_fixer.fix_categorical_dataframe(factors_df)
                        
                        # 数据类型优化
                        numeric_cols = factors_df.select_dtypes(include=[np.number]).columns
                        factors_df[numeric_cols] = factors_df[numeric_cols].astype('float32')
                        
                        # 去除NaN和常数列
                        factors_df = factors_df.dropna(axis=1, how='all')
                        for col in factors_df.columns:
                            if factors_df[col].nunique() <= 1:
                                factors_df = factors_df.drop(columns=[col])
                        
                        # 保存价格数据
                        price_series = df['close'].astype('float32')
                        price_series.name = symbol
                        price_dfs.append(price_series)
                        
                        # 保存因子数据
                        factor_data[symbol] = factors_df
                        
            except Exception as e:
                print(f"⚠️ 加载 {symbol} 失败: {e}")
                continue
        
        if not price_dfs:
            raise ValueError("没有成功加载任何数据")
        
        # Step 2: 构建价格DataFrame (time × asset)
        price_df = pd.concat(price_dfs, axis=1, sort=True).dropna()
        price_df = price_df.astype('float32')
        
        print(f"✅ 价格矩阵: {price_df.shape} (time × asset)")
        
        # Step 3: 构建因子数据结构 (pandas版本)
        all_factor_names = set()
        
        # 收集所有因子名称
        for factors_df in factor_data.values():
            all_factor_names.update(factors_df.columns)
        
        all_factor_names = sorted(list(all_factor_names))
        
        # 构建统一的因子DataFrame字典
        aligned_factor_data = {}
        for symbol in price_df.columns:
            if symbol in factor_data:
                symbol_factors = factor_data[symbol].reindex(price_df.index)
                
                # 确保所有因子都存在，缺失的用NaN填充
                for factor_name in all_factor_names:
                    if factor_name not in symbol_factors.columns:
                        symbol_factors[factor_name] = np.nan
                
                # 重新排序列
                symbol_factors = symbol_factors.reindex(columns=all_factor_names)
                symbol_factors = symbol_factors.astype('float32')
                
                aligned_factor_data[symbol] = symbol_factors
        
        # 构建因子信息
        factor_info = {
            'data': aligned_factor_data,
            'factor_names': all_factor_names,
            'shape': (len(price_df), len(price_df.columns), len(all_factor_names))
        }
        
        elapsed = time.time() - start_time
        print(f"✅ 因子数据: {factor_info['shape']} (time × asset × factor)")
        print(f"✅ 向量化预加载完成: 耗时 {elapsed:.1f}s")
        
        return price_df, factor_info
    
    def generate_vectorized_signals(self, factor_info: Dict[str, Any], price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """🔥 生成向量化信号 (pandas版本)"""
        print(f"⚡ 生成向量化信号...")
        
        # 动态窗口大小
        time_len = len(price_df)
        window = min(288, time_len // 4) if time_len > 100 else time_len
        min_periods = min(20, window // 2)
        
        print(f"   信号窗口: {window}, 最小周期: {min_periods}")
        
        # 为所有 asset-factor 组合生成信号
        all_entries = []
        all_exits = []
        multi_columns = []
        
        for asset in price_df.columns:
            asset_factors = factor_info['data'][asset]
            
            for factor_name in factor_info['factor_names']:
                factor_series = asset_factors[factor_name]
                
                # 计算滚动分位数
                entry_threshold = factor_series.rolling(window=window, min_periods=min_periods).quantile(self.entry_percentile)
                exit_threshold = factor_series.rolling(window=window, min_periods=min_periods).quantile(self.exit_percentile)
                
                # 生成信号
                entries = (factor_series > entry_threshold).astype('int8')
                exits = (factor_series < exit_threshold).astype('int8')
                
                all_entries.append(entries)
                all_exits.append(exits)
                multi_columns.append((asset, factor_name))
        
        # 构建MultiIndex DataFrame
        entries_df = pd.concat(all_entries, axis=1)
        exits_df = pd.concat(all_exits, axis=1)
        
        entries_df.columns = pd.MultiIndex.from_tuples(multi_columns, names=['asset', 'factor'])
        exits_df.columns = pd.MultiIndex.from_tuples(multi_columns, names=['asset', 'factor'])
        
        print(f"✅ 信号生成完成: entries {entries_df.shape}, exits {exits_df.shape}")
        
        return entries_df, exits_df
    
    def batch_evaluate(self, symbols: List[str], factor_data: Dict[str, pd.DataFrame],
                      price_data: Dict[str, pd.DataFrame], factor_names: List[str],
                      timeframe: str = '5m') -> pd.DataFrame:
        """🚀 Phase 2核心: VectorBT原生批量评估"""
        
        print(f"🚀 Phase 2 VectorBT原生批量回测启动...")
        total_start = time.time()
        
        # 记录初始内存
        start_memory = self.log_memory("开始")
        
        try:
            # Step 1: 预加载向量化数据
            price_df, factor_info = self.preload_vectorized_data(symbols, timeframe=timeframe)
            
            # Step 2: 生成向量化信号
            entries_df, exits_df = self.generate_vectorized_signals(factor_info, price_df)
            
            # Step 3: 构建价格矩阵 (time × [asset_factor])
            print(f"🔧 构建价格矩阵...")
            
            # 为每个asset-factor组合复制价格数据
            price_matrix = []
            for asset in price_df.columns:
                for factor_name in factor_info['factor_names']:
                    price_matrix.append(price_df[asset])
            
            price_matrix_df = pd.concat(price_matrix, axis=1)
            price_matrix_df.columns = entries_df.columns  # 使用相同的MultiIndex
            
            print(f"✅ 价格矩阵完成: {price_matrix_df.shape} (time × [asset_factor])")
            
            # Step 4: 🔥 单次VectorBT批量回测
            print(f"🚀 执行单次VectorBT批量回测...")
            
            pf = vbt.Portfolio.from_signals(
                close=price_matrix_df,
                entries=entries_df,
                exits=exits_df,
                sl_stop=self.sl_stop if self.sl_stop > 0 else None,
                tp_stop=self.tp_stop if self.tp_stop > 0 else None,
                direction=self.direction,
                init_cash=100000,
                fees=self.fees,
                slippage=self.slippage,
                freq='5min'  # 修复频率警告
                # 注意：移除broadcast参数，让VectorBT自动处理
            )
            
            print(f"✅ VectorBT批量回测完成")
            
            # Step 5: 提取批量结果
            print(f"📊 提取批量结果...")
            
            # 获取所有指标
            total_returns = pf.total_return()
            sharpe_ratios = pf.sharpe_ratio()
            
            # 年化调整
            annual_factor = self.get_annual_factor(timeframe)
            adjusted_sharpe = sharpe_ratios / annual_factor
            
            # 处理无效值
            adjusted_sharpe = adjusted_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 按因子聚合
            results = []
            for factor in factor_info['factor_names']:
                # 获取该因子在所有资产上的表现
                factor_mask = adjusted_sharpe.index.get_level_values('factor') == factor
                factor_sharpes = adjusted_sharpe[factor_mask]
                
                if len(factor_sharpes) > 0:
                    # 过滤有效值
                    valid_sharpes = factor_sharpes[~factor_sharpes.isna()]
                    
                    if len(valid_sharpes) > 0:
                        mean_sharpe = valid_sharpes.mean()
                        std_sharpe = valid_sharpes.std()
                        
                        # 检查是否为有效数值
                        if not (np.isnan(mean_sharpe) or np.isinf(mean_sharpe)):
                            results.append({
                                'factor': factor,
                                'sharpe': float(mean_sharpe),
                                'sharpe_std': float(std_sharpe) if not np.isnan(std_sharpe) else 0.0,
                                'asset_count': len(valid_sharpes),
                                'annual_factor': float(annual_factor)
                            })
            
            # 记录结束内存
            end_memory = self.log_memory("结束")
            
            total_time = time.time() - total_start
            print(f"✅ Phase 2批量评估完成")
            print(f"TotalTime {total_time:.1f} s  PeakRAM {max(start_memory, end_memory):.1f} MB")
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"❌ Phase 2批量回测失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


def main():
    """Phase 2验证主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2 VectorBT原生批量回测验证')
    parser.add_argument('--mini', nargs=2, help='迷你测试模式: stock_count factor_count')
    args = parser.parse_args()
    
    print("🚀 Phase 2 - VectorBT原生批量回测验证")
    print("=" * 60)
    
    # 初始化评估器
    evaluator = CTAEvaluatorVectorized()
    
    if args.mini:
        stock_count, factor_count = map(int, args.mini)
        print(f"🧪 迷你测试模式: {stock_count}只股票 × {factor_count}个因子")
        
        # 测试股票列表
        all_symbols = ['0005.HK', '0020.HK', '0175.HK', '0291.HK', '0340.HK']
        test_symbols = all_symbols[:stock_count]
    else:
        # 完整测试
        test_symbols = ['0005.HK', '0020.HK', '0175.HK', '0291.HK', '0340.HK']
        factor_count = 10
    
    try:
        # 预加载数据
        price_df, factor_info = evaluator.preload_vectorized_data(test_symbols)
        
        # 获取因子名称
        all_factors = factor_info['factor_names']
        factor_names = all_factors[:factor_count] if args.mini else all_factors[:10]
        
        print(f"📊 测试配置: {len(test_symbols)}只股票 × {len(factor_names)}个因子")
        
        # 构建模拟输入（兼容原接口）
        price_data = {symbol: pd.DataFrame({'close': price_df[symbol]}) for symbol in test_symbols}
        factor_data = {}
        
        for symbol in test_symbols:
            symbol_factors = factor_info['data'][symbol]
            # 确保因子名称在数据中存在
            available_factors = [f for f in factor_names if f in symbol_factors.columns]
            factor_data[symbol] = symbol_factors[available_factors]
        
        # 更新factor_names为实际可用的因子
        factor_names = available_factors
        
        # 执行Phase 2评估
        results = evaluator.batch_evaluate(
            symbols=test_symbols,
            factor_data=factor_data,
            price_data=price_data,
            factor_names=factor_names,
            timeframe='5m'
        )
        
        # 验证结果
        if not results.empty:
            print(f"✅ 生成 {len(results)} 个因子结果")
            print(f"📈 最佳夏普率: {results['sharpe'].max():.4f}")
            print(f"📊 平均夏普率: {results['sharpe'].mean():.4f}")
            
            # 对比Phase 1 (如果需要)
            if args.mini:
                print(f"\n🔍 Phase 2 vs Phase 1 对比:")
                print(f"   因子数量: {len(results)} vs 预期{factor_count}")
                print(f"   平均夏普: {results['sharpe'].mean():.4f}")
        else:
            print("❌ 评估结果为空")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
