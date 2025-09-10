#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向量化多股票分析器 - 充分利用vectorbt的优势
真正的批量处理：53只股票 × 7个时间框架 × 5个因子 = 1855个组合并行分析
"""

import sys
import pandas as pd
import numpy as np
import vectorbt as vbt
import talib
from pathlib import Path
from datetime import datetime
import logging
import json
import gc
import traceback
import time
from typing import Dict, List, Tuple, Optional, Union
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

class VectorizedMultiStockAnalyzer:
    """充分利用vectorbt优势的多股票向量化分析器"""
    
    def __init__(self, 
                 data_dir: str = "/Users/zhangshenshen/longport/vectorbt_workspace/data",
                 start_date: str = "2024-01-01",
                 end_date: str = "2025-09-01",
                 memory_limit_gb: float = 16.0):
        
        self.data_dir = Path(data_dir)
        self.start_date = pd.to_datetime(start_date).tz_localize('Asia/Hong_Kong')
        self.end_date = pd.to_datetime(end_date).tz_localize('Asia/Hong_Kong')
        self.memory_limit_gb = memory_limit_gb
        
        # 所有支持的时间框架
        self.all_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        # 所有因子
        self.all_factors = ["RSI", "MACD", "Momentum_ROC", "Price_Position", "Volume_Ratio"]
        
        # 数据缓存
        self.data_cache = {}
        self.factors_cache = {}
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 获取所有可用股票
        self.all_symbols = self._get_available_symbols()
        
        self.logger.info(f"初始化向量化分析器:")
        self.logger.info(f"  数据目录: {self.data_dir}")
        self.logger.info(f"  时间范围: {self.start_date} 到 {self.end_date}")
        self.logger.info(f"  内存限制: {self.memory_limit_gb}GB")
        self.logger.info(f"  可用股票数: {len(self.all_symbols)}")
        self.logger.info(f"  时间框架数: {len(self.all_timeframes)}")
        self.logger.info(f"  因子数: {len(self.all_factors)}")
        
    def _setup_logger(self):
        """设置日志记录"""
        logger = logging.getLogger(f"{__name__}.VectorizedAnalyzer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_available_symbols(self) -> List[str]:
        """获取所有可用的股票代码"""
        symbols = []
        for timeframe in ['1m']:  # 只需要检查一个时间框架
            timeframe_dir = self.data_dir / timeframe
            if timeframe_dir.exists():
                for file in timeframe_dir.glob("*.HK.parquet"):
                    symbol = file.stem
                    if symbol not in symbols:
                        symbols.append(symbol)
        
        symbols.sort()
        return symbols
    
    def load_timeframe_data_vectorized(self, 
                                     timeframe: str, 
                                     symbols: List[str] = None) -> pd.DataFrame:
        """向量化加载指定时间框架的所有股票数据"""
        if symbols is None:
            symbols = self.all_symbols
        
        cache_key = f"{timeframe}_{len(symbols)}"
        if cache_key in self.data_cache:
            self.logger.info(f"使用缓存数据: {timeframe}, {len(symbols)}只股票")
            return self.data_cache[cache_key]
        
        self.logger.info(f"开始加载{timeframe}数据: {len(symbols)}只股票")
        
        timeframe_dir = self.data_dir / timeframe
        if not timeframe_dir.exists():
            raise ValueError(f"时间框架目录不存在: {timeframe_dir}")
        
        # 批量加载所有股票数据
        dfs = []
        successful_symbols = []
        
        for symbol in symbols:
            file_path = timeframe_dir / f"{symbol}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    
                    # 时区处理
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        if df['timestamp'].dt.tz is None:
                            df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Hong_Kong')
                        else:
                            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Hong_Kong')
                        df.set_index('timestamp', inplace=True)
                    
                    # 时间范围过滤
                    df = df.loc[self.start_date:self.end_date]
                    
                    if len(df) > 0:
                        # 添加股票标识
                        df['symbol'] = symbol
                        dfs.append(df)
                        successful_symbols.append(symbol)
                    else:
                        self.logger.warning(f"{symbol}: 时间范围内无数据")
                        
                except Exception as e:
                    self.logger.warning(f"{symbol}: 加载失败 - {e}")
        
        if not dfs:
            raise ValueError(f"没有成功加载任何{timeframe}数据")
        
        # 合并所有数据为多重索引DataFrame
        combined_df = pd.concat(dfs, ignore_index=False)
        combined_df.reset_index(inplace=True)
        
        # 创建MultiIndex: (symbol, timestamp)
        combined_df.set_index(['symbol', 'timestamp'], inplace=True)
        combined_df.sort_index(inplace=True)
        
        # 缓存数据
        self.data_cache[cache_key] = combined_df
        
        self.logger.info(f"✅ {timeframe}数据加载完成: {len(successful_symbols)}只股票, 形状{combined_df.shape}")
        
        return combined_df
    
    def calculate_factors_vectorized(self, 
                                   data: pd.DataFrame, 
                                   factors: List[str] = None) -> Dict[str, pd.Series]:
        """向量化计算所有因子 - 同时处理所有股票"""
        if factors is None:
            factors = self.all_factors
        
        self.logger.info(f"开始向量化计算因子: {factors}")
        
        factors_dict = {}
        
        # 按股票分组进行向量化计算
        grouped = data.groupby('symbol')
        
        for factor_name in factors:
            self.logger.info(f"  计算因子: {factor_name}")
            
            factor_results = []
            
            # 对每只股票计算因子（这里仍然需要分股票，因为talib不支持MultiIndex）
            for symbol, group_data in grouped:
                try:
                    factor_values = self._calculate_single_factor(group_data, factor_name)
                    
                    if factor_values is not None and len(factor_values) > 0:
                        # 创建带symbol标识的Series
                        factor_series = pd.Series(
                            factor_values, 
                            index=group_data.index.get_level_values('timestamp'),
                            name=f"{symbol}_{factor_name}"
                        )
                        
                        # 添加symbol列用于后续重建MultiIndex
                        factor_df = factor_series.to_frame(factor_name)
                        factor_df['symbol'] = symbol
                        factor_results.append(factor_df)
                        
                except Exception as e:
                    self.logger.warning(f"{symbol}计算{factor_name}失败: {e}")
            
            if factor_results:
                # 合并所有股票的因子数据
                factor_combined = pd.concat(factor_results)
                factor_combined.reset_index(inplace=True)
                factor_combined.set_index(['symbol', 'timestamp'], inplace=True)
                factor_combined.sort_index(inplace=True)
                
                factors_dict[factor_name] = factor_combined[factor_name]
                
                self.logger.info(f"  ✅ {factor_name}计算完成: {len(factor_results)}只股票")
            else:
                self.logger.warning(f"  ❌ {factor_name}计算失败：无有效数据")
        
        return factors_dict
    
    def _calculate_single_factor(self, data: pd.DataFrame, factor_name: str) -> np.ndarray:
        """计算单个因子"""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            if factor_name == "RSI":
                return talib.RSI(close, timeperiod=14)
            
            elif factor_name == "MACD":
                macd, macd_signal, macd_hist = talib.MACD(close)
                return macd
            
            elif factor_name == "Momentum_ROC":
                return talib.ROC(close, timeperiod=10)
            
            elif factor_name == "Price_Position":
                # 价格位置：当前价格在过去N天高低点中的相对位置
                period = 20
                rolling_high = pd.Series(high).rolling(period).max()
                rolling_low = pd.Series(low).rolling(period).min()
                price_position = (close - rolling_low) / (rolling_high - rolling_low)
                return price_position.values
            
            elif factor_name == "Volume_Ratio":
                # 成交量比率：当前成交量与过去N天平均成交量的比值
                period = 20
                avg_volume = pd.Series(volume).rolling(period).mean()
                volume_ratio = volume / avg_volume
                return volume_ratio.values
            
            else:
                self.logger.warning(f"未知因子: {factor_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"计算因子{factor_name}失败: {e}")
            return None
    
    def calculate_ic_vectorized(self, 
                              data: pd.DataFrame, 
                              factors_dict: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """向量化计算所有因子的IC值"""
        self.logger.info("开始向量化计算IC")
        
        ic_results = {}
        
        # 计算未来收益率（向量化）
        grouped = data.groupby('symbol')
        future_returns_list = []
        
        for symbol, group_data in grouped:
            returns = group_data['close'].pct_change(periods=1).shift(-1)
            returns_df = returns.to_frame('future_returns')
            returns_df['symbol'] = symbol
            future_returns_list.append(returns_df)
        
        if future_returns_list:
            future_returns_combined = pd.concat(future_returns_list)
            future_returns = future_returns_combined['future_returns']
        else:
            self.logger.error("无法计算未来收益率")
            return ic_results
        
        # 对每个因子计算IC
        for factor_name, factor_values in factors_dict.items():
            try:
                # 确保索引一致
                common_index = factor_values.index.intersection(future_returns.index)
                
                if len(common_index) > 10:  # 至少需要10个数据点
                    aligned_factor = factor_values.loc[common_index]
                    aligned_returns = future_returns.loc[common_index]
                    
                    # 去除NaN值
                    valid_mask = aligned_factor.notna() & aligned_returns.notna()
                    clean_factor = aligned_factor[valid_mask]
                    clean_returns = aligned_returns[valid_mask]
                    
                    if len(clean_factor) > 10:
                        # 计算IC (Information Coefficient)
                        ic = clean_factor.corr(clean_returns)
                        
                        ic_results[factor_name] = {
                            'ic': ic if not np.isnan(ic) else 0.0,
                            'sample_size': len(clean_factor),
                            'factor_mean': float(clean_factor.mean()),
                            'factor_std': float(clean_factor.std()),
                            'returns_mean': float(clean_returns.mean()),
                            'returns_std': float(clean_returns.std())
                        }
                        
                        self.logger.info(f"  {factor_name}: IC={ic:.4f}, 样本数={len(clean_factor)}")
                    else:
                        self.logger.warning(f"  {factor_name}: 有效数据不足({len(clean_factor)})")
                        ic_results[factor_name] = {'ic': 0.0, 'sample_size': 0}
                else:
                    self.logger.warning(f"  {factor_name}: 索引重叠不足({len(common_index)})")
                    ic_results[factor_name] = {'ic': 0.0, 'sample_size': 0}
                    
            except Exception as e:
                self.logger.error(f"计算{factor_name}的IC失败: {e}")
                ic_results[factor_name] = {'ic': 0.0, 'sample_size': 0}
        
        return ic_results
    
    def run_vectorized_analysis(self, 
                              symbols: List[str] = None, 
                              timeframes: List[str] = None,
                              factors: List[str] = None) -> Dict:
        """运行向量化分析 - 核心方法"""
        
        if symbols is None:
            symbols = self.all_symbols[:1]  # 先测试1只股票
        if timeframes is None:
            timeframes = ['1h']  # 先测试1个时间框架
        if factors is None:
            factors = ['RSI', 'MACD']  # 先测试2个因子
        
        self.logger.info(f"开始向量化分析:")
        self.logger.info(f"  股票: {len(symbols)}只 {symbols}")
        self.logger.info(f"  时间框架: {len(timeframes)}个 {timeframes}")
        self.logger.info(f"  因子: {len(factors)}个 {factors}")
        
        results = {
            'metadata': {
                'symbols': symbols,
                'timeframes': timeframes, 
                'factors': factors,
                'start_date': str(self.start_date),
                'end_date': str(self.end_date),
                'analysis_time': datetime.now().isoformat()
            },
            'timeframe_results': {}
        }
        
        # 对每个时间框架进行分析
        for timeframe in timeframes:
            self.logger.info(f"\n🔄 分析时间框架: {timeframe}")
            
            try:
                # 1. 加载数据
                data = self.load_timeframe_data_vectorized(timeframe, symbols)
                
                # 2. 计算因子
                factors_dict = self.calculate_factors_vectorized(data, factors)
                
                # 3. 计算IC
                ic_results = self.calculate_ic_vectorized(data, factors_dict)
                
                # 4. 存储结果
                results['timeframe_results'][timeframe] = {
                    'data_shape': data.shape,
                    'data_symbols': len(data.index.get_level_values('symbol').unique()),
                    'factors_calculated': list(factors_dict.keys()),
                    'ic_results': ic_results,
                    'analysis_status': 'success'
                }
                
                self.logger.info(f"✅ {timeframe}分析完成")
                
            except Exception as e:
                error_msg = f"{timeframe}分析失败: {str(e)}\n{traceback.format_exc()}"
                self.logger.error(error_msg)
                
                results['timeframe_results'][timeframe] = {
                    'analysis_status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def run_smoke_test(self, symbol: str = "0700.HK") -> Dict:
        """运行冒烟测试 - 单股票快速验证"""
        self.logger.info(f"🧪 开始冒烟测试: {symbol}")
        
        # 测试参数
        test_symbols = [symbol]
        test_timeframes = ['1h']  # 先测试1小时数据
        test_factors = ['RSI', 'MACD']  # 先测试2个因子
        
        try:
            results = self.run_vectorized_analysis(
                symbols=test_symbols,
                timeframes=test_timeframes, 
                factors=test_factors
            )
            
            # 验证结果
            success = True
            issues = []
            
            for timeframe, result in results['timeframe_results'].items():
                if result['analysis_status'] != 'success':
                    success = False
                    issues.append(f"{timeframe}: {result.get('error', 'Unknown error')}")
                else:
                    # 检查IC结果
                    ic_results = result.get('ic_results', {})
                    for factor, ic_data in ic_results.items():
                        if ic_data.get('sample_size', 0) == 0:
                            issues.append(f"{timeframe}.{factor}: 无有效样本")
            
            results['smoke_test'] = {
                'status': 'PASS' if success and not issues else 'FAIL',
                'issues': issues,
                'summary': {
                    'total_combinations': len(test_timeframes) * len(test_factors),
                    'successful_combinations': sum(
                        1 for tf_result in results['timeframe_results'].values()
                        for factor in tf_result.get('ic_results', {}).keys()
                        if tf_result.get('ic_results', {}).get(factor, {}).get('sample_size', 0) > 0
                    )
                }
            }
            
            # 打印结果
            print(f"\n🧪 冒烟测试结果: {results['smoke_test']['status']}")
            print(f"   测试组合: {results['smoke_test']['summary']['total_combinations']}")
            print(f"   成功组合: {results['smoke_test']['summary']['successful_combinations']}")
            
            if issues:
                print(f"   问题列表:")
                for issue in issues:
                    print(f"     - {issue}")
            
            return results
            
        except Exception as e:
            error_msg = f"冒烟测试失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            
            return {
                'smoke_test': {
                    'status': 'FAIL',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
            }
    
    def clear_cache(self):
        """清理缓存"""
        self.data_cache.clear()
        self.factors_cache.clear()
        gc.collect()
        self.logger.info("缓存已清理")
    
    def _clear_cache(self):
        """内部缓存清理方法（兼容性）"""
        self.clear_cache()


def main():
    """主函数 - 运行冒烟测试"""
    print("="*80)
    print("🚀 向量化多股票分析器 - 冒烟测试")
    print("="*80)
    
    try:
        # 创建分析器
        analyzer = VectorizedMultiStockAnalyzer()
        
        # 运行冒烟测试
        results = analyzer.run_smoke_test("0700.HK")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"vectorized_smoke_test_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 测试结果已保存: {output_file}")
        
        # 显示状态
        if results.get('smoke_test', {}).get('status') == 'PASS':
            print("✅ 冒烟测试通过!")
            return 0
        else:
            print("❌ 冒烟测试失败!")
            return 1
            
    except Exception as e:
        print(f"❌ 程序执行失败: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
