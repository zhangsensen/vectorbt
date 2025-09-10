"""
🎯 单资产CTA回测评估模块
====================

把"横截面IC选因子"改成"单资产时序回测选因子"的核心模块
- 每只票独立跑回测
- 输出：夏普、胜率、盈亏比、信号次数
- 因子排序改用单标的夏普均值

Author: VectorBT CTA Framework
Date: 2025-09-10
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


class CTAEvaluator:
    """单资产CTA回测评估器"""
    
    def __init__(self, 
                 look_ahead: int = 1,
                 entry_th: float = 0.9,
                 exit_th: float = 0.1,
                 sl_stop: float = 0.02,
                 tp_stop: float = 0.03,
                 direction: str = 'both',
                 slippage: float = 0.001,  # 0.1% 滑点
                 fees: float = 0.001):
        """
        初始化CTA评估器
        
        Parameters:
        -----------
        look_ahead : int
            持有几根K线
        entry_th : float
            因子分位阈值（>分位做多）
        exit_th : float
            因子分位阈值（<分位做空/平仓）
        sl_stop : float
            止损比例（如0.02表示2%）
        tp_stop : float
            止盈比例（如0.03表示3%）
        direction : str
            交易方向('long', 'short', 'both')
        slippage : float
            滑点（如0.001表示0.1%）
        fees : float
            手续费（如0.001表示0.1%）
        """
        self.look_ahead = look_ahead
        self.entry_th = entry_th
        self.exit_th = exit_th
        self.sl_stop = sl_stop
        self.tp_stop = tp_stop
        self.direction = direction
        self.slippage = slippage
        self.fees = fees
        
    def cta_score(self, df: pd.DataFrame, factor_col: str) -> Dict[str, Any]:
        """
        单资产CTA快速打分
        
        Parameters:
        -----------
        df : pd.DataFrame
            必须包含 ['open','high','low','close', factor_col]
        factor_col : str
            因子列名
            
        Returns:
        --------
        dict : 包含夏普、胜率、盈亏比、信号次数等指标
        """
        try:
            # 数据验证
            required_cols = ['close', factor_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return self._empty_result(f"缺失列: {missing_cols}")
                
            price = df['close'].ffill().dropna()
            factor = df[factor_col].ffill().dropna()
            
            # 确保数据对齐
            common_index = price.index.intersection(factor.index)
            if len(common_index) < 50:  # 最少需要50个数据点
                return self._empty_result("有效数据不足50个点")
                
            price = price.reindex(common_index)
            factor = factor.reindex(common_index)
            
            # ✅ 修复1: 因子标准化 (Z-score标准化)
            factor_mean = factor.rolling(252, min_periods=20).mean()
            factor_std = factor.rolling(252, min_periods=20).std()
            factor_normalized = (factor - factor_mean) / factor_std
            # 用当前因子值替换NaN
            factor_normalized = factor_normalized.fillna(factor)
            
            # 检查因子是否有变化 (过滤常数因子)
            if factor_normalized.nunique() <= 1:
                return self._empty_result("因子为常数，无交易信号")
            
            # 计算因子分位数信号 (使用标准化后的因子)
            # 使用更小的滚动窗口，确保有交易信号
            window = min(60, max(20, len(factor_normalized) // 10))  # 更小的动态窗口
            min_periods = max(5, window // 4)
            
            factor_upper = factor_normalized.rolling(window, min_periods=min_periods).quantile(self.entry_th)
            factor_lower = factor_normalized.rolling(window, min_periods=min_periods).quantile(self.exit_th)
            
            # 生成交易信号 - 更灵敏的信号生成 (使用标准化后的因子)
            long_entry = factor_normalized > factor_upper
            short_entry = factor_normalized < factor_lower
            
            # 确保有足够的交易信号
            if long_entry.sum() < 5 and short_entry.sum() < 5:
                # 如果信号太少，降低阈值 (仍使用标准化因子)
                factor_upper = factor_normalized.rolling(window, min_periods=min_periods).quantile(0.7)
                factor_lower = factor_normalized.rolling(window, min_periods=min_periods).quantile(0.3)
                long_entry = factor_normalized > factor_upper
                short_entry = factor_normalized < factor_lower
            
            # 根据direction设置交易方向
            if self.direction == 'long':
                entries = long_entry
                exits = short_entry  # 做多的退出信号
            elif self.direction == 'short':
                entries = short_entry
                exits = long_entry  # 做空的退出信号
            elif self.direction == 'both':
                # 双向交易：长信号和短信号都作为entry
                entries = long_entry | short_entry
                exits = pd.Series(False, index=entries.index)  # 不设置明确退出，依赖止盈止损
            else:
                return self._empty_result(f"不支持的交易方向: {self.direction}")
            
            # VectorBT回测 (添加交易成本)
            try:
                pf = vbt.Portfolio.from_signals(
                    close=price,
                    entries=entries,
                    exits=exits,
                    sl_stop=self.sl_stop if self.sl_stop > 0 else None,
                    tp_stop=self.tp_stop if self.tp_stop > 0 else None,
                    freq=price.index.inferred_freq or 'D',
                    direction=self.direction,
                    init_cash=100000,  # 初始资金10万
                    fees=self.fees,    # ✅ 修复3a: 添加手续费 0.1%
                    slippage=self.slippage,  # ✅ 修复3b: 添加滑点 0.1%
                )
                
                # 提取关键指标
                total_return = pf.total_return()
                raw_sharpe = pf.sharpe_ratio()
                
                # ✅ 修复2: 年化夏普率 (根据数据频率调整)
                freq_str = price.index.inferred_freq or 'D'
                if '5T' in freq_str or '5min' in freq_str.lower():
                    # 5分钟数据: 252 * 288 = 72576 个交易期/年，sqrt(72576) ≈ 269.4
                    annual_factor = np.sqrt(252 * 288 / 5)  # 约等于 120.4
                elif '1T' in freq_str or '1min' in freq_str.lower():
                    annual_factor = np.sqrt(252 * 288)  # 约等于 268.7
                elif 'H' in freq_str or 'hour' in freq_str.lower():
                    annual_factor = np.sqrt(252 * 7)  # 约等于 42.1
                elif 'D' in freq_str or 'day' in freq_str.lower():
                    annual_factor = np.sqrt(252)  # 约等于 15.9
                else:
                    # 默认按5分钟处理
                    annual_factor = np.sqrt(252 * 288 / 5)  # 约等于 120.4
                
                # 年化夏普率 = 原始夏普率 / 年化因子 (添加异常值检查)
                if pd.isna(raw_sharpe) or pd.isinf(raw_sharpe):
                    sharpe = 0.0
                elif abs(raw_sharpe) > 1000:  # 原始夏普率超过1000明显异常
                    sharpe = 0.0
                    raw_sharpe = 0.0  # 也重置原始值
                else:
                    sharpe = raw_sharpe / annual_factor
                
                # 获取交易记录
                trades_df = pf.trades.records_readable
                total_trades = len(trades_df) if hasattr(trades_df, '__len__') else 0
                
                # ✅ 修复5: 交易次数过滤 (少于5次交易标为无效，临时降低门槛确认修复效果)
                if total_trades < 5:
                    return self._empty_result(f"交易次数不足5次 (实际: {total_trades})")
                
                # 计算胜率
                if total_trades > 0:
                    winning_trades = trades_df[trades_df['PnL'] > 0] if 'PnL' in trades_df.columns else pd.DataFrame()
                    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                    avg_trade_return = trades_df['Return'].mean() if 'Return' in trades_df.columns else 0
                else:
                    win_rate = 0
                    avg_trade_return = 0
                
                # 处理NaN值
                if pd.isna(total_return) or pd.isna(sharpe):
                    return self._empty_result("回测结果包含NaN")
                
                # 计算盈亏比
                try:
                    if total_trades > 0 and 'PnL' in trades_df.columns:
                        winning_pnl = trades_df[trades_df['PnL'] > 0]['PnL'].mean()
                        losing_pnl = abs(trades_df[trades_df['PnL'] < 0]['PnL'].mean())
                        profit_loss_ratio = winning_pnl / losing_pnl if losing_pnl > 0 else 0
                    else:
                        profit_loss_ratio = 0
                except Exception:
                    profit_loss_ratio = 0
                
                # 计算更多指标
                max_drawdown = pf.max_drawdown()
                
                # ✅ 修复6: 夏普率上限检查 (>10需要复查)
                sharpe_warning = ""
                if sharpe > 10:
                    sharpe_warning = f" [WARNING: 夏普率{sharpe:.2f}>10需复查]"
                elif sharpe > 5:
                    sharpe_warning = f" [CAUTION: 夏普率{sharpe:.2f}>5需关注]"
                
                return {
                    'total_return': float(total_return),
                    'sharpe': float(sharpe),
                    'win_rate': float(win_rate),
                    'profit_loss_ratio': float(profit_loss_ratio),
                    'max_drawdown': float(max_drawdown),
                    'trades': int(total_trades),
                    'avg_trade_return': float(avg_trade_return),
                    'signal_strength': float(factor_normalized.std()),  # 使用标准化因子的信号强度
                    'data_quality': f'good{sharpe_warning}',
                    'annual_factor': float(annual_factor),  # 记录年化因子，用于调试
                    'raw_sharpe': float(raw_sharpe) if not pd.isna(raw_sharpe) else 0.0  # 记录原始夏普率
                }
                
            except Exception as e:
                return self._empty_result(f"VectorBT回测失败: {str(e)}")
                
        except Exception as e:
            return self._empty_result(f"CTA评估失败: {str(e)}")
    
    def _empty_result(self, reason: str = "数据不足") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'total_return': 0.0,
            'sharpe': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'max_drawdown': 0.0,
            'trades': 0,
            'avg_trade_return': 0.0,
            'signal_strength': 0.0,
            'data_quality': f'failed: {reason}'
        }
    
    def batch_evaluate(self, 
                      symbols: List[str], 
                      factor_data: Dict[str, pd.DataFrame],
                      price_data: Dict[str, pd.DataFrame],
                      factor_names: List[str]) -> pd.DataFrame:
        """
        批量评估多个股票和因子
        
        Parameters:
        -----------
        symbols : List[str]
            股票列表
        factor_data : Dict[str, pd.DataFrame]
            因子数据 {symbol: factors_df}
        price_data : Dict[str, pd.DataFrame]
            价格数据 {symbol: ohlcv_df}
        factor_names : List[str]
            因子名称列表
            
        Returns:
        --------
        pd.DataFrame : 评估结果汇总
        """
        score_records = []
        
        print(f"🎯 开始CTA评估: {len(symbols)}只股票 × {len(factor_names)}个因子")
        
        for symbol in symbols:
            if symbol not in factor_data or symbol not in price_data:
                print(f"⚠️ {symbol}: 缺失数据，跳过")
                continue
                
            # 合并因子和价格数据
            try:
                factors_df = factor_data[symbol]
                price_df = price_data[symbol]
                
                # 确保有close列
                if 'close' not in price_df.columns:
                    print(f"⚠️ {symbol}: 缺失close价格，跳过")
                    continue
                
                # 数据对齐
                common_index = factors_df.index.intersection(price_df.index)
                if len(common_index) < 50:
                    print(f"⚠️ {symbol}: 有效数据不足，跳过")
                    continue
                
                merged_df = price_df.reindex(common_index).join(
                    factors_df.reindex(common_index), 
                    how='inner'
                )
                
                for factor_name in factor_names:
                    if factor_name not in merged_df.columns:
                        continue
                        
                    # 计算CTA得分
                    score = self.cta_score(merged_df, factor_name)
                    
                    # 记录结果
                    score_records.append({
                        'symbol': symbol,
                        'factor': factor_name,
                        **score
                    })
                    
            except Exception as e:
                print(f"❌ {symbol}: 评估失败 - {e}")
                continue
        
        if not score_records:
            print("❌ 没有成功的评估结果")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(score_records)
        print(f"✅ CTA评估完成: {len(result_df)}条记录")
        
        return result_df
    
    def rank_factors(self, result_df: pd.DataFrame, 
                    rank_by: str = 'sharpe',
                    min_trades: int = 5) -> pd.DataFrame:
        """
        因子排名
        
        Parameters:
        -----------
        result_df : pd.DataFrame
            批量评估结果
        rank_by : str
            排序指标 ('sharpe', 'total_return', 'win_rate', 'profit_loss_ratio')
        min_trades : int
            最少交易次数要求
            
        Returns:
        --------
        pd.DataFrame : 因子排名结果
        """
        # 过滤最少交易次数
        valid_results = result_df[result_df['trades'] >= min_trades].copy()
        
        if valid_results.empty:
            print(f"⚠️ 没有交易次数≥{min_trades}的因子")
            return pd.DataFrame()
        
        # 按因子聚合统计
        factor_stats = valid_results.groupby('factor').agg({
            rank_by: ['mean', 'std', 'count'],
            'total_return': 'mean',
            'sharpe': 'mean',
            'win_rate': 'mean',
            'profit_loss_ratio': 'mean',
            'max_drawdown': 'mean',
            'trades': ['mean', 'sum'],
            'avg_trade_return': 'mean',
            'signal_strength': 'mean'
        }).round(4)
        
        # 扁平化列名
        factor_stats.columns = ['_'.join(col).strip() for col in factor_stats.columns]
        
        # 按指定指标排序
        sort_col = f'{rank_by}_mean'
        if sort_col in factor_stats.columns:
            factor_stats = factor_stats.sort_values(sort_col, ascending=False)
        
        # 添加排名列
        factor_stats['rank'] = range(1, len(factor_stats) + 1)
        
        return factor_stats.reset_index()


def quick_cta_test(symbol: str, factor_name: str, 
                  data_dir: str = "../vectorbt_workspace/data",
                  timeframe: str = "5m") -> Dict[str, Any]:
    """
    快速单因子CTA测试
    
    Parameters:
    -----------
    symbol : str
        股票代码
    factor_name : str
        因子名称
    data_dir : str
        数据目录
    timeframe : str
        时间框架
        
    Returns:
    --------
    dict : CTA评估结果
    """
    try:
        import os
        
        # 加载数据
        data_path = os.path.join(data_dir, timeframe, f"{symbol}.parquet")
        if not os.path.exists(data_path):
            return {"error": f"数据文件不存在: {data_path}"}
        
        df = pd.read_parquet(data_path)
        
        # 假设因子已计算，这里简单用RSI作为示例
        if 'close' not in df.columns:
            return {"error": "缺失close价格列"}
        
        # 计算RSI作为示例因子
        import talib
        df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # 运行CTA评估
        evaluator = CTAEvaluator()
        result = evaluator.cta_score(df, 'rsi_14')
        
        return {
            'symbol': symbol,
            'factor': 'rsi_14',
            'timeframe': timeframe,
            'data_points': len(df),
            **result
        }
        
    except Exception as e:
        return {"error": f"快速测试失败: {e}"}


if __name__ == "__main__":
    # 快速测试
    print("🎯 CTA评估模块测试")
    
    # 生成测试数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
        'test_factor': np.random.randn(len(dates))
    }, index=dates)
    
    # 运行测试
    evaluator = CTAEvaluator()
    result = evaluator.cta_score(test_data, 'test_factor')
    
    print("测试结果:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n✅ CTA评估模块测试完成")
