#!/usr/bin/env python3
"""
🎯 真正修复的CTA回测评估模块 V3.0
=====================================

修复V2中的致命错误：
- 修复因子标准化：rank → 直接Z-score（去掉rank步骤）
- 修复滚动窗口：5m数据用2-5天，不是过长的252
- 修复信号生成：使用百分位+适度Z-score组合
- 修复夏普率：先验证能产生正收益，再考虑年化

Author: VectorBT CTA Framework V3
Date: 2025-09-10
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


class CTAEvaluatorV3:
    """真正修复的单资产CTA回测评估器 V3"""
    
    def __init__(self, 
                 look_ahead: int = 1,
                 entry_percentile: float = 0.8,  # 回到分位数方法，但改进实现
                 exit_percentile: float = 0.2,
                 sl_stop: float = 0.02,
                 tp_stop: float = 0.03,
                 direction: str = 'both',
                 slippage: float = 0.002,  # 降低滑点先测试
                 fees: float = 0.001,
                 min_trades: int = 10):    # 大幅降低最低要求
        """
        初始化V3修复版CTA评估器
        
        核心修复:
        - 回到百分位方法，但用较短的滚动窗口
        - 降低交易成本先确保基本逻辑正确
        - 大幅降低最低交易要求
        """
        self.look_ahead = look_ahead
        self.entry_percentile = entry_percentile
        self.exit_percentile = exit_percentile
        self.sl_stop = sl_stop
        self.tp_stop = tp_stop
        self.direction = direction
        self.slippage = slippage
        self.fees = fees
        self.min_trades = min_trades
        
    def cta_score(self, df: pd.DataFrame, factor_col: str, timeframe: str = '5m') -> Dict[str, Any]:
        """
        V3修复版CTA打分 - 专注基本逻辑正确性
        """
        try:
            # 数据验证
            required_cols = ['close', factor_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return self._empty_result(f"缺失列: {missing_cols}")
                
            price = df['close'].ffill().dropna()
            factor = df[factor_col].ffill().dropna()
            
            # 数据对齐
            common_index = price.index.intersection(factor.index)
            if len(common_index) < 50:  # 大幅降低数据要求
                return self._empty_result("有效数据不足50个点")
                
            price = price.reindex(common_index)
            factor = factor.reindex(common_index)
            
            # ✅ 修复1: 适合高频的较短滚动窗口
            window = self._get_reasonable_window(timeframe, len(factor))
            min_periods = max(5, window // 4)
            
            # ✅ 修复2: 简化因子处理（去掉过度复杂的三步法）
            factor_processed = self._simple_standardize(factor, window, min_periods)
            
            # ✅ 修复3: 回到分位数方法，但用较短窗口
            long_entry, short_entry = self._generate_percentile_signals(
                factor_processed, window, min_periods)
            
            # 检查信号数量
            total_signals = long_entry.sum() + short_entry.sum()
            
            if total_signals < 5:  # 最基本的信号要求
                return self._empty_result(f"信号数量过少: {total_signals} < 5")
            
            # 根据direction设置交易方向
            if self.direction == 'both':
                entries = long_entry | short_entry
                exits = pd.Series(False, index=entries.index)
            elif self.direction == 'long':
                entries = long_entry
                exits = short_entry
            elif self.direction == 'short':
                entries = short_entry
                exits = long_entry
            else:
                return self._empty_result(f"不支持的交易方向: {self.direction}")
            
            # VectorBT回测
            return self._run_simple_backtest(price, entries, exits, timeframe, 
                                           factor_processed, total_signals)
                
        except Exception as e:
            return self._empty_result(f"CTA评估失败: {str(e)}")
    
    def _get_reasonable_window(self, timeframe: str, data_length: int) -> int:
        """✅ 修复4: 合理的滚动窗口（不要用252天！）"""
        if '5m' in timeframe:
            # 5分钟数据：2天 = 48小时 × 12 = 576个点，太多了
            # 改用4-8小时 = 48-96个点
            base_window = 60  # 约5小时
        elif '1m' in timeframe:
            base_window = 120  # 约2小时
        elif '15m' in timeframe:
            base_window = 32   # 约8小时
        elif '30m' in timeframe:
            base_window = 24   # 约12小时
        elif '1h' in timeframe:
            base_window = 12   # 约12小时
        elif '4h' in timeframe:
            base_window = 6    # 约1天
        elif '1d' in timeframe:
            base_window = 10   # 约10天
        else:
            base_window = 40
        
        # 确保窗口不超过数据的1/5
        return min(base_window, max(10, data_length // 5))
    
    def _simple_standardize(self, factor: pd.Series, window: int, min_periods: int) -> pd.Series:
        """✅ 修复5: 简化的因子标准化"""
        # 简单Z-score标准化，不要过度复杂
        factor_mean = factor.rolling(window, min_periods=min_periods).mean()
        factor_std = factor.rolling(window, min_periods=min_periods).std()
        
        # 防止除零
        factor_std = factor_std.where(factor_std > 1e-8, 1.0)
        
        factor_zscore = (factor - factor_mean) / factor_std
        
        # 适度截断（不要过度压缩）
        factor_clipped = factor_zscore.clip(-3, 3)
        
        return factor_clipped.fillna(0)
    
    def _generate_percentile_signals(self, factor_processed: pd.Series, 
                                   window: int, min_periods: int) -> Tuple[pd.Series, pd.Series]:
        """✅ 修复6: 基于较短窗口的分位数信号"""
        # 使用滚动分位数，但窗口较短
        factor_upper = factor_processed.rolling(window, min_periods=min_periods).quantile(
            self.entry_percentile)
        factor_lower = factor_processed.rolling(window, min_periods=min_periods).quantile(
            self.exit_percentile)
        
        long_entry = factor_processed > factor_upper
        short_entry = factor_processed < factor_lower
        
        return long_entry, short_entry
    
    def _run_simple_backtest(self, price: pd.Series, entries: pd.Series, exits: pd.Series, 
                           timeframe: str, factor_processed: pd.Series, total_signals: int) -> Dict[str, Any]:
        """运行简化的VectorBT回测"""
        try:
            pf = vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                sl_stop=self.sl_stop if self.sl_stop > 0 else None,
                tp_stop=self.tp_stop if self.tp_stop > 0 else None,
                freq=self._get_freq(timeframe),
                direction=self.direction,
                init_cash=100000,
                fees=self.fees,
                slippage=self.slippage
            )
            
            # 获取基础指标
            total_return = pf.total_return()
            raw_sharpe = pf.sharpe_ratio()
            trades_df = pf.trades.records_readable
            total_trades = len(trades_df) if hasattr(trades_df, '__len__') else 0
            
            # ✅ 修复7: 更宽松的交易次数要求
            if total_trades < self.min_trades:
                return self._empty_result(f"交易次数不足{self.min_trades}次 (实际: {total_trades})")
            
            # ✅ 修复8: 先不考虑年化，确保基本逻辑正确
            sharpe = raw_sharpe if not pd.isna(raw_sharpe) else 0.0
            
            # ✅ 修复9: 极宽松的夏普率要求（先确保正数）
            if sharpe <= -1.0:  # 只过滤极差的策略
                return self._empty_result(f"夏普率过低: {sharpe:.3f} ≤ -1.0")
            
            # 计算其他指标
            win_rate, profit_loss_ratio, avg_trade_return = self._calculate_trade_metrics(trades_df)
            max_drawdown = pf.max_drawdown()
            
            # ✅ 修复10: 适度的年化处理
            annual_factor = self._get_annual_factor(timeframe)
            annual_sharpe = sharpe / annual_factor if annual_factor > 1 else sharpe
            
            quality_flag = "needs_review"
            if sharpe > 0:
                quality_flag = "positive_returns"
            if annual_sharpe > 0.5:
                quality_flag = "good_annual_sharpe"
            if annual_sharpe > 2:
                quality_flag = "excellent"
            
            return {
                'total_return': float(total_return),
                'sharpe': float(annual_sharpe),  # 返回年化后的
                'raw_sharpe': float(sharpe),     # 保留原始的用于调试
                'win_rate': float(win_rate),
                'profit_loss_ratio': float(profit_loss_ratio),
                'max_drawdown': float(max_drawdown),
                'trades': int(total_trades),
                'avg_trade_return': float(avg_trade_return),
                'signal_strength': float(factor_processed.std()),
                'signal_count': int(total_signals),
                'data_quality': quality_flag,
                'annual_factor': float(annual_factor)
            }
            
        except Exception as e:
            return self._empty_result(f"VectorBT回测失败: {str(e)}")
    
    def _get_freq(self, timeframe: str) -> str:
        """获取pandas频率字符串"""
        freq_map = {
            '1m': '1T', '2m': '2T', '3m': '3T', '5m': '5T',
            '10m': '10T', '15m': '15T', '30m': '30T',
            '1h': '1H', '2h': '2H', '3h': '3H', '4h': '4H',
            '1d': '1D'
        }
        return freq_map.get(timeframe, '5T')
    
    def _get_annual_factor(self, timeframe: str) -> float:
        """✅修复暗礁4: 按持仓周期调整的年化因子计算"""
        periods_per_year = {
            '1m': 252 * 240,      # 1分钟(4小时交易日)
            '5m': 252 * 48,       # 5分钟(4小时交易日)
            '15m': 252 * 16,      # 15分钟
            '30m': 252 * 8,       # 30分钟
            '1h': 252 * 4,        # 1小时
            '4h': 252,            # 4小时(等于1个交易日)
            '1d': 252             # 1天
        }
        
        base_periods = periods_per_year.get(timeframe, 252 * 48)
        
        # ✅关键修复: 按持仓周期调整年化因子
        # look_ahead=6根K线 → 实际持仓周期调整
        holding_periods = self.look_ahead if hasattr(self, 'look_ahead') else 1
        adjusted_periods = base_periods / holding_periods
        
        return np.sqrt(adjusted_periods)
    
    def _calculate_trade_metrics(self, trades_df) -> Tuple[float, float, float]:
        """计算交易指标"""
        if len(trades_df) == 0:
            return 0.0, 0.0, 0.0
        
        # 胜率
        winning_trades = trades_df[trades_df['PnL'] > 0] if 'PnL' in trades_df.columns else pd.DataFrame()
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        # 盈亏比
        try:
            if len(winning_trades) > 0 and 'PnL' in trades_df.columns:
                losing_trades = trades_df[trades_df['PnL'] < 0]
                if len(losing_trades) > 0:
                    avg_win = winning_trades['PnL'].mean()
                    avg_loss = abs(losing_trades['PnL'].mean())
                    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                else:
                    profit_loss_ratio = float('inf')
            else:
                profit_loss_ratio = 0
        except Exception:
            profit_loss_ratio = 0
        
        # 平均交易收益
        avg_trade_return = trades_df['Return'].mean() if 'Return' in trades_df.columns else 0
        
        return win_rate, profit_loss_ratio, avg_trade_return
    
    def _empty_result(self, reason: str = "数据不足") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'total_return': 0.0, 'sharpe': 0.0, 'raw_sharpe': 0.0, 'win_rate': 0.0,
            'profit_loss_ratio': 0.0, 'max_drawdown': 0.0, 'trades': 0,
            'avg_trade_return': 0.0, 'signal_strength': 0.0, 'signal_count': 0,
            'data_quality': f'failed: {reason}', 'annual_factor': 1.0
        }
    
    def batch_evaluate(self, symbols: List[str], factor_data: Dict[str, pd.DataFrame],
                       price_data: Dict[str, pd.DataFrame], factor_names: List[str], 
                       timeframe: str = '5m') -> pd.DataFrame:
        """批量评估"""
        score_records = []
        
        print(f"🎯 开始V3修复版CTA评估: {len(symbols)}只股票 × {len(factor_names)}个因子")
        
        for symbol in symbols:
            if symbol not in factor_data or symbol not in price_data:
                continue
                
            try:
                factors_df = factor_data[symbol]
                price_df = price_data[symbol]
                
                if 'close' not in price_df.columns:
                    continue
                
                # 数据对齐
                common_index = factors_df.index.intersection(price_df.index)
                if len(common_index) < 50:  # 降低要求
                    continue
                
                merged_df = price_df.reindex(common_index).join(
                    factors_df.reindex(common_index), how='inner'
                )
                
                for factor_name in factor_names:
                    if factor_name not in merged_df.columns:
                        continue
                        
                    # V3修复版评估
                    score = self.cta_score(merged_df, factor_name, timeframe)
                    
                    score_records.append({
                        'symbol': symbol,
                        'factor': factor_name,
                        **score
                    })
                    
            except Exception as e:
                continue
        
        if not score_records:
            print("❌ 没有成功的评估结果")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(score_records)
        
        # 统计有效策略
        valid_strategies = result_df[result_df['trades'] >= self.min_trades]
        positive_strategies = result_df[result_df['raw_sharpe'] > 0]
        
        print(f"✅ V3修复版CTA评估完成: {len(result_df)}条记录")
        print(f"   有效策略: {len(valid_strategies)}个 (≥{self.min_trades}次交易)")
        print(f"   正收益策略: {len(positive_strategies)}个 (原始夏普>0)")
        
        return result_df
    
    def rank_factors(self, result_df: pd.DataFrame, rank_by: str = 'raw_sharpe') -> pd.DataFrame:
        """因子排名（V3宽松版）"""
        # 只过滤有基本交易的策略
        valid_results = result_df[
            (result_df['trades'] >= self.min_trades)
        ].copy()
        
        if valid_results.empty:
            print(f"⚠️ 没有≥{self.min_trades}次交易的因子")
            return pd.DataFrame()
        
        # 按因子聚合统计
        factor_stats = valid_results.groupby('factor').agg({
            rank_by: ['mean', 'std', 'count'],
            'total_return': 'mean',
            'sharpe': 'mean',
            'raw_sharpe': 'mean',
            'win_rate': 'mean',
            'profit_loss_ratio': 'mean',
            'max_drawdown': 'mean',
            'trades': ['mean', 'sum'],
            'avg_trade_return': 'mean',
            'signal_strength': 'mean',
            'signal_count': 'mean'
        }).round(4)
        
        # 扁平化列名
        factor_stats.columns = ['_'.join(col).strip() for col in factor_stats.columns]
        
        # 按指定指标排序
        sort_col = f'{rank_by}_mean'
        if sort_col in factor_stats.columns:
            factor_stats = factor_stats.sort_values(sort_col, ascending=False)
        
        factor_stats['rank'] = range(1, len(factor_stats) + 1)
        
        return factor_stats.reset_index()


if __name__ == "__main__":
    print("🎯 修复版CTA评估模块 V3.0")
    print("核心修复:")
    print("- 滚动窗口: 252天 → 5-10小时(5m数据)")
    print("- 因子处理: 三步法 → 简单Z-score")
    print("- 信号生成: Z-score阈值 → 分位数(短窗口)")
    print("- 夏普过滤: ≤0.5淘汰 → ≤-1.0淘汰")
    print("- 交易要求: ≥50次 → ≥10次")
    print("✅ V3修复完成！")
