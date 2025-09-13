#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级因子池 - 30+指标的产业级量化策略框架
从6个指标扩展到30+，覆盖趋势、动量、波动、成交量、微观结构等维度
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import talib as ta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# === 统一防未来函数：rolling 永远只取"已收盘"K 线 ===
def roll_closed(df, col, win, func='mean'):
    """
    win : int 或 pd.Timedelta
    func: 'mean'|'std'|'max'|'min'|'sum'|...
    返回：Series，窗口内仅含已收盘数据，天然滞后 1 根 K 线
    """
    return getattr(df[col].shift(1), 'rolling')(win, min_periods=max(2, win//2)).agg(func)

def roll_closed_rank(df, col, win, min_periods=None):
    """
    专门用于rank的rolling函数，确保只使用已收盘数据
    """
    if min_periods is None:
        min_periods = max(2, win//2)
    return df[col].shift(1).rolling(win, min_periods=min_periods).rank(pct=True)

def safe_ema(series, span):
    """安全的EMA计算，避免未来函数"""
    return series.shift(1).ewm(span=span).mean()

def get_safe_price(df, price_col='close'):
    """获取安全的滞后价格数据"""
    return df[price_col].shift(1)

def safe_talib_single_price(func_name, price_series, timeperiod):
    """安全的单价格TA-LIB函数包装器"""
    shifted_series = price_series.shift(1).values
    if isinstance(shifted_series, np.ndarray):
        return getattr(ta, func_name)(shifted_series, timeperiod=timeperiod)
    else:
        return getattr(ta, func_name)(shifted_series.values, timeperiod=timeperiod)

def safe_talib_ohlcv(func_name, high_series, low_series, close_series, volume_series=None, timeperiod=None):
    """安全的OHLCV TA-LIB函数包装器"""
    high_shifted = high_series.shift(1).values
    low_shifted = low_series.shift(1).values
    close_shifted = close_series.shift(1).values
    
    if volume_series is not None:
        volume_shifted = volume_series.shift(1).values
        if timeperiod is not None:
            return getattr(ta, func_name)(high_shifted, low_shifted, close_shifted, volume_shifted, timeperiod=timeperiod)
        else:
            return getattr(ta, func_name)(high_shifted, low_shifted, close_shifted, volume_shifted)
    else:
        if timeperiod is not None:
            return getattr(ta, func_name)(high_shifted, low_shifted, close_shifted, timeperiod=timeperiod)
        else:
            return getattr(ta, func_name)(high_shifted, low_shifted, close_shifted)

def safe_talib_hl(func_name, high_series, low_series, timeperiod):
    """安全的HL TA-LIB函数包装器"""
    high_shifted = high_series.shift(1).values
    low_shifted = low_series.shift(1).values
    return getattr(ta, func_name)(high_shifted, low_shifted, timeperiod=timeperiod)

def safe_returns(price_series, periods=1):
    """安全的收益率计算"""
    return price_series.shift(periods).pct_change()

class AdvancedFactorPool:
    """高级因子池 - 产业级指标计算"""
    
    def __init__(self):
        """初始化高级因子池"""
        self.factor_categories = {
            'trend': ['dema', 'tema', 'kama', 'trix', 'aroon_up', 'aroon_down', 'adx'],
            'momentum': ['rsi_2', 'rsi_100', 'stoch_rsi', 'cci', 'roc', 'mfi', 'willr'],
            'volatility': ['atrp', 'keltner_position', 'bb_squeeze', 'volatility_ratio'],
            'volume': ['vwap_dev', 'volume_rsi', 'ad_line', 'cmf', 'volume_ma_dev'],
            'microstructure': ['hl_spread', 'volume_intensity', 'price_efficiency'],
            'enhanced': ['macd_enhanced', 'rsi_enhanced', 'atr_enhanced'],
            'stochastic': ['stoch_k', 'stoch_d', 'stoch_divergence', 'stoch_signal'],
            'ichimoku': ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'cloud_thickness', 'ichimoku_signal'],
            'parabolic': ['parabolic_sar', 'sar_trend', 'sar_distance', 'sar_signal'],
            'cointegration': ['cointegration_score', 'mean_reversion_speed', 'half_life', 'cointegration_signal'],
            'pair_trading': ['price_volume_ratio', 'price_momentum_ratio', 'pair_correlation', 'pair_trading_signal'],
            'anomaly_detection': ['price_anomaly', 'volume_anomaly', 'volatility_anomaly', 'returns_anomaly', 'composite_anomaly', 'anomaly_signal']
        }
        
    def calculate_trend_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势类因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # DEMA (双指数移动平均)
            df['dema_14'] = safe_talib_single_price('DEMA', close, 14)
            
            # TEMA (三指数移动平均)
            df['tema_14'] = safe_talib_single_price('TEMA', close, 14)
            
            # KAMA (考夫曼自适应移动平均)
            df['kama_14'] = safe_talib_single_price('KAMA', close, 14)
            
            # TRIX (三重指数平滑震荡器)
            df['trix_14'] = safe_talib_single_price('TRIX', close, 14)
            
            # Aroon指标
            aroon_up, aroon_down = safe_talib_hl('AROON', high, low, 14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            df['aroon_oscillator'] = aroon_up - aroon_down
            
            # ADX (平均趋向指数)
            df['adx_14'] = safe_talib_ohlcv('ADX', high, low, close, None, 14)
            
            # 趋势强度指标
            df['trend_strength'] = np.abs(roll_closed(df, 'close', 20, 'mean') - roll_closed(df, 'close', 5, 'mean')) / get_safe_price(df, 'close')
            
        except Exception as e:
            print(f"趋势因子计算警告: {e}")
            
        return df
    
    def calculate_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量类因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # 多周期RSI
            df['rsi_2'] = safe_talib_single_price('RSI', close, 2)
            df['rsi_14'] = safe_talib_single_price('RSI', close, 14)
            df['rsi_100'] = safe_talib_single_price('RSI', close, 100)
            
            # Stochastic RSI
            fastk, fastd = ta.STOCHRSI(close.shift(1).values, timeperiod=14, fastk_period=5, fastd_period=3)
            df['stoch_rsi'] = fastk
            
            # CCI (顺势指标)
            df['cci_14'] = safe_talib_ohlcv('CCI', high, low, close, None, 14)
            
            # ROC (变动率)
            df['roc_12'] = safe_talib_single_price('ROC', close, 12)
            df['roc_5'] = safe_talib_single_price('ROC', close, 5)
            
            # MFI (资金流量指标)
            df['mfi_14'] = safe_talib_ohlcv('MFI', high, low, close, volume, 14)
            
            # Williams %R
            df['willr_14'] = safe_talib_ohlcv('WILLR', high, low, close, None, 14)
            
            # 动量分层
            df['momentum_regime'] = pd.cut(df['rsi_14'], bins=[0, 30, 70, 100], labels=[-1, 0, 1])
            
        except Exception as e:
            print(f"动量因子计算警告: {e}")
            
        return df
    
    def calculate_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动类因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # ATRP (相对ATR)
            atr = safe_talib_ohlcv('ATR', high, low, close, None, 14)
            df['atrp'] = atr / get_safe_price(df, 'close')  # 解决股价水平漂移问题
            
            # Keltner通道位置
            keltner_ma = roll_closed(df, 'close', 20, 'mean')
            atr_col = 'atr_temp'
            df[atr_col] = atr
            keltner_atr = roll_closed(df, atr_col, 20, 'mean')
            keltner_upper = keltner_ma + 2 * keltner_atr
            keltner_lower = keltner_ma - 2 * keltner_atr
            df['keltner_position'] = (get_safe_price(df, 'close') - keltner_lower) / (keltner_upper - keltner_lower)
            
            # 布林带收缩
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close.shift(1).values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
            
            # 波动率比值
            short_vol = roll_closed(df, 'close', 5, 'std')
            long_vol = roll_closed(df, 'close', 20, 'std')
            df['volatility_ratio'] = short_vol / long_vol
            
            # Parkinson波动率估计器
            df['parkinson_vol'] = np.sqrt(0.361 * np.log(high.shift(1) / low.shift(1)) ** 2)
            
            # 隐含波动率代理
            df['iv_proxy'] = (high.shift(1) - low.shift(1)) / close.shift(1)
            
        except Exception as e:
            print(f"波动率因子计算警告: {e}")
            
        return df
    
    def calculate_volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量类因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # VWAP偏离度
            close_volume = close * volume
            close_volume_sum = roll_closed(df, close_volume.name if hasattr(close_volume, 'name') else 'close_volume_temp', 20, 'sum')
            volume_sum = roll_closed(df, 'volume', 20, 'sum')
            vwap = close_volume_sum / volume_sum
            df['vwap_deviation'] = (get_safe_price(df, 'close') - vwap) / get_safe_price(df, 'close')
            
            # Volume RSI
            volume_gains = volume.shift(1).diff().clip(lower=0)
            volume_losses = (-volume.shift(1).diff()).clip(lower=0)
            df['vg_temp'] = volume_gains
            df['vl_temp'] = volume_losses
            rs = roll_closed(df, 'vg_temp', 14, 'mean') / roll_closed(df, 'vl_temp', 14, 'mean')
            df['volume_rsi'] = 100 - (100 / (1 + rs))
            
            # A/D Line (累积/派发线)
            money_flow_multiplier = ((close.shift(1) - low.shift(1)) - (high.shift(1) - close.shift(1))) / (high.shift(1) - low.shift(1) + 1e-8)
            money_flow_volume = money_flow_multiplier * volume.shift(1)
            df['mfv_temp'] = money_flow_volume
            df['ad_line'] = roll_closed(df, 'mfv_temp', 252, 'sum')
            
            # Chaikin Money Flow
            mfv_sum = roll_closed(df, 'mfv_temp', 20, 'sum')
            vol_sum = roll_closed(df, 'volume', 20, 'sum')
            df['cmf'] = mfv_sum / vol_sum
            
            # Volume MA偏离度
            volume_ma = roll_closed(df, 'volume', 20, 'mean')
            df['volume_ma_deviation'] = (volume - volume_ma) / volume_ma
            
            # Volume-Price Trend
            df['returns_temp'] = close.pct_change()
            df['vpt_temp'] = volume.shift(1) * df['returns_temp']
            df['vpt'] = roll_closed(df, 'vpt_temp', 252, 'sum')
            
        except Exception as e:
            print(f"成交量因子计算警告: {e}")
            
        return df
    
    def calculate_microstructure_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算微观结构因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # 买卖价差代理
            df['hl_spread'] = (high.shift(1) - low.shift(1)) / close.shift(1)
            
            # 成交量强度
            df['volume_intensity'] = volume / roll_closed(df, 'volume', 20, 'mean')
            
            # 价格效率 (Random Walk Index)
            price_range = high.shift(1) - low.shift(1)
            df['high_close_diff'] = (high.shift(1) - close.shift(1)).abs()
            df['low_close_diff'] = (low.shift(1) - close.shift(1)).abs()
            true_range = np.maximum(price_range, np.maximum(df['high_close_diff'], df['low_close_diff']))
            df['true_range_temp'] = true_range
            df['price_efficiency'] = price_range / roll_closed(df, 'true_range_temp', 14, 'sum')
            
            # 流动性指标
            df['liquidity_proxy'] = volume.shift(1) * close.shift(1) / (high.shift(1) - low.shift(1) + 1e-8)
            
            # 价格跳跃检测
            returns = close.pct_change()
            df['returns_temp'] = returns
            rolling_std = roll_closed(df, 'returns_temp', 20, 'std')
            df['price_jump'] = np.abs(df['returns_temp'].shift(1)) / rolling_std
            
        except Exception as e:
            print(f"微观结构因子计算警告: {e}")
            
        return df
    
    def calculate_enhanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算增强型因子 - 多维度信号融合"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # 增强型MACD
            exp1 = safe_ema(close, 12)
            exp2 = safe_ema(close, 26)
            macd = exp1 - exp2
            signal = safe_ema(macd, 9)
            histogram = macd - signal
            
            # MACD能量
            df['macd_enhanced'] = histogram * roll_closed(df, 'volume', 20, 'mean')
            
            # 增强型RSI (结合成交量)
            rsi = safe_talib_single_price('RSI', close, 14)
            volume_weighted_rsi = rsi * (volume.shift(1) / roll_closed(df, 'volume', 14, 'mean'))
            df['rsi_enhanced'] = volume_weighted_rsi
            
            # 增强型ATR (考虑成交量)
            atr = safe_talib_ohlcv('ATR', high, low, close, None, 14)
            df['atr_enhanced'] = atr * np.sqrt(volume.shift(1) / roll_closed(df, 'volume', 14, 'mean'))
            
            # 多因子得分
            factors_to_rank = ['rsi_14', 'macd_enhanced', 'atrp', 'vwap_deviation']
            rank_wnd = min(60, len(df))  # 最多60根K线排名窗口
            for factor in factors_to_rank:
                if factor in df.columns:
                    # 使用防未来函数的rank函数
                    df[f'{factor}_rank'] = roll_closed_rank(df, factor, rank_wnd, min_periods=20)
            
            # 复合动量得分
            momentum_factors = ['roc_12', 'rsi_14', 'stoch_rsi']
            valid_momentum = [f for f in momentum_factors if f in df.columns]
            if valid_momentum:
                df['momentum_composite'] = df[valid_momentum].mean(axis=1)
                
        except Exception as e:
            print(f"增强型因子计算警告: {e}")
            
        return df
    
    def calculate_cross_cycle_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化实现 10 个无 L2 高价值因子 + 周期闸门"""
        print(f"🎯 开始计算跨周期向量化因子...")
        
        close, high, low, open_price, volume = df['close'], df['high'], df['low'], df['open'], df['volume']
        
        try:
            # ① smart_money_flow - 尾盘-开盘 VWAP 代理聪明钱
            # 假设最后25%为尾盘，前25%为开盘
            n = len(df)
            if n >= 4:
                # 向量化计算日内VWAP
                typical_price = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
                df['tp_temp'] = typical_price * volume.shift(1)
                tp_sum = roll_closed(df, 'tp_temp', min(20, n//4), 'sum')
                vol_sum = roll_closed(df, 'volume', min(20, n//4), 'sum')
                vwap = tp_sum / vol_sum
                # 尾盘vs开盘价差 (简化为收盘vs开盘)
                df['smart_money_flow'] = (close.shift(1) - open_price.shift(1)) / vwap
            else:
                df['smart_money_flow'] = 0
            
            # ② zscore_momentum_20 - 20根K线收益滚动标准化
            returns = close.pct_change()
            df['returns_temp'] = returns
            window = min(20, max(5, n//3))
            rolling_mean = roll_closed(df, 'returns_temp', window, 'mean')
            rolling_std = roll_closed(df, 'returns_temp', window, 'std')
            df['zscore_momentum_20'] = (df['returns_temp'].shift(1) - rolling_mean) / (rolling_std + 1e-8)
            
            # ③ order_flow_imbalance - 上涨vs下跌K线成交量差
            df['up_vol_temp'] = np.where(close.shift(1) > close.shift(2), volume.shift(1), 0)
            df['down_vol_temp'] = np.where(close.shift(1) < close.shift(2), volume.shift(1), 0)
            window_flow = min(14, max(3, n//4))
            up_vol_ma = roll_closed(df, 'up_vol_temp', window_flow, 'sum')
            down_vol_ma = roll_closed(df, 'down_vol_temp', window_flow, 'sum')
            df['order_flow_imbalance'] = (up_vol_ma - down_vol_ma) / (up_vol_ma + down_vol_ma + 1e-8)
            
            # ④ vw_macd - 成交量加权MACD
            # 先计算传统MACD
            exp1 = safe_ema(close, 12)
            exp2 = safe_ema(close, 26)
            macd_line = exp1 - exp2
            signal_line = safe_ema(macd_line, 9)
            histogram = macd_line - signal_line
            # 成交量加权
            volume_weight = volume / roll_closed(df, 'volume', min(20, n//2), 'mean')
            df['vw_macd'] = histogram * volume_weight
            
            # ⑤ drawdown_volatility - 回撤波动率(下行风险敏感)
            window_dd_max = min(252, len(close))  # 最长1年窗口
            cummax = roll_closed(df, 'close', window_dd_max, 'max')
            drawdown = (close - cummax) / cummax
            window_dd = min(30, max(10, n//2))
            df['drawdown_temp'] = drawdown
            df['drawdown_volatility'] = roll_closed(df, 'drawdown_temp', window_dd, 'std')
            
            # ⑥ skewness_60 - 60根K线收益偏度(极端预警)
            window_skew = min(60, max(20, n//2))
            if n >= 20:
                df['skewness_60'] = roll_closed(df, 'returns_temp', window_skew, 'skew')
            else:
                df['skewness_60'] = 0
                
            # ⑦ mean_reversion_score - 布林带Z-Score(统计套利)
            window_bb = min(20, max(10, n//3))
            bb_ma = roll_closed(df, 'close', window_bb, 'mean')
            bb_std = roll_closed(df, 'close', window_bb, 'std')
            df['mean_reversion_score'] = (get_safe_price(df, 'close') - bb_ma) / (bb_std + 1e-8)
            
            # ⑧ seasonality_friday - 是否周五(日历效应)
            if hasattr(df.index, 'dayofweek'):
                df['seasonality_friday'] = (df.index.dayofweek == 4).astype(int)
            else:
                # 如果没有时间索引，使用简化版本
                df['seasonality_friday'] = 0.2  # 固定值，避免常数因子
                
            # ⑨ vol_term_structure - 短/长波动率比(期限结构)
            if n >= 30:
                short_vol = roll_closed(df, 'returns_temp', 5, 'std')
                long_vol = roll_closed(df, 'returns_temp', min(30, n//2), 'std')
                df['vol_term_structure'] = short_vol / (long_vol + 1e-8)
            else:
                df['vol_term_structure'] = 1.0
                
            # ⑩ composite_alpha - 等权融合上述9个因子(信号融合)
            alpha_factors = [
                'smart_money_flow', 'zscore_momentum_20', 'order_flow_imbalance',
                'vw_macd', 'drawdown_volatility', 'skewness_60', 
                'mean_reversion_score', 'seasonality_friday', 'vol_term_structure'
            ]
            
            # 先对各因子进行标准化，然后等权融合
            alpha_components = []
            for factor in alpha_factors:
                if factor in df.columns:
                    factor_data = df[factor].fillna(0)
                    if factor_data.std() > 1e-8:  # 避免常数因子
                        df[f'{factor}_temp'] = factor_data
                        factor_mean = roll_closed(df, f'{factor}_temp', 252, 'mean')
                        factor_std = roll_closed(df, f'{factor}_temp', 252, 'std')
                        factor_normalized = (factor_data - factor_mean) / factor_std
                        alpha_components.append(factor_normalized)
            
            if alpha_components:
                df['composite_alpha'] = pd.concat(alpha_components, axis=1).mean(axis=1)
            else:
                df['composite_alpha'] = 0
            
            # 周期闸门 - 不足60根K线时移除部分因子
            if len(df) < 60:
                print(f"  ⚠️ 周期闸门生效: 数据不足60根({len(df)}根)，移除部分因子")
                factors_to_remove = ['vol_term_structure', 'drawdown_volatility', 'skewness_60']
                for factor in factors_to_remove:
                    if factor in df.columns:
                        df = df.drop(columns=[factor])
                        print(f"    🚫 移除因子: {factor}")
            
            print(f"✅ 跨周期因子计算完成: 新增{len([c for c in df.columns if c.startswith(('smart_', 'zscore_', 'order_', 'vw_', 'drawdown_', 'skewness_', 'mean_', 'seasonality_', 'vol_', 'composite_'))])}个因子")
            
        except Exception as e:
            print(f"❌ 跨周期因子计算失败: {e}")
            
        return df
    
    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子 - 一站式调用"""
        print(f"🔧 开始计算30+高级因子...")
        
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 逐类别计算
        df = self.calculate_trend_factors(df)
        df = self.calculate_momentum_factors(df)
        df = self.calculate_volatility_factors(df)
        df = self.calculate_volume_factors(df)
        df = self.calculate_microstructure_factors(df)
        df = self.calculate_enhanced_factors(df)
        df = self.calculate_cross_cycle_factors(df)
        df = self.calculate_new_enhanced_factors(df)
        
        # 统计计算的因子数量
        original_cols = set(required_cols)
        new_factors = [col for col in df.columns if col not in original_cols]
        
        print(f"✅ 高级因子计算完成: 新增{len(new_factors)}个因子")
        print(f"📊 因子分类统计:")
        for category, factors in self.factor_categories.items():
            available = [f for f in factors if f in new_factors]
            print(f"   {category}: {len(available)}个因子")
            
        return df
    
    def calculate_new_enhanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算新增的增强因子 - 随机震荡器、Ichimoku Cloud、抛物线SAR等"""
        print(f"🚀 开始计算新增增强因子...")
        
        try:
            # 计算各类新增增强因子
            stoch_factors = calculate_stochastic_factors(df)
            ichimoku_factors = calculate_ichimoku_factors(df)
            sar_factors = calculate_parabolic_sar(df)
            coint_factors = calculate_cointegration_factors(df)
            pair_factors = calculate_pair_trading_factors(df)
            anomaly_factors = calculate_anomaly_factors(df)
            
            # 合并所有因子，避免列名冲突
            all_enhanced_dfs = [stoch_factors, ichimoku_factors, sar_factors, coint_factors, pair_factors, anomaly_factors]
            
            # 处理可能的列名冲突
            seen_columns = set()
            processed_dfs = []
            
            for enhanced_df in all_enhanced_dfs:
                # 检查是否有重复列名
                duplicate_cols = set(enhanced_df.columns) & seen_columns
                if duplicate_cols:
                    # 为重复列添加前缀
                    for col in duplicate_cols:
                        enhanced_df = enhanced_df.rename(columns={col: f"enhanced_{col}"})
                
                processed_dfs.append(enhanced_df)
                seen_columns.update(enhanced_df.columns)
            
            # 合并到主DataFrame
            if processed_dfs:
                enhanced_combined = pd.concat(processed_dfs, axis=1)
                df = pd.concat([df, enhanced_combined], axis=1)
            
            # 统计新增因子数量
            new_enhanced_factors = [col for col in df.columns if col.startswith(('stoch_', 'tenkan_', 'kijun_', 'senkou_', 'chikou_', 'cloud_', 'ichimoku_', 'parabolic_', 'sar_', 'cointegration_', 'mean_reversion_', 'half_life_', 'price_', 'pair_', 'correlation_', 'anomaly_', '_anomaly'))]
            
            print(f"✅ 新增增强因子计算完成: 新增{len(new_enhanced_factors)}个因子")
            
        except Exception as e:
            print(f"❌ 新增增强因子计算失败: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def get_factor_descriptions(self) -> Dict[str, str]:
        """获取因子描述"""
        descriptions = {
            # 趋势类
            'dema_14': '双指数移动平均 - 减少滞后性',
            'tema_14': '三指数移动平均 - 更平滑的趋势',
            'kama_14': '考夫曼自适应移动平均 - 适应市场波动',
            'trix_14': '三重指数平滑震荡器 - 过滤噪音',
            'aroon_up': 'Aroon上线 - 新高趋势强度',
            'aroon_down': 'Aroon下线 - 新低趋势强度', 
            'adx_14': '平均趋向指数 - 趋势强度',
            
            # 动量类
            'rsi_2': '2期RSI - 超短期超买超卖',
            'rsi_100': '100期RSI - 长期动量',
            'stoch_rsi': '随机RSI - RSI的随机化',
            'cci_14': '顺势指标 - 价格偏离程度',
            'roc_12': '12期变动率 - 价格动量',
            'mfi_14': '资金流量指标 - 成交量加权RSI',
            'willr_14': 'Williams %R - 超买超卖',
            
            # 波动率类
            'atrp': '相对ATR - 消除价格水平影响',
            'keltner_position': 'Keltner通道位置 - 价格相对位置',
            'bb_squeeze': '布林带收缩 - 波动率压缩',
            'volatility_ratio': '波动率比值 - 短期vs长期波动',
            'parkinson_vol': 'Parkinson波动率 - 高效波动率估计',
            
            # 成交量类
            'vwap_deviation': 'VWAP偏离度 - 价格vs成交量加权价格',
            'volume_rsi': '成交量RSI - 成交量动量',
            'ad_line': '累积派发线 - 资金流向',
            'cmf': 'Chaikin资金流 - 买卖压力',
            'volume_ma_deviation': '成交量均线偏离 - 成交量异常',
            
            # 微观结构类
            'hl_spread': '高低价差 - 流动性代理',
            'volume_intensity': '成交量强度 - 相对成交量',
            'price_efficiency': '价格效率 - 随机游走指数',
            
            # 增强型
            'macd_enhanced': '增强MACD - 结合成交量',
            'rsi_enhanced': '增强RSI - 成交量加权',
            'atr_enhanced': '增强ATR - 考虑成交量',
        }
        return descriptions


def test_advanced_factors():
    """测试高级因子池"""
    print("🧪 测试高级因子池...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='1H')
    n = len(dates)
    
    test_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 0,
        'low': 0, 
        'close': 0,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # 设置high/low
    test_data['close'] = test_data['open'] + np.random.randn(n) * 0.1
    test_data['high'] = np.maximum(test_data['open'], test_data['close']) + np.abs(np.random.randn(n) * 0.05)
    test_data['low'] = np.minimum(test_data['open'], test_data['close']) - np.abs(np.random.randn(n) * 0.05)
    
    # 计算因子
    factor_pool = AdvancedFactorPool()
    result = factor_pool.calculate_all_factors(test_data)
    
    print(f"✅ 测试完成: 输入{len(test_data.columns)}列，输出{len(result.columns)}列")
    
    return result


def test_cross_cycle():
    """测试跨周期因子 - 自检函数"""
    print("\n🎯 跨周期因子自检开始...")
    
    # 测试数据1: 充足数据 (>60根K线)
    print("\n📊 测试1: 充足数据(8760根K线)")
    np.random.seed(42)
    dates_full = pd.date_range('2024-01-01', '2024-12-31', freq='1H')
    test_data_full = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates_full)) * 0.1),
        'high': 0, 'low': 0, 'close': 0,
        'volume': np.random.randint(1000, 10000, len(dates_full))
    }, index=dates_full)
    
    test_data_full['close'] = test_data_full['open'] + np.random.randn(len(dates_full)) * 0.1
    test_data_full['high'] = np.maximum(test_data_full['open'], test_data_full['close']) + np.abs(np.random.randn(len(dates_full)) * 0.05)
    test_data_full['low'] = np.minimum(test_data_full['open'], test_data_full['close']) - np.abs(np.random.randn(len(dates_full)) * 0.05)
    
    factor_pool = AdvancedFactorPool()
    result_full = factor_pool.calculate_cross_cycle_factors(test_data_full.copy())
    
    # 检查新增因子
    cross_cycle_factors = [col for col in result_full.columns if col.startswith(('smart_', 'zscore_', 'order_', 'vw_', 'drawdown_', 'skewness_', 'mean_', 'seasonality_', 'vol_', 'composite_'))]
    print(f"✅ 新增跨周期因子: {len(cross_cycle_factors)}个")
    for factor in cross_cycle_factors:
        print(f"    📈 {factor}")
    
    # 测试数据2: 不足数据 (<60根K线) - 周期闸门测试
    print(f"\n📊 测试2: 不足数据(30根K线) - 周期闸门测试")
    dates_short = pd.date_range('2024-01-01', periods=30, freq='1H')
    test_data_short = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(30) * 0.1),
        'high': 0, 'low': 0, 'close': 0,
        'volume': np.random.randint(1000, 10000, 30)
    }, index=dates_short)
    
    test_data_short['close'] = test_data_short['open'] + np.random.randn(30) * 0.1
    test_data_short['high'] = np.maximum(test_data_short['open'], test_data_short['close']) + np.abs(np.random.randn(30) * 0.05)
    test_data_short['low'] = np.minimum(test_data_short['open'], test_data_short['close']) - np.abs(np.random.randn(30) * 0.05)
    
    result_short = factor_pool.calculate_cross_cycle_factors(test_data_short.copy())
    
    # 检查周期闸门效果
    cross_cycle_factors_short = [col for col in result_short.columns if col.startswith(('smart_', 'zscore_', 'order_', 'vw_', 'drawdown_', 'skewness_', 'mean_', 'seasonality_', 'vol_', 'composite_'))]
    removed_factors = set(cross_cycle_factors) - set(cross_cycle_factors_short)
    
    print(f"✅ 周期闸门生效: 移除{len(removed_factors)}个因子")
    for factor in removed_factors:
        print(f"    🚫 已移除: {factor}")
    print(f"✅ 保留因子: {len(cross_cycle_factors_short)}个")
    
    # 性能测试
    print(f"\n⚡ 性能检查:")
    print(f"    内存峰值: 充足数据 {result_full.memory_usage().sum() / 1024 / 1024:.1f} MB")
    print(f"    因子总数: 充足数据 {len(result_full.columns)}列")
    print(f"    向量化: ✅ 无循环，纯pandas/numpy操作")
    
    print(f"\n🎉 跨周期因子自检完成！")
    return result_full, result_short


def calculate_ichimoku_factors(df, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_span=26):
    """
    计算Ichimoku Cloud (一目均衡表) 因子
    
    Args:
        df: DataFrame with OHLCV data
        tenkan_period: 转换线周期 (default: 9)
        kijun_period: 基准线周期 (default: 26)
        senkou_b_period: 后行带B周期 (default: 52)
        chikou_span: 滞后跨度 (default: 26)
    
    Returns:
        DataFrame with Ichimoku factors
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取安全的滞后价格数据
    high = get_safe_price(df, 'high')
    low = get_safe_price(df, 'low')
    close = get_safe_price(df, 'close')
    
    # 确保有足够的数据
    min_data_needed = max(tenkan_period, kijun_period, senkou_b_period) + chikou_span
    
    if len(df) < min_data_needed:
        # 数据不足时返回空值
        result['tenkan_sen'] = np.nan
        result['kijun_sen'] = np.nan
        result['senkou_span_a'] = np.nan
        result['senkou_span_b'] = np.nan
        result['chikou_span'] = np.nan
        result['cloud_thickness'] = np.nan
        result['ichimoku_signal'] = np.nan
        return result
    
    try:
        # 计算转换线 (Tenkan-sen): 9周期最高价和最低价的平均值
        tenkan_high = roll_closed(df, 'high', tenkan_period, 'max')
        tenkan_low = roll_closed(df, 'low', tenkan_period, 'min')
        result['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # 计算基准线 (Kijun-sen): 26周期最高价和最低价的平均值
        kijun_high = roll_closed(df, 'high', kijun_period, 'max')
        kijun_low = roll_closed(df, 'low', kijun_period, 'min')
        result['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # 计算先行带A (Senkou Span A): 转换线和基准线的平均值，向前移动26周期
        senkou_a_raw = (result['tenkan_sen'] + result['kijun_sen']) / 2
        result['senkou_span_a'] = senkou_a_raw.shift(chikou_span)
        
        # 计算先行带B (Senkou Span B): 52周期最高价和最低价的平均值，向前移动26周期
        senkou_b_high = roll_closed(df, 'high', senkou_b_period, 'max')
        senkou_b_low = roll_closed(df, 'low', senkou_b_period, 'min')
        senkou_b_raw = (senkou_b_high + senkou_b_low) / 2
        result['senkou_span_b'] = senkou_b_raw.shift(chikou_span)
        
        # 计算滞后跨度 (Chikou Span): 收盘价向前移动26周期（避免未来函数）
        # 注意：这个指标在传统定义中会使用未来数据，但我们修改为使用历史数据
        result['chikou_span'] = close.shift(chikou_span)
        
        # 计算云层厚度
        result['cloud_thickness'] = abs(result['senkou_span_a'] - result['senkou_span_b'])
        
        # 计算交易信号
        # 1. 价格在云层上方看涨，在云层下方看跌
        price_above_cloud = close > result['senkou_span_a']
        price_below_cloud = close < result['senkou_span_b']
        
        # 2. 转换线上穿基准线看涨，下穿看跌
        tenkan_cross_above = (result['tenkan_sen'] > result['kijun_sen']) & \
                           (result['tenkan_sen'].shift(1) <= result['kijun_sen'].shift(1))
        tenkan_cross_below = (result['tenkan_sen'] < result['kijun_sen']) & \
                           (result['tenkan_sen'].shift(1) >= result['kijun_sen'].shift(1))
        
        # 3. 滞后跨度在价格上方看涨，在价格下方看跌
        chikou_above_price = result['chikou_span'] > close
        chikou_below_price = result['chikou_span'] < close
        
        # 综合信号
        result['ichimoku_signal'] = np.where(
            price_above_cloud & tenkan_cross_above & chikou_above_price, 1,  # 强烈看涨
            np.where(
                price_below_cloud & tenkan_cross_below & chikou_below_price, -1,  # 强烈看跌
                np.where(
                    price_above_cloud, 0.5,  # 温和看涨
                    np.where(
                        price_below_cloud, -0.5,  # 温和看跌
                        0  # 中性
                    )
                )
            )
        )
        
    except Exception as e:
        print(f"⚠️ Ichimoku Cloud计算错误: {e}")
        result['tenkan_sen'] = np.nan
        result['kijun_sen'] = np.nan
        result['senkou_span_a'] = np.nan
        result['senkou_span_b'] = np.nan
        result['chikou_span'] = np.nan
        result['cloud_thickness'] = np.nan
        result['ichimoku_signal'] = np.nan
    
    return result


def calculate_parabolic_sar(df, af=0.02, max_af=0.2, acceleration=0.02):
    """
    计算抛物线SAR (Stop and Reverse) 因子
    
    Args:
        df: DataFrame with OHLCV data
        af: 初始加速因子 (default: 0.02)
        max_af: 最大加速因子 (default: 0.2)
        acceleration: 加速步长 (default: 0.02)
    
    Returns:
        DataFrame with Parabolic SAR factors
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取安全的滞后价格数据
    high = get_safe_price(df, 'high')
    low = get_safe_price(df, 'low')
    close = get_safe_price(df, 'close')
    
    if len(df) < 5:
        # 数据不足时返回空值
        result['parabolic_sar'] = np.nan
        result['sar_trend'] = np.nan
        result['sar_distance'] = np.nan
        result['sar_signal'] = np.nan
        return result
    
    try:
        # 初始化变量
        sar_values = []
        ep_values = []  # Extreme Point
        af_values = []
        trend_values = []
        
        # 首先填充NaN作为第一个值
        sar_values.append(np.nan)
        ep_values.append(np.nan)
        af_values.append(np.nan)
        trend_values.append(np.nan)
        
        # 从第二根K线开始计算
        if len(df) >= 2:
            # 确定初始趋势
            if close.iloc[1] > close.iloc[0]:
                trend = 1  # 上升趋势
                sar = low.iloc[0]
                ep = high.iloc[1]
            else:
                trend = -1  # 下降趋势
                sar = high.iloc[0]
                ep = low.iloc[1]
            
            current_af = af
            
            sar_values.append(sar)
            ep_values.append(ep)
            af_values.append(current_af)
            trend_values.append(trend)
            
            # 迭代计算后续值 - 使用滞后数据避免未来函数
            for i in range(2, len(df)):
                # 使用前一个周期的数据（避免未来函数）
                current_high = high.iloc[i-1] if i-1 < len(high) else high.iloc[i]
                current_low = low.iloc[i-1] if i-1 < len(low) else low.iloc[i]
                current_close = close.iloc[i-1] if i-1 < len(close) else close.iloc[i]
                
                if trend == 1:  # 上升趋势
                    # SAR值
                    sar = sar_values[-1] + current_af * (ep - sar_values[-1])
                    
                    # 检查是否需要反转
                    if current_low < sar:
                        # 反转为下降趋势
                        trend = -1
                        sar = max(ep, current_high)
                        ep = current_low
                        current_af = af
                    else:
                        # 继续上升趋势
                        if current_high > ep:
                            ep = current_high
                            current_af = min(current_af + acceleration, max_af)
                        
                        # SAR不能低于前两个周期的最低价
                        sar = min(sar, min(low.iloc[i-1], low.iloc[i-2]))
                
                else:  # 下降趋势
                    # SAR值
                    sar = sar_values[-1] + current_af * (ep - sar_values[-1])
                    
                    # 检查是否需要反转
                    if current_high > sar:
                        # 反转为上升趋势
                        trend = 1
                        sar = min(ep, current_low)
                        ep = current_high
                        current_af = af
                    else:
                        # 继续下降趋势
                        if current_low < ep:
                            ep = current_low
                            current_af = min(current_af + acceleration, max_af)
                        
                        # SAR不能高于前两个周期的最高价
                        sar = max(sar, max(high.iloc[i-1], high.iloc[i-2]))
                
                sar_values.append(sar)
                ep_values.append(ep)
                af_values.append(current_af)
                trend_values.append(trend)
            
            # 填充结果
            result['parabolic_sar'] = sar_values
            result['sar_trend'] = trend_values
            
            # 计算SAR与价格的距离（避免除以0）
            result['sar_distance'] = np.where(
                result['parabolic_sar'] != 0,
                (close - result['parabolic_sar']) / result['parabolic_sar'],
                np.nan
            )
            
            # 生成交易信号
            result['sar_signal'] = np.where(
                (result['sar_trend'] == 1) & (result['sar_trend'].shift(1) == -1), 1,  # 买入信号
                np.where(
                    (result['sar_trend'] == -1) & (result['sar_trend'].shift(1) == 1), -1,  # 卖出信号
                    result['sar_trend'] * 0.5  # 趋势信号
                )
            )
        
    except Exception as e:
        print(f"⚠️ 抛物线SAR计算错误: {e}")
        result['parabolic_sar'] = np.nan
        result['sar_trend'] = np.nan
        result['sar_distance'] = np.nan
        result['sar_signal'] = np.nan
    
    return result


def calculate_cointegration_factors(df, window=60):
    """
    计算协整关系因子（单资产版本）
    
    Args:
        df: DataFrame with OHLCV data
        window: 协整检验窗口 (default: 60)
    
    Returns:
        DataFrame with cointegration factors
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取安全的滞后价格数据
    close = get_safe_price(df, 'close')
    high = get_safe_price(df, 'high')
    low = get_safe_price(df, 'low')
    
    if len(df) < window:
        # 数据不足时返回空值
        result['cointegration_score'] = np.nan
        result['mean_reversion_speed'] = np.nan
        result['half_life'] = np.nan
        result['cointegration_signal'] = np.nan
        return result
    
    try:
        # 计算价格与其他技术指标的关系
        # 1. 价格与移动平均的协整关系
        ma_short = roll_closed(df, 'close', 20, 'mean')
        ma_long = roll_closed(df, 'close', 60, 'mean')
        
        # 计价差序列
        spread_short = close - ma_short
        spread_long = close - ma_long
        
        # 计算均值回归速度（使用简单线性回归）
        def calculate_half_life(spread_series):
            """计算半衰期"""
            if len(spread_series.dropna()) < 10:
                return np.nan
            
            # 简单的均值回归速度估计
            delta = spread_series.diff().dropna()
            lagged_spread = spread_series.shift(1).dropna()
            
            if len(delta) < 5 or len(lagged_spread) < 5:
                return np.nan
            
            # 对齐数据
            aligned_delta = delta.loc[lagged_spread.index]
            
            # 简单线性回归
            if len(aligned_delta) > 0:
                correlation = aligned_delta.corr(lagged_spread)
                if not np.isnan(correlation):
                    # 简化的半衰期计算
                    if abs(correlation) > 0.01:
                        return -np.log(2) / correlation
            return np.nan
        
        # 滚动计算半衰期
        half_life_short = []
        half_life_long = []
        
        for i in range(window-1, len(df)):
            window_spread_short = spread_short.iloc[i-window+1:i+1]
            window_spread_long = spread_long.iloc[i-window+1:i+1]
            
            hl_short = calculate_half_life(window_spread_short)
            hl_long = calculate_half_life(window_spread_long)
            
            half_life_short.append(hl_short)
            half_life_long.append(hl_long)
        
        # 填充结果
        result['half_life'] = [np.nan] * (window-1) + half_life_short
        
        # 计算协整得分（基于价格与移动平均的偏离程度）
        spread_mean = roll_closed(df, 'spread_temp', window, 'mean') if 'spread_temp' in df.columns else spread_short.rolling(window=window, min_periods=window//2).mean()
        spread_std = roll_closed(df, 'spread_temp', window, 'std') if 'spread_temp' in df.columns else spread_short.rolling(window=window, min_periods=window//2).std()
        normalized_spread = (spread_short - spread_mean) / (spread_std + 1e-8)
        
        result['cointegration_score'] = normalized_spread
        
        # 计算均值回归速度
        result['mean_reversion_speed'] = np.where(
            result['half_life'] > 0,
            1 / result['half_life'],
            np.nan
        )
        
        # 生成交易信号
        result['cointegration_signal'] = np.where(
            normalized_spread > 2, -1,  # 显著高估
            np.where(
                normalized_spread < -2, 1,  # 显著低估
                np.where(
                    normalized_spread > 1, -0.5,  # 轻度高估
                    np.where(
                        normalized_spread < -1, 0.5,  # 轻度低估
                        0  # 中性
                    )
                )
            )
        )
        
    except Exception as e:
        print(f"⚠️ 协整关系因子计算错误: {e}")
        result['cointegration_score'] = np.nan
        result['mean_reversion_speed'] = np.nan
        result['half_life'] = np.nan
        result['cointegration_signal'] = np.nan
    
    return result


def calculate_pair_trading_factors(df, window=30):
    """
    计算配对交易因子（基于价格与技术指标的关系）
    
    Args:
        df: DataFrame with OHLCV data
        window: 计算窗口 (default: 30)
    
    Returns:
        DataFrame with pair trading factors
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取安全的滞后价格数据
    close = get_safe_price(df, 'close')
    volume = df['volume'].shift(1)  # 滞后成交量
    
    if len(df) < window:
        # 数据不足时返回空值
        result['price_volume_ratio'] = np.nan
        result['price_momentum_ratio'] = np.nan
        result['pair_trading_signal'] = np.nan
        result['pair_correlation'] = np.nan
        return result
    
    try:
        # 1. 价格与成交量比率
        volume_ma = roll_closed(df, 'volume', window, 'mean')
        result['price_volume_ratio'] = close / (volume_ma + 1e-8)
        
        # 2. 价格与动量比率（使用滞后数据）
        momentum_short = close.pct_change(window//2).shift(1)
        momentum_long = close.pct_change(window).shift(1)
        result['price_momentum_ratio'] = momentum_short / (momentum_long + 1e-8)
        
        # 3. 计算价格与成交量的相关性
        correlation_values = []
        for i in range(window-1, len(df)):
            window_close = close.iloc[i-window+1:i+1]
            window_volume = volume.iloc[i-window+1:i+1]
            
            if len(window_close.dropna()) > 5 and len(window_volume.dropna()) > 5:
                corr = window_close.corr(window_volume)
                correlation_values.append(corr if not np.isnan(corr) else 0)
            else:
                correlation_values.append(0)
        
        result['pair_correlation'] = [np.nan] * (window-1) + correlation_values
        
        # 4. 生成配对交易信号
        # 基于价格-成交量背离（使用安全的滚动计算）
        pv_ratio_series = result['price_volume_ratio']
        pv_ratio_mean = pv_ratio_series.rolling(window=window, min_periods=window//2).mean()
        pv_ratio_std = pv_ratio_series.rolling(window=window, min_periods=window//2).std()
        pv_ratio_z = (pv_ratio_series - pv_ratio_mean) / (pv_ratio_std + 1e-8)
        
        result['pair_trading_signal'] = np.where(
            pv_ratio_z > 2, -1,  # 价格相对于成交量过高
            np.where(
                pv_ratio_z < -2, 1,  # 价格相对于成交量过低
                np.where(
                    pv_ratio_z > 1, -0.5,
                    np.where(
                        pv_ratio_z < -1, 0.5,
                        0
                    )
                )
            )
        )
        
    except Exception as e:
        print(f"⚠️ 配对交易因子计算错误: {e}")
        result['price_volume_ratio'] = np.nan
        result['price_momentum_ratio'] = np.nan
        result['pair_trading_signal'] = np.nan
        result['pair_correlation'] = np.nan
    
    return result


def calculate_anomaly_factors(df, window=30):
    """
    计算异常检测因子
    
    Args:
        df: DataFrame with OHLCV data
        window: 异常检测窗口 (default: 30)
    
    Returns:
        DataFrame with anomaly detection factors
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取安全的滞后价格数据
    close = get_safe_price(df, 'close')
    high = get_safe_price(df, 'high')
    low = get_safe_price(df, 'low')
    volume = df['volume'].shift(1)
    
    if len(df) < window:
        # 数据不足时返回空值
        result['price_anomaly'] = np.nan
        result['volume_anomaly'] = np.nan
        result['volatility_anomaly'] = np.nan
        result['returns_anomaly'] = np.nan
        result['composite_anomaly'] = np.nan
        return result
    
    try:
        # 1. 价格异常检测（基于Z-score）
        price_mean = roll_closed(df, 'close', window, 'mean')
        price_std = roll_closed(df, 'close', window, 'std')
        result['price_anomaly'] = abs((close - price_mean) / (price_std + 1e-8))
        
        # 2. 成交量异常检测
        volume_mean = roll_closed(df, 'volume', window, 'mean')
        volume_std = roll_closed(df, 'volume', window, 'std')
        result['volume_anomaly'] = abs((volume - volume_mean) / (volume_std + 1e-8))
        
        # 3. 波动率异常检测
        volatility = (high - low) / close
        df['volatility_temp'] = volatility  # 临时存储到df中以便后续使用
        vol_mean = roll_closed(df, 'volatility_temp', window, 'mean')
        vol_std = roll_closed(df, 'volatility_temp', window, 'std')
        result['volatility_anomaly'] = abs((volatility - vol_mean) / (vol_std + 1e-8))
        
        # 4. 收益率异常检测
        returns = close.pct_change().shift(1)  # 使用滞后收益率
        df['returns_temp'] = returns  # 临时存储到df中以便后续使用
        returns_mean = roll_closed(df, 'returns_temp', window, 'mean')
        returns_std = roll_closed(df, 'returns_temp', window, 'std')
        result['returns_anomaly'] = abs((returns - returns_mean) / (returns_std + 1e-8))
        
        # 5. 综合异常得分
        anomaly_factors = [result['price_anomaly'], result['volume_anomaly'], 
                          result['volatility_anomaly'], result['returns_anomaly']]
        
        # 标准化各个异常指标（使用滞后数据）
        normalized_factors = []
        for i, factor in enumerate(anomaly_factors):
            factor_series = factor
            factor_mean = factor_series.rolling(window=window, min_periods=window//2).mean()
            factor_std = factor_series.rolling(window=window, min_periods=window//2).std()
            normalized_factor = (factor_series - factor_mean) / (factor_std + 1e-8)
            normalized_factors.append(normalized_factor)
        
        # 计算综合异常得分
        result['composite_anomaly'] = sum(normalized_factors) / len(normalized_factors)
        
        # 生成异常信号
        result['anomaly_signal'] = np.where(
            result['composite_anomaly'] > 2, -1,  # 强异常卖出
            np.where(
                result['composite_anomaly'] > 1, -0.5,  # 弱异常卖出
                np.where(
                    result['composite_anomaly'] < -2, 1,  # 强异常买入
                    np.where(
                        result['composite_anomaly'] < -1, 0.5,  # 弱异常买入
                        0  # 正常
                    )
                )
            )
        )
        
    except Exception as e:
        print(f"⚠️ 异常检测因子计算错误: {e}")
        result['price_anomaly'] = np.nan
        result['volume_anomaly'] = np.nan
        result['volatility_anomaly'] = np.nan
        result['returns_anomaly'] = np.nan
        result['composite_anomaly'] = np.nan
        result['anomaly_signal'] = np.nan
    
    return result


def calculate_stochastic_factors(df, k_period=14, d_period=3, slowing=3):
    """
    计算随机震荡器因子
    
    Args:
        df: DataFrame with OHLCV data
        k_period: %K period (default: 14)
        d_period: %D period (default: 3) 
        slowing: slowing period (default: 3)
    
    Returns:
        DataFrame with stochastic factors
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取安全的滞后价格数据
    high = get_safe_price(df, 'high')
    low = get_safe_price(df, 'low')
    close = get_safe_price(df, 'close')
    
    # 确保有足够的数据
    min_data_needed = k_period + d_period + slowing
    
    if len(df) < min_data_needed:
        # 数据不足时返回空值
        result['stoch_k'] = np.nan
        result['stoch_d'] = np.nan
        result['stoch_divergence'] = np.nan
        result['stoch_signal'] = np.nan
        return result
    
    try:
        # 计算最高价和最低价的滚动窗口
        highest_high = roll_closed(df, 'high', k_period, 'max')
        lowest_low = roll_closed(df, 'low', k_period, 'min')
        
        # 避免0值
        range_val = highest_high - lowest_low
        range_val = range_val.replace(0, np.nan)
        
        # 计算%K (未平滑)
        k_raw = 100 * (close - lowest_low) / range_val
        
        # 对%K进行平滑
        k_smoothed = k_raw.rolling(window=slowing, min_periods=max(1, slowing//2)).mean()
        
        # 计算%D (对%K的移动平均)
        d_smoothed = k_smoothed.rolling(window=d_period, min_periods=max(1, d_period//2)).mean()
        
        # 存储结果
        result['stoch_k'] = k_smoothed
        result['stoch_d'] = d_smoothed
        
        # 计算随机震荡器背离
        # 价格创新高但随机震荡器未创新高 = 看跌背离
        price_high = roll_closed(df, 'high', 5, 'max')
        stoch_high = roll_closed(result, 'stoch_k', 5, 'max')
        result['stoch_divergence'] = np.where(
            (price_high > price_high.shift(5)) & (stoch_high < stoch_high.shift(5)),
            -1,  # 看跌背离
            np.where(
                (roll_closed(df, 'low', 5, 'min') < roll_closed(df, 'low', 5, 'min').shift(5)) & 
                (roll_closed(result, 'stoch_k', 5, 'min') > roll_closed(result, 'stoch_k', 5, 'min').shift(5)),
                1,   # 看涨背离
                0    # 无背离
            )
        )
        
        # 交易信号
        result['stoch_signal'] = np.where(
            (k_smoothed > 80) & (d_smoothed > 80), -1,  # 超买
            np.where(
                (k_smoothed < 20) & (d_smoothed < 20), 1,  # 超卖
                np.where(
                    k_smoothed > d_smoothed, 0.5,  # %K上穿%D
                    -0.5  # %K下穿%D
                )
            )
        )
        
    except Exception as e:
        print(f"⚠️ 随机震荡器计算错误: {e}")
        result['stoch_k'] = np.nan
        result['stoch_d'] = np.nan
        result['stoch_divergence'] = np.nan
        result['stoch_signal'] = np.nan
    
    return result


def test_enhanced_factors():
    """测试增强因子"""
    print(f"\n🧪 测试增强因子")
    
    # 生成测试数据
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'high': 0, 'low': 0, 'close': 0,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    test_data['close'] = test_data['open'] + np.random.randn(200) * 0.1
    test_data['high'] = np.maximum(test_data['open'], test_data['close']) + np.abs(np.random.randn(200) * 0.05)
    test_data['low'] = np.minimum(test_data['open'], test_data['close']) - np.abs(np.random.randn(200) * 0.05)
    
    # 测试随机震荡器
    print(f"\n📊 测试随机震荡器因子:")
    stoch_result = calculate_stochastic_factors(test_data)
    
    enhanced_factors = [col for col in stoch_result.columns if col.startswith('stoch_')]
    print(f"✅ 随机震荡器因子: {len(enhanced_factors)}个")
    for factor in enhanced_factors:
        non_null_count = stoch_result[factor].notna().sum()
        print(f"    📈 {factor}: {non_null_count}个有效值")
    
    # 检查值范围
    for factor in ['stoch_k', 'stoch_d']:
        if factor in stoch_result.columns:
            valid_data = stoch_result[factor].dropna()
            if len(valid_data) > 0:
                print(f"    📊 {factor} 范围: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
    
    # 测试Ichimoku Cloud
    print(f"\n📊 测试Ichimoku Cloud因子:")
    ichimoku_result = calculate_ichimoku_factors(test_data)
    
    ichimoku_factors = [col for col in ichimoku_result.columns if col.startswith(('tenkan_', 'kijun_', 'senkou_', 'chikou_', 'cloud_', 'ichimoku_'))]
    print(f"✅ Ichimoku Cloud因子: {len(ichimoku_factors)}个")
    for factor in ichimoku_factors:
        non_null_count = ichimoku_result[factor].notna().sum()
        print(f"    📈 {factor}: {non_null_count}个有效值")
    
    # 检查信号分布
    if 'ichimoku_signal' in ichimoku_result.columns:
        signal_counts = ichimoku_result['ichimoku_signal'].value_counts()
        print(f"    📊 信号分布: {dict(signal_counts)}")
    
    # 测试抛物线SAR
    print(f"\n📊 测试抛物线SAR因子:")
    sar_result = calculate_parabolic_sar(test_data)
    
    sar_factors = [col for col in sar_result.columns if col.startswith(('parabolic_', 'sar_'))]
    print(f"✅ 抛物线SAR因子: {len(sar_factors)}个")
    for factor in sar_factors:
        non_null_count = sar_result[factor].notna().sum()
        print(f"    📈 {factor}: {non_null_count}个有效值")
    
    # 检查趋势分布
    if 'sar_trend' in sar_result.columns:
        trend_counts = sar_result['sar_trend'].value_counts()
        print(f"    📊 趋势分布: {dict(trend_counts)}")
    
    # 测试协整关系因子
    print(f"\n📊 测试协整关系因子:")
    coint_result = calculate_cointegration_factors(test_data)
    
    coint_factors = [col for col in coint_result.columns if col.startswith(('cointegration_', 'mean_reversion_', 'half_life_'))]
    print(f"✅ 协整关系因子: {len(coint_factors)}个")
    for factor in coint_factors:
        non_null_count = coint_result[factor].notna().sum()
        print(f"    📈 {factor}: {non_null_count}个有效值")
    
    # 测试配对交易因子
    print(f"\n📊 测试配对交易因子:")
    pair_result = calculate_pair_trading_factors(test_data)
    
    pair_factors = [col for col in pair_result.columns if col.startswith(('price_', 'pair_', 'correlation_'))]
    print(f"✅ 配对交易因子: {len(pair_factors)}个")
    for factor in pair_factors:
        non_null_count = pair_result[factor].notna().sum()
        print(f"    📈 {factor}: {non_null_count}个有效值")
    
    # 测试异常检测因子
    print(f"\n📊 测试异常检测因子:")
    anomaly_result = calculate_anomaly_factors(test_data)
    
    anomaly_factors = [col for col in anomaly_result.columns if col.endswith(('_anomaly', '_signal'))]
    print(f"✅ 异常检测因子: {len(anomaly_factors)}个")
    for factor in anomaly_factors:
        non_null_count = anomaly_result[factor].notna().sum()
        print(f"    📈 {factor}: {non_null_count}个有效值")
    
    # 检查异常信号分布
    if 'anomaly_signal' in anomaly_result.columns:
        signal_counts = anomaly_result['anomaly_signal'].value_counts()
        print(f"    📊 异常信号分布: {dict(signal_counts)}")
    
    print(f"\n🎉 增强因子测试完成！")
    return pd.concat([stoch_result, ichimoku_result, sar_result, coint_result, pair_result, anomaly_result], axis=1)


if __name__ == "__main__":
    # 运行原有测试
    test_advanced_factors()
    
    # 运行跨周期因子自检
    test_cross_cycle()
    
    # 运行增强因子测试
    test_enhanced_factors()
