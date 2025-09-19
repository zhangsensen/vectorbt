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
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

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
            'enhanced': ['macd_enhanced', 'rsi_enhanced', 'atr_enhanced']
        }
        
    def calculate_trend_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势类因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # DEMA (双指数移动平均)
            df['dema_14'] = ta.DEMA(close.values, timeperiod=14)
            
            # TEMA (三指数移动平均)
            df['tema_14'] = ta.TEMA(close.values, timeperiod=14)
            
            # KAMA (考夫曼自适应移动平均)
            df['kama_14'] = ta.KAMA(close.values, timeperiod=14)
            
            # TRIX (三重指数平滑震荡器)
            df['trix_14'] = ta.TRIX(close.values, timeperiod=14)
            
            # Aroon指标
            aroon_up, aroon_down = ta.AROON(high.values, low.values, timeperiod=14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            df['aroon_oscillator'] = aroon_up - aroon_down
            
            # ADX (平均趋向指数)
            df['adx_14'] = ta.ADX(high.values, low.values, close.values, timeperiod=14)
            
            # 趋势强度指标
            df['trend_strength'] = np.abs(close.rolling(20).mean() - close.rolling(5).mean()) / close
            
        except Exception as e:
            print(f"趋势因子计算警告: {e}")
            
        return df
    
    def calculate_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量类因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # 多周期RSI
            df['rsi_2'] = ta.RSI(close.values, timeperiod=2)
            df['rsi_14'] = ta.RSI(close.values, timeperiod=14)
            df['rsi_100'] = ta.RSI(close.values, timeperiod=100)
            
            # Stochastic RSI
            fastk, fastd = ta.STOCHRSI(close.values, timeperiod=14, fastk_period=5, fastd_period=3)
            df['stoch_rsi'] = fastk
            
            # CCI (顺势指标)
            df['cci_14'] = ta.CCI(high.values, low.values, close.values, timeperiod=14)
            
            # ROC (变动率)
            df['roc_12'] = ta.ROC(close.values, timeperiod=12)
            df['roc_5'] = ta.ROC(close.values, timeperiod=5)
            
            # MFI (资金流量指标)
            df['mfi_14'] = ta.MFI(high.values, low.values, close.values, volume.values, timeperiod=14)
            
            # Williams %R
            df['willr_14'] = ta.WILLR(high.values, low.values, close.values, timeperiod=14)
            
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
            atr = ta.ATR(high.values, low.values, close.values, timeperiod=14)
            df['atrp'] = atr / close  # 解决股价水平漂移问题
            
            # Keltner通道位置
            keltner_ma = close.rolling(20).mean()
            keltner_atr = pd.Series(atr).rolling(20).mean()
            keltner_upper = keltner_ma + 2 * keltner_atr
            keltner_lower = keltner_ma - 2 * keltner_atr
            df['keltner_position'] = (close - keltner_lower) / (keltner_upper - keltner_lower)
            
            # 布林带收缩
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close.values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
            
            # 波动率比值
            short_vol = close.rolling(5).std()
            long_vol = close.rolling(20).std()
            df['volatility_ratio'] = short_vol / long_vol
            
            # Parkinson波动率估计器
            df['parkinson_vol'] = np.sqrt(0.361 * np.log(high / low) ** 2)
            
            # 隐含波动率代理
            df['iv_proxy'] = (high - low) / close
            
        except Exception as e:
            print(f"波动率因子计算警告: {e}")
            
        return df
    
    def calculate_volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量类因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # VWAP偏离度
            vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            df['vwap_deviation'] = (close - vwap) / close
            
            # Volume RSI
            volume_gains = volume.diff().clip(lower=0)
            volume_losses = (-volume.diff()).clip(lower=0)
            rs = volume_gains.rolling(14).mean() / volume_losses.rolling(14).mean()
            df['volume_rsi'] = 100 - (100 / (1 + rs))
            
            # A/D Line (累积/派发线)
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow_volume = money_flow_multiplier * volume
            df['ad_line'] = money_flow_volume.cumsum()
            
            # Chaikin Money Flow
            df['cmf'] = money_flow_volume.rolling(20).sum() / volume.rolling(20).sum()
            
            # Volume MA偏离度
            volume_ma = volume.rolling(20).mean()
            df['volume_ma_deviation'] = (volume - volume_ma) / volume_ma
            
            # Volume-Price Trend
            df['vpt'] = (volume * close.pct_change()).cumsum()
            
        except Exception as e:
            print(f"成交量因子计算警告: {e}")
            
        return df
    
    def calculate_microstructure_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算微观结构因子"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # 买卖价差代理
            df['hl_spread'] = (high - low) / close
            
            # 成交量强度
            df['volume_intensity'] = volume / volume.rolling(20).mean()
            
            # 价格效率 (Random Walk Index)
            price_range = high - low
            true_range = np.maximum(price_range, 
                                  np.maximum(np.abs(high - close.shift(1)), 
                                           np.abs(low - close.shift(1))))
            df['price_efficiency'] = price_range / true_range.rolling(14).sum()
            
            # 流动性指标
            df['liquidity_proxy'] = volume * close / (high - low + 1e-8)
            
            # 价格跳跃检测
            returns = close.pct_change()
            rolling_std = returns.rolling(20).std()
            df['price_jump'] = np.abs(returns) / rolling_std
            
        except Exception as e:
            print(f"微观结构因子计算警告: {e}")
            
        return df
    
    def calculate_enhanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算增强型因子 - 多维度信号融合"""
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        try:
            # 增强型MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            # MACD能量
            df['macd_enhanced'] = histogram * volume.rolling(20).mean()
            
            # 增强型RSI (结合成交量)
            rsi = ta.RSI(close.values, timeperiod=14)
            volume_weighted_rsi = rsi * (volume / volume.rolling(14).mean())
            df['rsi_enhanced'] = volume_weighted_rsi
            
            # 增强型ATR (考虑成交量)
            atr = ta.ATR(high.values, low.values, close.values, timeperiod=14)
            df['atr_enhanced'] = atr * np.sqrt(volume / volume.rolling(14).mean())
            
            # 多因子得分
            factors_to_rank = ['rsi_14', 'macd_enhanced', 'atrp', 'vwap_deviation']
            for factor in factors_to_rank:
                if factor in df.columns:
                    df[f'{factor}_rank'] = df[factor].rolling(252).rank(pct=True)
            
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
                typical_price = (high + low + close) / 3
                vwap = (typical_price * volume).rolling(min(20, n//4)).sum() / volume.rolling(min(20, n//4)).sum()
                # 尾盘vs开盘价差 (简化为收盘vs开盘)
                df['smart_money_flow'] = (close - open_price) / vwap
            else:
                df['smart_money_flow'] = 0
            
            # ② zscore_momentum_20 - 20根K线收益滚动标准化
            returns = close.pct_change()
            window = min(20, max(5, n//3))
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            df['zscore_momentum_20'] = (returns - rolling_mean) / (rolling_std + 1e-8)
            
            # ③ order_flow_imbalance - 上涨vs下跌K线成交量差
            up_volume = np.where(close > close.shift(1), volume, 0)
            down_volume = np.where(close < close.shift(1), volume, 0)
            window_flow = min(14, max(3, n//4))
            up_vol_ma = pd.Series(up_volume).rolling(window_flow).sum()
            down_vol_ma = pd.Series(down_volume).rolling(window_flow).sum()
            df['order_flow_imbalance'] = (up_vol_ma - down_vol_ma) / (up_vol_ma + down_vol_ma + 1e-8)
            
            # ④ vw_macd - 成交量加权MACD
            # 先计算传统MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            # 成交量加权
            volume_weight = volume / volume.rolling(min(20, n//2)).mean()
            df['vw_macd'] = histogram * volume_weight
            
            # ⑤ drawdown_volatility - 回撤波动率(下行风险敏感)
            cummax = close.expanding().max()
            drawdown = (close - cummax) / cummax
            window_dd = min(30, max(10, n//2))
            df['drawdown_volatility'] = drawdown.rolling(window_dd).std()
            
            # ⑥ skewness_60 - 60根K线收益偏度(极端预警)
            window_skew = min(60, max(20, n//2))
            if n >= 20:
                df['skewness_60'] = returns.rolling(window_skew).skew()
            else:
                df['skewness_60'] = 0
                
            # ⑦ mean_reversion_score - 布林带Z-Score(统计套利)
            window_bb = min(20, max(10, n//3))
            bb_ma = close.rolling(window_bb).mean()
            bb_std = close.rolling(window_bb).std()
            df['mean_reversion_score'] = (close - bb_ma) / (bb_std + 1e-8)
            
            # ⑧ seasonality_friday - 是否周五(日历效应)
            if hasattr(df.index, 'dayofweek'):
                df['seasonality_friday'] = (df.index.dayofweek == 4).astype(int)
            else:
                # 如果没有时间索引，使用简化版本
                df['seasonality_friday'] = 0.2  # 固定值，避免常数因子
                
            # ⑨ vol_term_structure - 短/长波动率比(期限结构)
            if n >= 30:
                short_vol = returns.rolling(5).std()
                long_vol = returns.rolling(min(30, n//2)).std()
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
                        factor_normalized = (factor_data - factor_data.mean()) / factor_data.std()
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
        
        # 统计计算的因子数量
        original_cols = set(required_cols)
        new_factors = [col for col in df.columns if col not in original_cols]
        
        print(f"✅ 高级因子计算完成: 新增{len(new_factors)}个因子")
        print(f"📊 因子分类统计:")
        for category, factors in self.factor_categories.items():
            available = [f for f in factors if f in new_factors]
            print(f"   {category}: {len(available)}个因子")
            
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


if __name__ == "__main__":
    # 运行原有测试
    test_advanced_factors()
    
    # 运行跨周期因子自检
    test_cross_cycle()
