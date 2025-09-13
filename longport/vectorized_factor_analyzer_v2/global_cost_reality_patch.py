#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局成本现实化补丁 - 统一4.5‱成本模型
所有时间框架使用相同的实盘成本标准
"""

import pandas as pd
from types import MethodType

def patch_global_cost_reality(engine_instance):
    """
    全局成本现实化补丁
    所有时间框架统一使用4.5‱实盘成本
    """
    
    # 三轨制实盘成本标准（港股）
    REALITY_MIN_COST = {
        'hft': 2.2 / 10000,   # 2.2‱ 高频高滑点
        'mft': 1.7 / 10000,   # 1.7‱ 中频中等滑点  
        'lft': 1.3 / 10000,   # 1.3‱ 低频低滑点
    }
    
    def _cost_adjustment_screening_reality(self, df, timeframe, symbol):
        """统一成本调整函数"""
        print(f"[调试] 全局成本调整: timeframe={timeframe}, df_shape={df.shape if not df.empty else 'empty'}")
        
        if df.empty:
            return df
        
        # 获取时间框架层级
        tier = self._get_timeframe_tier(timeframe)
        cfg = self.thresholds[tier]
        print(f"[调试] tier={tier}, cfg_sharpe_cost={cfg['sharpe_cost']}")
        
        results = []
        print(f"[调试] 处理 {len(df)} 个因子...")
        
        for idx, row in df.iterrows():
            print(f"[调试] === 处理因子 {idx+1}/{len(df)}: {row['factor']} ===")
            print(f"[调试]   原始数据: 夏普={row['sharpe_mean']:.3f}, 年化收益={row['annual_return']:.4f}, "
                  f"波动率={row['annual_volatility']:.4f}, 年交易次数={row['trades_per_year']:.1f}")
            
            # 获取流动性层级（但仍使用统一成本）
            liquidity_tier = self._get_liquidity_tier(symbol)
            
            # 🔥 使用三轨制实盘成本
            base_cost = REALITY_MIN_COST[tier]  # 分层成本
            timeframe_mult = 1.0
            
            total_cost_bps = base_cost * timeframe_mult
            
            # 计算年化成本
            position_turnover_ratio = min(1.0, row['trades_per_year'] / 1000)
            annual_cost = row['trades_per_year'] * total_cost_bps * position_turnover_ratio
            
            # 三轨制成本上限
            cost_cap_ratio = 0.30 if tier == 'hft' else 0.35 if tier == 'mft' else 0.40
            annual_cost = min(annual_cost, row['annual_return'] * cost_cap_ratio)
            
            # 计算成本调整后的夏普率
            sharpe_cost = (row['annual_return'] - annual_cost) / row['annual_volatility'] \
                          if row['annual_volatility'] > 0 else 0
            
            print(f"[调试]   成本计算: 统一成本={base_cost*10000:.2f}‱, 总成本={total_cost_bps*10000:.2f}‱, "
                  f"换手率={position_turnover_ratio:.3f}")
            print(f"[调试]   年化成本={annual_cost:.4f} (封顶前={row['annual_return'] * cost_cap_ratio:.4f})")
            print(f"[调试]   结果: 成本后收益={row['annual_return']-annual_cost:.4f}, "
                  f"成本后夏普={sharpe_cost:.3f}, 阈值={cfg['sharpe_cost']:.3f}")
            
            if sharpe_cost >= cfg['sharpe_cost']:
                print(f"[调试] ✅ {row['factor']} 通过成本筛选")
                row_copy = row.copy()
                row_copy['sharpe_cost'] = sharpe_cost
                row_copy['annual_cost'] = annual_cost
                row_copy['cost_cap_ratio'] = cost_cap_ratio
                row_copy['liquidity_tier'] = liquidity_tier
                row_copy['base_cost_bps'] = base_cost * 10000
                row_copy['total_cost_bps'] = total_cost_bps * 10000
                results.append(row_copy)
            else:
                print(f"[调试] ❌ {row['factor']} 未通过成本筛选")
        
        print(f"[调试] === 成本筛选完成，通过 {len(results)}/{len(df)} 个因子 ===")
        return pd.DataFrame(results)
    
    # 保存原方法
    if not hasattr(engine_instance, '_cost_adjustment_screening_original'):
        engine_instance._cost_adjustment_screening_original = engine_instance._cost_adjustment_screening
    
    # 绑定新方法
    engine_instance._cost_adjustment_screening = MethodType(
        _cost_adjustment_screening_reality, engine_instance
    )
    
    print(f"[全局成本补丁] 已启用：三轨制分层成本，HFT={REALITY_MIN_COST['hft']*10000:.1f}‱, MFT={REALITY_MIN_COST['mft']*10000:.1f}‱, LFT={REALITY_MIN_COST['lft']*10000:.1f}‱")
    
    return engine_instance

def create_cross_timeframe_test_data():
    """创建跨时间框架测试数据"""
    test_data = []
    
    # 1d 框架因子
    test_data.extend([
        {'factor': 'kama_14_1d', 'timeframe': '1d', 'sharpe_mean': 0.504, 'annual_return': 0.2426, 
         'annual_volatility': 0.4807, 'trades_per_year': 58.6, 'win_rate_mean': 0.7746, 'trades_sum': 667,
         'max_drawdown_mean': 0.152, 'profit_loss_ratio_mean': 0.9566, 'signal_strength_mean': 1.2169},
        {'factor': 'bb_squeeze_1d', 'timeframe': '1d', 'sharpe_mean': 0.491, 'annual_return': 0.2300,
         'annual_volatility': 0.4677, 'trades_per_year': 56.9, 'win_rate_mean': 0.7835, 'trades_sum': 613,
         'max_drawdown_mean': 0.148, 'profit_loss_ratio_mean': 1.3383, 'signal_strength_mean': 1.2909},
    ])
    
    # 4h 框架因子
    test_data.extend([
        {'factor': 'trend_strength_4h', 'timeframe': '4h', 'sharpe_mean': 0.46, 'annual_return': 0.22,
         'annual_volatility': 0.48, 'trades_per_year': 120, 'win_rate_mean': 0.75, 'trades_sum': 800,
         'max_drawdown_mean': 0.16, 'profit_loss_ratio_mean': 1.1, 'signal_strength_mean': 1.2},
        {'factor': 'momentum_4h', 'timeframe': '4h', 'sharpe_mean': 0.44, 'annual_return': 0.21,
         'annual_volatility': 0.48, 'trades_per_year': 110, 'win_rate_mean': 0.74, 'trades_sum': 750,
         'max_drawdown_mean': 0.17, 'profit_loss_ratio_mean': 1.0, 'signal_strength_mean': 1.15},
    ])
    
    # 1h 框架因子
    test_data.extend([
        {'factor': 'rsi_100_1h', 'timeframe': '1h', 'sharpe_mean': 0.42, 'annual_return': 0.20,
         'annual_volatility': 0.48, 'trades_per_year': 200, 'win_rate_mean': 0.73, 'trades_sum': 1200,
         'max_drawdown_mean': 0.18, 'profit_loss_ratio_mean': 0.9, 'signal_strength_mean': 1.1},
        {'factor': 'macd_1h', 'timeframe': '1h', 'sharpe_mean': 0.40, 'annual_return': 0.19,
         'annual_volatility': 0.48, 'trades_per_year': 180, 'win_rate_mean': 0.72, 'trades_sum': 1100,
         'max_drawdown_mean': 0.19, 'profit_loss_ratio_mean': 0.85, 'signal_strength_mean': 1.05},
    ])
    
    # 30m 框架因子（中频）
    test_data.extend([
        {'factor': 'adx_14_30m', 'timeframe': '30m', 'sharpe_mean': 0.38, 'annual_return': 0.18,
         'annual_volatility': 0.48, 'trades_per_year': 300, 'win_rate_mean': 0.71, 'trades_sum': 1500,
         'max_drawdown_mean': 0.20, 'profit_loss_ratio_mean': 0.8, 'signal_strength_mean': 1.0},
        {'factor': 'cci_30m', 'timeframe': '30m', 'sharpe_mean': 0.36, 'annual_return': 0.17,
         'annual_volatility': 0.48, 'trades_per_year': 280, 'win_rate_mean': 0.70, 'trades_sum': 1400,
         'max_drawdown_mean': 0.21, 'profit_loss_ratio_mean': 0.75, 'signal_strength_mean': 0.95},
    ])
    
    # 5m 框架因子（高频）
    test_data.extend([
        {'factor': 'rsi_5m', 'timeframe': '5m', 'sharpe_mean': 0.15, 'annual_return': 0.08,
         'annual_volatility': 0.50, 'trades_per_year': 2000, 'win_rate_mean': 0.65, 'trades_sum': 8000,
         'max_drawdown_mean': 0.25, 'profit_loss_ratio_mean': 0.6, 'signal_strength_mean': 0.8},
        {'factor': 'macd_5m', 'timeframe': '5m', 'sharpe_mean': 0.12, 'annual_return': 0.06,
         'annual_volatility': 0.50, 'trades_per_year': 1800, 'win_rate_mean': 0.62, 'trades_sum': 7200,
         'max_drawdown_mean': 0.28, 'profit_loss_ratio_mean': 0.55, 'signal_strength_mean': 0.75},
    ])
    
    # 添加 days 字段
    for item in test_data:
        item['days'] = 252
    
    return pd.DataFrame(test_data)

def test_global_cost_model():
    """测试全局成本模型"""
    print("🌍 全局成本现实化测试")
    print("=" * 50)
    
    # 创建测试数据
    test_data = create_cross_timeframe_test_data()
    print(f"📊 测试数据: {len(test_data)} 个因子，覆盖 {test_data['timeframe'].nunique()} 个时间框架")
    
    # 测试原始引擎
    print("\n🔵 测试原始引擎...")
    engine_original = FactorFilterEngine(mode='loose', debug=False)
    
    results_original = {}
    for tf in test_data['timeframe'].unique():
        tf_data = test_data[test_data['timeframe'] == tf]
        result = engine_original.filter_factors(tf_data, tf, '0700.HK')
        results_original[tf] = result
    
    # 测试补丁引擎
    print("\n🔴 测试全局成本补丁...")
    engine_patched = FactorFilterEngine(mode='loose', debug=False)
    engine_patched = patch_global_cost_reality(engine_patched)
    
    results_patched = {}
    for tf in test_data['timeframe'].unique():
        tf_data = test_data[test_data['timeframe'] == tf]
        result = engine_patched.filter_factors(tf_data, tf, '0700.HK')
        results_patched[tf] = result
    
    return {
        'original': results_original,
        'patched': results_patched,
        'test_data': test_data
    }

if __name__ == "__main__":
    from core.factor_filter.engine_complete import FactorFilterEngine
    
    # 运行全局测试
    results = test_global_cost_model()
    
    # 分析结果
    print("\n📊 全局成本模型效果分析")
    print("=" * 60)
    
    test_data = results['test_data']
    for tf in test_data['timeframe'].unique():
        orig_result = results['original'][tf]
        patch_result = results['patched'][tf]
        
        print(f"\n时间框架: {tf}")
        print(f"  原始通过: {len(orig_result['valid_factors'])} 个因子")
        print(f"  补丁通过: {len(patch_result['valid_factors'])} 个因子")
        
        if len(patch_result['valid_factors']) > 0:
            avg_sharpe = patch_result['valid_factors']['sharpe_cost'].mean()
            avg_cost = patch_result['valid_factors']['annual_cost'].mean()
            print(f"  平均夏普: {avg_sharpe:.3f}")
            print(f"  平均成本: {avg_cost:.4f} ({avg_cost*100:.2f}%)")