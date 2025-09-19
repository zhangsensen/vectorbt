#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整因子探查脚本 - 个股独立IC分析
针对重点做因子探查的需求，实现：
1. 个股独立IC计算
2. 横截面统计
3. 跨时间框架因子稳定性排行
"""

import sys
import os

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from full_stock_analysis import FullStockAnalyzer

def main():
    """运行完整的因子探查分析"""
    print("🎯 开始完整因子探查分析...")
    print("🔍 重点：个股独立IC分析 → 横截面平均统计")
    
    # 创建分析器
    analyzer = FullStockAnalyzer()
    
    # 设置测试范围（可根据需要调整）
    test_symbols = [
        '0700.HK', '0005.HK', '0388.HK', '0981.HK', '1211.HK',  # 大盘股
        '0939.HK', '1288.HK', '1398.HK', '2018.HK', '2628.HK'   # 中盘股
    ]  # 10只代表性股票
    
    test_timeframes = ['15m', '1h', '4h', '1d']  # 4个关键时间框架
    
    print(f"📊 分析配置:")
    print(f"   股票数量: {len(test_symbols)}只")
    print(f"   时间框架: {test_timeframes}")
    print(f"   因子数量: {len(analyzer.test_factors)}个")
    print(f"   预计组合: {len(test_symbols) * len(test_timeframes) * len(analyzer.test_factors)}个")
    
    # 临时设置测试范围
    original_symbols = analyzer.analyzer.all_symbols
    original_timeframes = analyzer.analyzer.all_timeframes
    
    analyzer.analyzer.all_symbols = test_symbols
    analyzer.analyzer.all_timeframes = test_timeframes
    
    try:
        # 运行分析
        print("\n🚀 开始执行因子探查...")
        results = analyzer.run_comprehensive_analysis()
        
        # 保存结果
        print("\n💾 保存结果...")
        results_dir = analyzer.save_results(results)
        
        # 显示关键结果
        stats = results['overall_statistics']
        print(f"\n📈 探查结果:")
        print(f"   总组合数: {stats['total_combinations']}")
        print(f"   成功组合: {stats['successful_combinations']}")
        print(f"   成功率: {stats['overall_success_rate']:.1f}%")
        print(f"   执行时间: {stats['total_execution_time']:.2f}秒")
        
        # 显示各时间框架最佳因子
        print(f"\n🏆 各时间框架最佳因子:")
        for timeframe in test_timeframes:
            tf_results = results['timeframe_results'].get(timeframe, {})
            cross_sectional = tf_results.get('cross_sectional_summary', {})
            best_factor = cross_sectional.get('best_factor', {})
            
            if best_factor:
                factor_name = best_factor.get('factor_name', 'Unknown')
                ic_ir = best_factor.get('ic_ir', 0)
                mean_ic = best_factor.get('mean_ic', 0)
                pos_ratio = best_factor.get('positive_ic_ratio', 0)
                
                # 性能等级
                if abs(ic_ir) > 2.0:
                    grade = "🔥 卓越"
                elif abs(ic_ir) > 1.0:
                    grade = "🏆 优秀"
                elif abs(ic_ir) > 0.5:
                    grade = "✅ 良好"
                else:
                    grade = "⚠️ 一般"
                
                print(f"   {timeframe}: {grade} {factor_name}")
                print(f"        IC_IR={ic_ir:.2f}, 平均IC={mean_ic:.4f}, 正IC比例={pos_ratio:.1%}")
        
        print(f"\n📄 详细报告已保存: {results_dir}")
        print(f"🎯 重点查看: {results_dir}/factor_exploration_report.md")
        
        return True
        
    finally:
        # 恢复原始配置
        analyzer.analyzer.all_symbols = original_symbols
        analyzer.analyzer.all_timeframes = original_timeframes

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ 因子探查完成!")
        else:
            print("\n❌ 因子探查失败!")
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断分析")
    except Exception as e:
        print(f"\n💥 分析异常: {e}")
        import traceback
        traceback.print_exc()
