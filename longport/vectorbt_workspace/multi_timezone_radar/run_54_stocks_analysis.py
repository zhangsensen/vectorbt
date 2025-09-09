#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
54只港股全量因子分析回测脚本
基于0700单股测试的成功模式，采用全局数据加载方法修改
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from core.vectorbt_wfo_analyzer import VectorbtWFOAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_54_stocks_analysis_fixed.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_hk_stocks():
    """加载港股股票列表 - 基于实际可用的数据"""
    # 从数据目录中读取实际可用的股票
    data_dir = Path("/Users/zhangshenshen/longport/vectorbt_workspace/data/1m")
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    
    # 获取所有parquet文件
    stock_files = list(data_dir.glob("*.parquet"))
    
    # 提取股票代码
    hk_stocks = [f.stem for f in stock_files if f.stem.endswith('.HK')]
    
    logger.info(f"从数据目录加载了 {len(hk_stocks)} 只股票")
    
    return hk_stocks

def run_full_analysis():
    """执行54只股票全量分析"""
    
    print("🚀 开始54只港股全量因子分析...")
    logger.info("开始54只港股全量因子分析")
    
    # 创建分析器
    analyzer = VectorbtWFOAnalyzer()
    
    # 设置测试参数
    analyzer.start_date = pd.to_datetime("2024-01-01")  # 扩大时间范围
    analyzer.end_date = pd.to_datetime("2025-09-01")
    analyzer.memory_limit_gb = 16.0
    analyzer.n_workers = 6  # 减少工作进程避免内存过载
    
    # 获取股票列表
    all_stocks = load_hk_stocks()
    
    # 测试因子 - 选择核心因子
    test_factors = ["RSI", "Price_Position", "Momentum_ROC", "MACD", "Volume_Ratio"]
    
    # 主要时间框架 - 选择关键时间框架
    key_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    print(f"📊 测试配置:")
    print(f"   股票数量: {len(all_stocks)}")
    print(f"   因子: {test_factors}")
    print(f"   时间框架: {key_timeframes}")
    print(f"   总测试数: {len(all_stocks) * len(test_factors) * len(key_timeframes)}")
    print(f"   时间范围: {analyzer.start_date} 到 {analyzer.end_date}")
    
    logger.info(f"测试配置: {len(all_stocks)}只股票, {test_factors}, {key_timeframes}")
    
    # 存储所有结果
    all_results = {}
    
    # 分批处理股票，避免内存过载
    batch_size = 9  # 每批9只股票，共6批
    total_batches = (len(all_stocks) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_stocks))
        batch_stocks = all_stocks[start_idx:end_idx]
        
        print(f"\n🔄 处理第 {batch_idx + 1}/{total_batches} 批股票: {batch_stocks}")
        logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批股票: {batch_stocks}")
        
        # 批量分析
        batch_results = analyze_stock_batch(analyzer, batch_stocks, test_factors, key_timeframes)
        
        # 合并结果
        all_results.update(batch_results)
        
        # 清理内存
        gc.collect()
        
        print(f"✅ 第 {batch_idx + 1} 批完成")
        logger.info(f"第 {batch_idx + 1} 批完成")
    
    print(f"\n📊 全量分析完成!")
    successful_count = sum(1 for r in all_results.values() if r.get('success', False))
    print(f"   总测试数: {len(all_results)}")
    print(f"   成功数: {successful_count}")
    print(f"   失败数: {len(all_results) - successful_count}")
    
    logger.info(f"全量分析完成: 总测试{len(all_results)}, 成功{successful_count}")
    
    # 生成报告
    generate_comprehensive_report(all_results, all_stocks, test_factors, key_timeframes)
    
    return all_results

def analyze_stock_batch(analyzer, stocks, factors, timeframes):
    """分析一批股票 - 参考0700单股测试模式"""
    batch_results = {}
    
    # 为每只股票创建测试任务
    for stock in stocks:
        print(f"   📊 处理股票: {stock}")
        
        # 临时修改数据加载方法 - 关键：在股票级别修改，不是在测试级别
        original_load_method = analyzer.load_timeframe_data
        
        def load_single_stock(timeframe, symbols=None):
            return original_load_method(timeframe, [stock])
        
        analyzer.load_timeframe_data = load_single_stock
        
        # 创建所有测试任务
        test_tasks = []
        for factor in factors:
            for timeframe in timeframes:
                test_tasks.append((factor, timeframe))
        
        # 使用线程池并行处理这只股票的所有测试
        completed_count = 0
        successful_count = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(run_single_test_for_stock, analyzer, factor, timeframe): (factor, timeframe)
                for factor, timeframe in test_tasks
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                factor, timeframe = future_to_task[future]
                try:
                    result = future.result()
                    key = f"{stock}_{factor}_{timeframe}"
                    batch_results[key] = result
                    
                    completed_count += 1
                    success = result.get('success', False)
                    if success:
                        successful_count += 1
                    
                    # 显示进度
                    if completed_count % 10 == 0:
                        progress = completed_count / len(test_tasks) * 100
                        print(f"      {stock} 进度: {progress:.1f}% ({completed_count}/{len(test_tasks)})")
                    
                    # 记录错误
                    if not success:
                        error_msg = result.get('error', 'Unknown error')
                        logger.error(f"测试失败 {stock} {factor} {timeframe}: {error_msg}")
                
                except Exception as e:
                    key = f"{stock}_{factor}_{timeframe}"
                    error_msg = f"执行异常: {str(e)}"
                    batch_results[key] = {'success': False, 'error': error_msg, 'stock': stock, 'factor': factor, 'timeframe': timeframe}
                    logger.error(f"测试异常 {stock} {factor} {timeframe}: {error_msg}")
                    completed_count += 1
        
        # 恢复数据加载方法
        analyzer.load_timeframe_data = original_load_method
        
        print(f"   ✅ {stock} 完成: {successful_count}/{len(test_tasks)} 成功")
    
    return batch_results

def run_single_test_for_stock(analyzer, factor, timeframe):
    """为单只股票运行单个测试 - 参考0700的run_single_test函数"""
    try:
        # 直接调用analyze_single_factor，不需要传递symbols参数
        # 因为已经在股票级别修改了load_timeframe_data方法
        result = analyzer.analyze_single_factor(factor, timeframe)
        
        if 'error' in result:
            return {
                'success': False,
                'error': result['error'],
                'factor': factor,
                'timeframe': timeframe
            }
        
        return {
            'success': True,
            'result': result,
            'factor': factor,
            'timeframe': timeframe,
            'robustness_score': result.get('robustness_score', 0),
            'ic_stats': result.get('ic_stats', {}),
            'signal_analysis': result.get('signal_analysis', {}),
            'vectorbt_results': result.get('vectorbt_results', {})
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"执行异常: {str(e)}\n{traceback.format_exc()}",
            'factor': factor,
            'timeframe': timeframe
        }

def generate_comprehensive_report(all_results, stocks, factors, timeframes):
    """生成综合报告"""
    print(f"\n📋 生成综合分析报告...")
    
    # 创建报告目录
    report_dir = Path(f"full_54_stocks_analysis_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    report_dir.mkdir(exist_ok=True)
    
    # 统计信息
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results.values() if r.get('success', False))
    failed_tests = total_tests - successful_tests
    
    # 生成详细报告
    report_content = []
    report_content.append(f"# 54只港股全量因子分析报告 (修正版)\n")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 测试摘要
    report_content.append("## 测试摘要\n")
    report_content.append(f"- 总测试数: {total_tests}\n")
    report_content.append(f"- 成功测试数: {successful_tests}\n")
    report_content.append(f"- 失败测试数: {failed_tests}\n")
    report_content.append(f"- 成功率: {successful_tests/total_tests*100:.1f}%\n")
    report_content.append(f"- 测试股票数: {len(stocks)}\n")
    report_content.append(f"- 测试因子数: {len(factors)}\n")
    report_content.append(f"- 测试时间框架数: {len(timeframes)}\n\n")
    
    # 最佳因子分析
    report_content.append("## 最佳因子分析\n")
    best_factors = analyze_best_factors(all_results)
    for i, (factor, stats) in enumerate(best_factors):
        if stats['count'] > 0:  # 只显示有数据的因子
            report_content.append(f"{i+1}. **{factor}**: 平均得分 {stats['avg_score']:.2f}, 成功率 {stats['success_rate']:.1f}%\n")
    report_content.append("\n")
    
    # 最佳时间框架分析
    report_content.append("## 最佳时间框架分析\n")
    best_timeframes = analyze_best_timeframes(all_results)
    for i, (timeframe, stats) in enumerate(best_timeframes):
        if stats['count'] > 0:  # 只显示有数据的时间框架
            report_content.append(f"{i+1}. **{timeframe}**: 平均得分 {stats['avg_score']:.2f}, 成功率 {stats['success_rate']:.1f}%\n")
    report_content.append("\n")
    
    # 最佳股票分析
    report_content.append("## 最佳股票分析 (Top 10)\n")
    best_stocks = analyze_best_stocks(all_results)
    for i, (stock, stats) in enumerate(best_stocks[:10]):
        if stats['count'] > 0:  # 只显示有数据的股票
            report_content.append(f"{i+1}. **{stock}**: 平均得分 {stats['avg_score']:.2f}, 成功率 {stats['success_rate']:.1f}%\n")
    report_content.append("\n")
    
    # 最佳组合分析
    report_content.append("## 最佳组合分析 (Top 20)\n")
    best_combinations = analyze_best_combinations(all_results)
    report_content.append("| 股票 | 时间框架 | 因子 | 稳健性得分 | 平均IC | 信号质量 |\n")
    report_content.append("|------|----------|------|------------|--------|----------|\n")
    
    for i, combo in enumerate(best_combinations[:20]):
        stock = combo['stock']
        timeframe = combo['timeframe']
        factor = combo['factor']
        score = combo['score']
        avg_ic = combo.get('avg_ic', 0)
        signal_quality = combo.get('signal_quality', 0)
        
        report_content.append(f"| {stock} | {timeframe} | {factor} | {score:.2f} | {avg_ic:.4f} | {signal_quality:.1f} |\n")
    
    report_content.append("\n")
    
    # 错误分析
    if failed_tests > 0:
        report_content.append("## 错误分析\n")
        error_analysis = analyze_errors(all_results)
        for error_type, count in error_analysis.items():
            report_content.append(f"- **{error_type}**: {count}次\n")
        report_content.append("\n")
    
    # 保存报告
    report_file = report_dir / "comprehensive_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_content)
    
    # 保存JSON数据
    json_file = report_dir / "full_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"   📄 综合报告已保存: {report_file}")
    print(f"   📊 完整数据已保存: {json_file}")
    print(f"   📁 报告目录: {report_dir}")
    
    logger.info(f"综合报告已生成: {report_file}")

def analyze_best_factors(all_results):
    """分析最佳因子"""
    factor_stats = {}
    
    for key, result in all_results.items():
        if result.get('success', False):
            factor = result['factor']
            score = result['robustness_score']
            
            if factor not in factor_stats:
                factor_stats[factor] = {'scores': [], 'count': 0, 'success': 0}
            
            factor_stats[factor]['scores'].append(score)
            factor_stats[factor]['count'] += 1
            factor_stats[factor]['success'] += 1
        else:
            factor = result['factor']
            if factor not in factor_stats:
                factor_stats[factor] = {'scores': [], 'count': 0, 'success': 0}
            factor_stats[factor]['count'] += 1
    
    # 计算平均得分和成功率
    best_factors = []
    for factor, stats in factor_stats.items():
        if stats['scores']:
            avg_score = np.mean(stats['scores'])
            success_rate = stats['success'] / stats['count'] * 100
            best_factors.append((factor, {'avg_score': avg_score, 'success_rate': success_rate, 'count': stats['count']}))
    
    best_factors.sort(key=lambda x: x[1]['avg_score'], reverse=True)
    return best_factors

def analyze_best_timeframes(all_results):
    """分析最佳时间框架"""
    timeframe_stats = {}
    
    for key, result in all_results.items():
        if result.get('success', False):
            timeframe = result['timeframe']
            score = result['robustness_score']
            
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = {'scores': [], 'count': 0, 'success': 0}
            
            timeframe_stats[timeframe]['scores'].append(score)
            timeframe_stats[timeframe]['count'] += 1
            timeframe_stats[timeframe]['success'] += 1
        else:
            timeframe = result['timeframe']
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = {'scores': [], 'count': 0, 'success': 0}
            timeframe_stats[timeframe]['count'] += 1
    
    # 计算平均得分和成功率
    best_timeframes = []
    for timeframe, stats in timeframe_stats.items():
        if stats['scores']:
            avg_score = np.mean(stats['scores'])
            success_rate = stats['success'] / stats['count'] * 100
            best_timeframes.append((timeframe, {'avg_score': avg_score, 'success_rate': success_rate, 'count': stats['count']}))
    
    best_timeframes.sort(key=lambda x: x[1]['avg_score'], reverse=True)
    return best_timeframes

def analyze_best_stocks(all_results):
    """分析最佳股票"""
    stock_stats = {}
    
    for key, result in all_results.items():
        if result.get('success', False):
            # 从key中提取股票代码，格式为 "stock_factor_timeframe"
            stock = key.split('_')[0]
            score = result['robustness_score']
            
            if stock not in stock_stats:
                stock_stats[stock] = {'scores': [], 'count': 0, 'success': 0}
            
            stock_stats[stock]['scores'].append(score)
            stock_stats[stock]['count'] += 1
            stock_stats[stock]['success'] += 1
        else:
            stock = key.split('_')[0]
            if stock not in stock_stats:
                stock_stats[stock] = {'scores': [], 'count': 0, 'success': 0}
            stock_stats[stock]['count'] += 1
    
    # 计算平均得分和成功率
    best_stocks = []
    for stock, stats in stock_stats.items():
        if stats['scores']:
            avg_score = np.mean(stats['scores'])
            success_rate = stats['success'] / stats['count'] * 100
            best_stocks.append((stock, {'avg_score': avg_score, 'success_rate': success_rate, 'count': stats['count']}))
    
    best_stocks.sort(key=lambda x: x[1]['avg_score'], reverse=True)
    return best_stocks

def analyze_best_combinations(all_results):
    """分析最佳组合"""
    best_combinations = []
    
    for key, result in all_results.items():
        if result.get('success', False):
            # 从key中解析信息
            parts = key.split('_')
            stock = parts[0]
            factor = parts[1]
            timeframe = parts[2]
            score = result['robustness_score']
            
            ic_stats = result.get('ic_stats', {})
            signal_analysis = result.get('signal_analysis', {})
            
            avg_ic = ic_stats.get('mean_ic', 0)
            signal_quality = signal_analysis.get('aggregated_stats', {}).get('signal_quality_score', 0)
            
            best_combinations.append({
                'stock': stock,
                'factor': factor,
                'timeframe': timeframe,
                'score': score,
                'avg_ic': avg_ic,
                'signal_quality': signal_quality
            })
    
    best_combinations.sort(key=lambda x: x['score'], reverse=True)
    return best_combinations

def analyze_errors(all_results):
    """分析错误"""
    error_stats = {}
    
    for key, result in all_results.items():
        if not result.get('success', False):
            error = result.get('error', 'Unknown error')
            
            # 简化错误分类
            if '数据加载失败' in error:
                error_type = '数据加载失败'
            elif '内存不足' in error:
                error_type = '内存不足'
            elif '超时' in error:
                error_type = '超时'
            elif '除零' in error:
                error_type = '除零错误'
            else:
                error_type = '其他错误'
            
            error_stats[error_type] = error_stats.get(error_type, 0) + 1
    
    return error_stats

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 54只港股全量因子分析 (修正版)")
    print("=" * 80)
    
    # 执行全量分析
    results = run_full_analysis()
    
    print(f"\n" + "=" * 80)
    print("🎉 54只港股全量分析完成!")
    print("=" * 80)