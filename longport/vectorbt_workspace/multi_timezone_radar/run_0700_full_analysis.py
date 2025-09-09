#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
0700.HK全时间框架全因子分析回测脚本
基于成功的0700单股测试模式，采用完整的时间框架和因子覆盖
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import gc
import traceback
import os

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from core.vectorbt_wfo_analyzer import VectorbtWFOAnalyzer

def setup_timestamp_logging():
    """设置基于时间戳的日志记录"""
    # 创建时间戳目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f"logs/0700_analysis_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件路径
    log_file = log_dir / "0700_full_analysis.log"
    
    # 清除现有的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"日志文件: {log_file}")
    
    # 测试日志写入
    logger.info("日志系统初始化完成")
    
    return logger, log_dir, timestamp

def save_analysis_parameters(log_dir, timestamp):
    """保存分析参数配置"""
    parameters = {
        "analysis_type": "0700.HK全时间框架全因子分析",
        "timestamp": timestamp,
        "stock": "0700.HK",
        "timeframes": ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        "factors": ["RSI", "Price_Position", "Momentum_ROC", "MACD", "Volume_Ratio"],
        "start_date": "2024-01-01",
        "end_date": "2025-09-01",
        "memory_limit_gb": 16.0,
        "n_workers": 6,
        "analysis_mode": "single_stock_comprehensive"
    }
    
    params_file = log_dir / "analysis_parameters.json"
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(parameters, f, ensure_ascii=False, indent=2)
    
    return parameters

def run_0700_comprehensive_analysis(logger, log_dir, timestamp):
    """执行0700.HK全时间框架全因子分析"""
    
    print("🚀 开始0700.HK全时间框架全因子分析...")
    logger.info("开始0700.HK全时间框架全因子分析")
    
    # 创建分析器
    analyzer = VectorbtWFOAnalyzer()
    
    # 设置测试参数
    analyzer.start_date = pd.to_datetime("2024-01-01")
    analyzer.end_date = pd.to_datetime("2025-09-01")
    analyzer.memory_limit_gb = 16.0
    analyzer.n_workers = 6
    
    # 测试股票
    test_stock = "0700.HK"
    
    # 全时间框架
    all_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    # 全因子
    all_factors = ["RSI", "Price_Position", "Momentum_ROC", "MACD", "Volume_Ratio"]
    
    print(f"📊 测试配置:")
    print(f"   股票: {test_stock}")
    print(f"   时间框架: {all_timeframes}")
    print(f"   因子: {all_factors}")
    print(f"   总测试数: {len(all_timeframes) * len(all_factors)}")
    print(f"   时间范围: {analyzer.start_date} 到 {analyzer.end_date}")
    
    logger.info(f"测试配置: {test_stock}, {all_timeframes}, {all_factors}")
    
    # 存储所有结果
    all_results = {}
    
    # 临时修改数据加载方法 - 关键：在股票级别修改
    original_load_method = analyzer.load_timeframe_data
    
    def load_single_stock(timeframe, symbols=None):
        return original_load_method(timeframe, [test_stock])
    
    analyzer.load_timeframe_data = load_single_stock
    
    # 创建所有测试任务
    test_tasks = []
    for factor in all_factors:
        for timeframe in all_timeframes:
            test_tasks.append((factor, timeframe))
    
    print(f"🔄 开始执行 {len(test_tasks)} 个测试...")
    
    # 执行所有测试
    completed_count = 0
    successful_count = 0
    
    for factor, timeframe in test_tasks:
        try:
            print(f"   📊 测试: {factor} - {timeframe}")
            logger.info(f"开始测试: {factor} - {timeframe}")
            
            # 执行单个测试
            result = analyzer.analyze_single_factor(factor, timeframe)
            
            # 记录结果
            key = f"{test_stock}_{factor}_{timeframe}"
            all_results[key] = {
                'success': True,
                'result': result,
                'factor': factor,
                'timeframe': timeframe,
                'robustness_score': result.get('robustness_score', 0),
                'ic_stats': result.get('ic_stats', {}),
                'signal_analysis': result.get('signal_analysis', {}),
                'vectorbt_results': result.get('vectorbt_results', {})
            }
            
            successful_count += 1
            logger.info(f"测试成功: {factor} - {timeframe}")
            
        except Exception as e:
            key = f"{test_stock}_{factor}_{timeframe}"
            error_msg = f"执行异常: {str(e)}\n{traceback.format_exc()}"
            all_results[key] = {
                'success': False,
                'error': error_msg,
                'factor': factor,
                'timeframe': timeframe
            }
            logger.error(f"测试失败: {factor} - {timeframe} - {error_msg}")
        
        completed_count += 1
        
        # 显示进度
        progress = completed_count / len(test_tasks) * 100
        print(f"      进度: {progress:.1f}% ({completed_count}/{len(test_tasks)})")
        
        # 定期清理内存
        if completed_count % 5 == 0:
            gc.collect()
    
    # 恢复数据加载方法
    analyzer.load_timeframe_data = original_load_method
    
    print(f"\n📊 0700.HK全量分析完成!")
    print(f"   总测试数: {len(all_results)}")
    print(f"   成功数: {successful_count}")
    print(f"   失败数: {len(all_results) - successful_count}")
    print(f"   成功率: {successful_count/len(all_results)*100:.1f}%")
    
    logger.info(f"全量分析完成: 总测试{len(all_results)}, 成功{successful_count}")
    
    # 生成报告
    generate_0700_comprehensive_report(all_results, test_stock, all_factors, all_timeframes, log_dir, timestamp)
    
    return all_results

def generate_0700_comprehensive_report(all_results, stock, factors, timeframes, log_dir, timestamp):
    """生成0700.HK综合分析报告"""
    print(f"\n📋 生成0700.HK综合分析报告...")
    
    # 统计信息
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results.values() if r.get('success', False))
    failed_tests = total_tests - successful_tests
    
    # 生成详细报告
    report_content = []
    report_content.append(f"# 0700.HK全时间框架全因子分析报告\n")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"分析批次: {timestamp}\n")
    
    # 测试摘要
    report_content.append("## 测试摘要\n")
    report_content.append(f"- 分析股票: {stock}\n")
    report_content.append(f"- 总测试数: {total_tests}\n")
    report_content.append(f"- 成功测试数: {successful_tests}\n")
    report_content.append(f"- 失败测试数: {failed_tests}\n")
    report_content.append(f"- 成功率: {successful_tests/total_tests*100:.1f}%\n")
    report_content.append(f"- 测试因子数: {len(factors)}\n")
    report_content.append(f"- 测试时间框架数: {len(timeframes)}\n\n")
    
    # 按因子分析结果
    report_content.append("## 按因子分析结果\n")
    factor_results = {}
    for key, result in all_results.items():
        if result.get('success', False):
            factor = result['factor']
            timeframe = result['timeframe']
            score = result['robustness_score']
            
            if factor not in factor_results:
                factor_results[factor] = []
            
            factor_results[factor].append({
                'timeframe': timeframe,
                'score': score,
                'ic_stats': result.get('ic_stats', {}),
                'signal_analysis': result.get('signal_analysis', {})
            })
    
    for factor in factors:
        if factor in factor_results:
            report_content.append(f"### {factor}\n")
            report_content.append("| 时间框架 | 稳健性得分 | 平均IC | IC IR | 信号质量 |\n")
            report_content.append("|----------|------------|--------|-------|----------|\n")
            
            for item in factor_results[factor]:
                timeframe = item['timeframe']
                score = item['score']
                ic_stats = item['ic_stats']
                signal_analysis = item['signal_analysis']
                
                mean_ic = ic_stats.get('mean_ic', 0)
                ic_ir = ic_stats.get('ic_ir', 0)
                signal_quality = signal_analysis.get('aggregated_stats', {}).get('signal_quality_score', 0)
                
                report_content.append(f"| {timeframe} | {score:.2f} | {mean_ic:.4f} | {ic_ir:.2f} | {signal_quality:.1f} |\n")
            
            report_content.append("\n")
    
    # 最佳表现组合
    report_content.append("## 最佳表现组合 (Top 10)\n")
    best_combinations = []
    for key, result in all_results.items():
        if result.get('success', False):
            factor = result['factor']
            timeframe = result['timeframe']
            score = result['robustness_score']
            ic_stats = result.get('ic_stats', {})
            signal_analysis = result.get('signal_analysis', {})
            
            mean_ic = ic_stats.get('mean_ic', 0)
            signal_quality = signal_analysis.get('aggregated_stats', {}).get('signal_quality_score', 0)
            
            best_combinations.append({
                'factor': factor,
                'timeframe': timeframe,
                'score': score,
                'mean_ic': mean_ic,
                'signal_quality': signal_quality
            })
    
    best_combinations.sort(key=lambda x: x['score'], reverse=True)
    
    report_content.append("| 排名 | 因子 | 时间框架 | 稳健性得分 | 平均IC | 信号质量 |\n")
    report_content.append("|------|------|----------|------------|--------|----------|\n")
    
    for i, combo in enumerate(best_combinations[:10]):
        report_content.append(f"| {i+1} | {combo['factor']} | {combo['timeframe']} | {combo['score']:.2f} | {combo['mean_ic']:.4f} | {combo['signal_quality']:.1f} |\n")
    
    report_content.append("\n")
    
    # 时间框架表现对比
    report_content.append("## 时间框架表现对比\n")
    timeframe_stats = {}
    for key, result in all_results.items():
        if result.get('success', False):
            timeframe = result['timeframe']
            score = result['robustness_score']
            
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = []
            
            timeframe_stats[timeframe].append(score)
    
    report_content.append("| 时间框架 | 平均得分 | 最高得分 | 最低得分 | 测试数 |\n")
    report_content.append("|----------|----------|----------|----------|--------|\n")
    
    for timeframe in timeframes:
        if timeframe in timeframe_stats:
            scores = timeframe_stats[timeframe]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            count = len(scores)
            
            report_content.append(f"| {timeframe} | {avg_score:.2f} | {max_score:.2f} | {min_score:.2f} | {count} |\n")
    
    report_content.append("\n")
    
    # 因子表现对比
    report_content.append("## 因子表现对比\n")
    factor_stats = {}
    for key, result in all_results.items():
        if result.get('success', False):
            factor = result['factor']
            score = result['robustness_score']
            
            if factor not in factor_stats:
                factor_stats[factor] = []
            
            factor_stats[factor].append(score)
    
    report_content.append("| 因子 | 平均得分 | 最高得分 | 最低得分 | 测试数 |\n")
    report_content.append("|------|----------|----------|----------|--------|\n")
    
    for factor in factors:
        if factor in factor_stats:
            scores = factor_stats[factor]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            count = len(scores)
            
            report_content.append(f"| {factor} | {avg_score:.2f} | {max_score:.2f} | {min_score:.2f} | {count} |\n")
    
    report_content.append("\n")
    
    # 错误分析
    if failed_tests > 0:
        report_content.append("## 错误分析\n")
        for key, result in all_results.items():
            if not result.get('success', False):
                factor = result['factor']
                timeframe = result['timeframe']
                error = result.get('error', 'Unknown error')
                
                report_content.append(f"### {factor} - {timeframe}\n")
                report_content.append(f"```\n{error}\n```\n\n")
    
    # 保存报告
    report_file = log_dir / "0700_comprehensive_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_content)
    
    # 保存JSON数据
    json_file = log_dir / "0700_full_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存统计摘要
    summary = {
        "analysis_timestamp": timestamp,
        "stock": stock,
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "success_rate": successful_tests/total_tests*100,
        "best_factor": best_combinations[0]['factor'] if best_combinations else None,
        "best_timeframe": best_combinations[0]['timeframe'] if best_combinations else None,
        "best_score": best_combinations[0]['score'] if best_combinations else None,
        "factor_stats": {factor: {"avg_score": np.mean(scores), "max_score": np.max(scores), "min_score": np.min(scores)} 
                        for factor, scores in factor_stats.items()},
        "timeframe_stats": {timeframe: {"avg_score": np.mean(scores), "max_score": np.max(scores), "min_score": np.min(scores)} 
                           for timeframe, scores in timeframe_stats.items()}
    }
    
    summary_file = log_dir / "analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"   📄 综合报告已保存: {report_file}")
    print(f"   📊 完整数据已保存: {json_file}")
    print(f"   📈 分析摘要已保存: {summary_file}")
    print(f"   📁 日志目录: {log_dir}")
    
    return summary

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 0700.HK全时间框架全因子分析")
    print("=" * 80)
    
    # 设置日志记录
    logger, log_dir, timestamp = setup_timestamp_logging()
    
    # 保存分析参数
    parameters = save_analysis_parameters(log_dir, timestamp)
    
    try:
        # 执行全量分析
        results = run_0700_comprehensive_analysis(logger, log_dir, timestamp)
        
        print(f"\n" + "=" * 80)
        print("🎉 0700.HK全时间框架全因子分析完成!")
        print(f"📁 结果保存在: {log_dir}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"分析执行失败: {str(e)}\n{traceback.format_exc()}")
        print(f"❌ 分析执行失败: {str(e)}")
        raise