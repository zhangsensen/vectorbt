#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
53只港股逐个全时间框架全因子分析回测脚本
参照0700.HK的分析方式，对每只股票进行独立分析
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
import time
import argparse
from logging.handlers import RotatingFileHandler

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from core.vectorbt_wfo_analyzer import VectorbtWFOAnalyzer

def setup_timestamp_logging_for_stock(stock):
    """为指定股票设置基于时间戳的日志记录"""
    # 创建时间戳目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f"logs/individual_analysis/{stock}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件路径
    log_file = log_dir / f"{stock}_full_analysis.log"
    
    # 清除现有的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志 - 使用滚动日志防止文件过大
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=3, encoding='utf-8', mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"分析股票: {stock}")
    logger.info("日志系统初始化完成")
    
    return logger, log_dir, timestamp

def save_analysis_parameters(log_dir, timestamp, stock):
    """保存分析参数配置"""
    parameters = {
        "analysis_type": f"{stock}全时间框架全因子分析",
        "timestamp": timestamp,
        "stock": stock,
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

def save_checkpoint(overall_results, start_index, end_index, timestamp):
    """保存分析检查点"""
    checkpoint_dir = Path(f"logs/sequential_analysis_summary_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    checkpoint_data = {
        "timestamp": timestamp,
        "start_index": start_index,
        "end_index": end_index,
        "overall_results": overall_results,
        "saved_at": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 检查点已保存: {checkpoint_file}")
    return checkpoint_file

def load_checkpoint(checkpoint_path):
    """加载分析检查点"""
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        print(f"📂 检查点已加载: {checkpoint_path}")
        print(f"   分析范围: 第{checkpoint_data['start_index']+1}到第{checkpoint_data['end_index']}只股票")
        print(f"   已完成股票数: {len(checkpoint_data['overall_results'])}")
        
        return checkpoint_data
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None

def run_single_stock_full_analysis(stock, logger, log_dir, timestamp):
    """执行单只股票的全时间框架全因子分析"""
    
    print(f"🚀 开始{stock}全时间框架全因子分析...")
    logger.info(f"开始{stock}全时间框架全因子分析")
    
    # 创建分析器
    analyzer = VectorbtWFOAnalyzer()
    
    # 设置测试参数 - 确保时区一致性
    analyzer.start_date = pd.to_datetime("2024-01-01").tz_localize('Asia/Hong_Kong')
    analyzer.end_date = pd.to_datetime("2025-09-01").tz_localize('Asia/Hong_Kong')
    analyzer.memory_limit_gb = 16.0
    
    # 动态设置工作进程数
    import psutil
    analyzer.n_workers = min(6, psutil.cpu_count(logical=False) or 4)
    
    # 全时间框架
    all_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    # 全因子
    all_factors = ["RSI", "Price_Position", "Momentum_ROC", "MACD", "Volume_Ratio"]
    
    print(f"📊 测试配置:")
    print(f"   股票: {stock}")
    print(f"   时间框架: {all_timeframes}")
    print(f"   因子: {all_factors}")
    print(f"   总测试数: {len(all_timeframes) * len(all_factors)}")
    print(f"   时间范围: {analyzer.start_date} 到 {analyzer.end_date}")
    
    logger.info(f"测试配置: {stock}, {all_timeframes}, {all_factors}")
    
    # 存储所有结果
    all_results = {}
    
    # 临时修改数据加载方法 - 关键：在股票级别修改
    original_load_method = analyzer.load_timeframe_data
    
    def load_single_stock(timeframe, symbols=None):
        return original_load_method(timeframe, [stock])
    
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
            key = f"{stock}_{factor}_{timeframe}"
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
            key = f"{stock}_{factor}_{timeframe}"
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
            analyzer.clear_cache()
    
    # 恢复数据加载方法
    analyzer.load_timeframe_data = original_load_method
    
    print(f"\n📊 {stock}全量分析完成!")
    print(f"   总测试数: {len(all_results)}")
    print(f"   成功数: {successful_count}")
    print(f"   失败数: {len(all_results) - successful_count}")
    print(f"   成功率: {successful_count/len(all_results)*100:.1f}%")
    
    logger.info(f"全量分析完成: 总测试{len(all_results)}, 成功{successful_count}")
    
    # 生成报告
    generate_single_stock_comprehensive_report(all_results, stock, all_factors, all_timeframes, log_dir, timestamp)
    
    return all_results

def generate_single_stock_comprehensive_report(all_results, stock, factors, timeframes, log_dir, timestamp):
    """生成单只股票的综合分析报告"""
    print(f"\n📋 生成{stock}综合分析报告...")
    
    # 统计信息
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results.values() if r.get('success', False))
    failed_tests = total_tests - successful_tests
    
    # 生成详细报告
    report_content = []
    report_content.append(f"# {stock}全时间框架全因子分析报告\n")
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
    report_file = log_dir / f"{stock}_comprehensive_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_content)
    
    # 保存JSON数据
    json_file = log_dir / f"{stock}_full_results.json"
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
        json.dump(summary, f, ensure_ascii=False, indent=2, default=int)
    
    print(f"   📄 综合报告已保存: {report_file}")
    print(f"   📊 完整数据已保存: {json_file}")
    print(f"   📈 分析摘要已保存: {summary_file}")
    print(f"   📁 日志目录: {log_dir}")
    
    return summary

def get_all_hk_stocks():
    """获取所有可用的港股股票列表"""
    data_dir = Path("/Users/zhangshenshen/longport/vectorbt_workspace/data/1m")
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    # 获取所有parquet文件
    stock_files = list(data_dir.glob("*.HK.parquet"))
    
    # 提取股票代码并排序
    hk_stocks = sorted([f.stem for f in stock_files])
    
    print(f"📊 找到 {len(hk_stocks)} 只港股")
    
    return hk_stocks

def run_sequential_analysis(start_index=0, end_index=None, overall_results=None, timestamp=None):
    """逐个执行股票分析"""
    # 获取所有股票
    all_stocks = get_all_hk_stocks()
    
    if not all_stocks:
        print("❌ 没有找到可用的港股数据")
        return
    
    # 过滤掉0700.HK（已经分析过了）
    stocks_to_analyze = [stock for stock in all_stocks if stock != "0700.HK"]
    
    if end_index is None:
        end_index = len(stocks_to_analyze)
    
    # 确保索引范围有效
    start_index = max(0, min(start_index, len(stocks_to_analyze) - 1))
    end_index = max(start_index, min(end_index, len(stocks_to_analyze)))
    
    selected_stocks = stocks_to_analyze[start_index:end_index]
    
    # 如果没有提供overall_results，初始化空列表
    if overall_results is None:
        overall_results = []
    
    # 如果没有提供timestamp，生成新的
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"🚀 开始逐个股票分析")
    print(f"   总股票数: {len(stocks_to_analyze)}")
    print(f"   分析范围: 第{start_index+1}到第{end_index}只股票")
    print(f"   本次分析: {len(selected_stocks)}只股票")
    print(f"   股票列表: {selected_stocks}")
    if overall_results:
        print(f"   已完成股票数: {len(overall_results)}")
    
    # 记录总体结果
    overall_results = overall_results or []
    
    for i, stock in enumerate(selected_stocks):
        print(f"\n{'='*80}")
        print(f"📈 分析股票 {i+1}/{len(selected_stocks)}: {stock}")
        print(f"{'='*80}")
        
        try:
            # 设置日志记录
            logger, log_dir, timestamp = setup_timestamp_logging_for_stock(stock)
            
            # 保存分析参数
            parameters = save_analysis_parameters(log_dir, timestamp, stock)
            
            # 执行分析
            start_time = time.time()
            results = run_single_stock_full_analysis(stock, logger, log_dir, timestamp)
            end_time = time.time()
            
            # 记录结果
            successful_tests = sum(1 for r in results.values() if r.get('success', False))
            total_tests = len(results)
            success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0
            
            stock_result = {
                "stock": stock,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "execution_time": end_time - start_time,
                "log_dir": str(log_dir),
                "timestamp": timestamp
            }
            
            overall_results.append(stock_result)
            
            print(f"\n✅ {stock}分析完成!")
            print(f"   成功率: {success_rate:.1f}%")
            print(f"   执行时间: {end_time - start_time:.1f}秒")
            print(f"   日志目录: {log_dir}")
            
            # 保存检查点
            save_checkpoint(overall_results, start_index, end_index, timestamp)
            
            # 短暂休息，避免系统过载
            if i < len(selected_stocks) - 1:
                print(f"⏳ 休息5秒后继续...")
                time.sleep(5)
            
        except Exception as e:
            print(f"❌ {stock}分析失败: {str(e)}")
            overall_results.append({
                "stock": stock,
                "total_tests": 0,
                "successful_tests": 0,
                "success_rate": 0,
                "execution_time": 0,
                "log_dir": None,
                "timestamp": None,
                "error": str(e)
            })
    
    # 生成总体报告
    generate_overall_report(overall_results, start_index, end_index)
    
    return overall_results

def generate_overall_report(overall_results, start_index, end_index):
    """生成总体分析报告"""
    print(f"\n{'='*80}")
    print(f"📋 生成总体分析报告")
    print(f"{'='*80}")
    
    # 计算统计信息
    total_stocks = len(overall_results)
    successful_stocks = sum(1 for r in overall_results if r['success_rate'] > 0)
    total_tests = sum(r['total_tests'] for r in overall_results)
    successful_tests = sum(r['successful_tests'] for r in overall_results)
    total_time = sum(r['execution_time'] for r in overall_results)
    
    overall_success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0
    
    # 生成报告
    report_content = []
    report_content.append(f"# 53只港股逐个分析总体报告\n")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_content.append(f"分析范围: 第{start_index+1}到第{end_index}只股票\n\n")
    
    # 总体统计
    report_content.append("## 总体统计\n")
    report_content.append(f"- 分析股票数: {total_stocks}\n")
    report_content.append(f"- 成功股票数: {successful_stocks}\n")
    report_content.append(f"- 总测试数: {total_tests}\n")
    report_content.append(f"- 成功测试数: {successful_tests}\n")
    report_content.append(f"- 总体成功率: {overall_success_rate:.1f}%\n")
    report_content.append(f"- 总执行时间: {total_time:.1f}秒\n")
    report_content.append(f"- 平均每只股票时间: {total_time/total_stocks:.1f}秒\n\n")
    
    # 每只股票的结果
    report_content.append("## 各股票分析结果\n")
    report_content.append("| 股票 | 总测试数 | 成功测试数 | 成功率 | 执行时间(秒) | 状态 |\n")
    report_content.append("|------|----------|------------|--------|-------------|------|\n")
    
    for result in overall_results:
        stock = result['stock']
        total_tests = result['total_tests']
        successful_tests = result['successful_tests']
        success_rate = result['success_rate']
        execution_time = result['execution_time']
        status = "✅ 成功" if success_rate > 0 else "❌ 失败"
        
        report_content.append(f"| {stock} | {total_tests} | {successful_tests} | {success_rate:.1f}% | {execution_time:.1f} | {status} |\n")
    
    report_content.append("\n")
    
    # 最佳表现股票
    successful_results = [r for r in overall_results if r['success_rate'] > 0]
    if successful_results:
        successful_results.sort(key=lambda x: x['success_rate'], reverse=True)
        
        report_content.append("## 最佳表现股票 (Top 10)\n")
        report_content.append("| 排名 | 股票 | 成功率 | 成功测试数 | 总测试数 | 执行时间(秒) |\n")
        report_content.append("|------|------|--------|------------|----------|-------------|\n")
        
        for i, result in enumerate(successful_results[:10]):
            report_content.append(f"| {i+1} | {result['stock']} | {result['success_rate']:.1f}% | {result['successful_tests']} | {result['total_tests']} | {result['execution_time']:.1f} |\n")
    
    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(f"logs/sequential_analysis_summary_{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / "overall_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_content)
    
    # 保存JSON数据
    json_file = report_dir / "overall_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"📄 总体报告已保存: {report_file}")
    print(f"📊 完整数据已保存: {json_file}")
    print(f"📁 报告目录: {report_dir}")
    
    # 打印摘要
    print(f"\n🎉 逐个股票分析完成!")
    print(f"   分析股票数: {total_stocks}")
    print(f"   成功股票数: {successful_stocks}")
    print(f"   总体成功率: {overall_success_rate:.1f}%")
    print(f"   总执行时间: {total_time:.1f}秒")

def main():
    parser = argparse.ArgumentParser(description='53只港股逐个全时间框架全因子分析')
    parser.add_argument('--start', type=int, default=0, help='开始索引（从0开始）')
    parser.add_argument('--end', type=int, help='结束索引（不包含）')
    parser.add_argument('--stock', type=str, help='指定单个股票代码（如：0005.HK）')
    parser.add_argument('--resume', type=str, help='从检查点恢复分析（指定检查点文件路径）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 53只港股逐个全时间框架全因子分析")
    print("=" * 80)
    
    if args.stock:
        # 分析单个股票
        print(f"📊 分析单个股票: {args.stock}")
        
        try:
            # 设置日志记录
            logger, log_dir, timestamp = setup_timestamp_logging_for_stock(args.stock)
            
            # 保存分析参数
            parameters = save_analysis_parameters(log_dir, timestamp, args.stock)
            
            # 执行分析
            results = run_single_stock_full_analysis(args.stock, logger, log_dir, timestamp)
            
            print(f"\n" + "=" * 80)
            print(f"🎉 {args.stock}分析完成!")
            print(f"📁 结果保存在: {log_dir}")
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ {args.stock}分析失败: {str(e)}")
            raise
    elif args.resume:
        # 从检查点恢复分析
        print(f"📂 从检查点恢复分析: {args.resume}")
        
        checkpoint_data = load_checkpoint(args.resume)
        if checkpoint_data:
            # 从检查点恢复分析
            run_sequential_analysis(
                start_index=checkpoint_data['start_index'],
                end_index=checkpoint_data['end_index'],
                overall_results=checkpoint_data['overall_results'],
                timestamp=checkpoint_data['timestamp']
            )
        else:
            print("❌ 无法加载检查点，退出")
            sys.exit(1)
    else:
        # 逐个分析多个股票
        run_sequential_analysis(args.start, args.end)

if __name__ == "__main__":
    main()