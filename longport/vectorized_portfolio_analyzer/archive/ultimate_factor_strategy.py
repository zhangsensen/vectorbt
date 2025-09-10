#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极因子策略框架 - 产业级量化策略系统
集成30+指标、因子工程、高级IC分析、策略层优化
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vectorized_multi_stock_analyzer import VectorizedMultiStockAnalyzer
from advanced_factor_pool import AdvancedFactorPool
from factor_engineering import FactorEngineer
from advanced_ic_analysis import AdvancedICAnalyzer

class UltimateFactorStrategy:
    """终极因子策略 - 全栈量化解决方案"""
    
    def __init__(self, capital: float = 300000):
        """
        初始化终极因子策略
        
        Args:
            capital: 交易资金（港币）
        """
        print("🚀 初始化终极因子策略框架...")
        
        # 核心组件
        self.data_analyzer = VectorizedMultiStockAnalyzer()
        self.factor_pool = AdvancedFactorPool()
        self.factor_engineer = FactorEngineer()
        self.ic_analyzer = AdvancedICAnalyzer()
        
        # 资金配置
        self.capital = capital
        self.max_positions = 5  # 最多持仓数
        self.max_single_weight = 0.2  # 单只股票最大权重
        
        # 策略配置
        self.strategy_config = {
            'timeframes': ['15m', '1h', '4h', '1d'],  # 多周期
            'top_factors_per_tier': 3,  # 每层选择的顶级因子数
            'ensemble_method': 'weighted_average',  # 集成方法
            'rebalance_frequency': '1D',  # 再平衡频率
            'ic_window': 63,  # IC计算窗口
            'decay_analysis': True,  # 是否进行衰减分析
            'cost_adjustment': True,  # 是否进行成本调整
            'factor_engineering': True  # 是否进行因子工程
        }
        
        # 结果存储
        self.results = {}
        self.logger = self._setup_logger()
        
        print("✅ 终极因子策略初始化完成")
        
    def _setup_logger(self):
        """设置日志系统"""
        import logging
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/ultimate_strategy_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(f"{__name__}.UltimateStrategy")
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(f"{log_dir}/ultimate_strategy.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_ultimate_analysis(self, 
                            symbols: List[str], 
                            test_mode: bool = True) -> Dict:
        """
        运行终极因子分析
        
        Args:
            symbols: 股票代码列表
            test_mode: 是否为测试模式（影响计算深度）
            
        Returns:
            综合分析结果
        """
        print(f"🎯 开始终极因子分析...")
        print(f"   股票数量: {len(symbols)}")
        print(f"   时间框架: {self.strategy_config['timeframes']}")
        print(f"   测试模式: {test_mode}")
        
        start_time = time.time()
        analysis_results = {}
        
        # 阶段1：多时间框架数据加载与因子计算
        print("\n📊 阶段1: 多时间框架因子计算")
        factor_data_by_timeframe = {}
        
        for timeframe in self.strategy_config['timeframes']:
            print(f"   处理时间框架: {timeframe}")
            
            try:
                # 加载数据
                raw_data = self.data_analyzer.load_timeframe_data_vectorized(timeframe, symbols)
                
                # 计算高级因子
                factor_data = self._calculate_advanced_factors(raw_data, timeframe, test_mode)
                
                factor_data_by_timeframe[timeframe] = factor_data
                
                print(f"     ✅ {timeframe}: {factor_data.shape[1]}个因子")
                
            except Exception as e:
                self.logger.error(f"时间框架 {timeframe} 处理失败: {e}")
                continue
        
        # 阶段2：因子工程
        if self.strategy_config['factor_engineering']:
            print("\n🔧 阶段2: 因子工程")
            for timeframe in factor_data_by_timeframe:
                try:
                    engineered_factors = self.factor_engineer.process_factors(
                        factor_data_by_timeframe[timeframe],
                        methods=['cross_sectional', 'nonlinear', 'regime_based']
                    )
                    factor_data_by_timeframe[timeframe] = engineered_factors
                    
                except Exception as e:
                    self.logger.warning(f"因子工程失败 {timeframe}: {e}")
        
        # 阶段3：高级IC分析
        print("\n🔍 阶段3: 高级IC分析")
        ic_analysis_results = {}
        
        for timeframe in factor_data_by_timeframe:
            try:
                factor_data = factor_data_by_timeframe[timeframe]
                price_data = factor_data[['open', 'high', 'low', 'close', 'volume']]
                
                # 选择要分析的因子
                factor_columns = [col for col in factor_data.columns 
                                if col not in ['open', 'high', 'low', 'close', 'volume']]
                
                # 限制因子数量（测试模式）
                if test_mode:
                    factor_columns = factor_columns[:10]  # 只测试前10个因子
                
                # 高级IC分析
                ic_results = self.ic_analyzer.comprehensive_ic_analysis(
                    factor_data, price_data, factor_columns
                )
                
                ic_analysis_results[timeframe] = ic_results
                
                print(f"     ✅ {timeframe}: 分析了{len(factor_columns)}个因子")
                
            except Exception as e:
                self.logger.error(f"IC分析失败 {timeframe}: {e}")
                continue
        
        # 阶段4：因子选择与排序
        print("\n🎖️ 阶段4: 因子选择与排序")
        selected_factors = self._select_top_factors(ic_analysis_results, test_mode)
        
        # 阶段5：策略构建
        print("\n⚡ 阶段5: 策略构建")
        strategy_signals = self._build_strategy_signals(
            factor_data_by_timeframe, selected_factors, test_mode
        )
        
        # 阶段6：生成完整报告
        print("\n📋 阶段6: 生成综合报告")
        comprehensive_report = self._generate_comprehensive_report(
            ic_analysis_results, selected_factors, strategy_signals
        )
        
        # 汇总结果
        total_time = time.time() - start_time
        
        analysis_results = {
            'execution_time': total_time,
            'factor_data_by_timeframe': {tf: data.shape for tf, data in factor_data_by_timeframe.items()},
            'ic_analysis_results': ic_analysis_results,
            'selected_factors': selected_factors,
            'strategy_signals': strategy_signals,
            'comprehensive_report': comprehensive_report,
            'config': self.strategy_config,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        self._save_results(analysis_results)
        
        print(f"\n🎉 终极因子分析完成!")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   处理时间框架: {len(factor_data_by_timeframe)}个")
        print(f"   选择的顶级因子: {len(selected_factors)}个")
        
        return analysis_results
    
    def _calculate_advanced_factors(self, 
                                  raw_data: pd.DataFrame, 
                                  timeframe: str,
                                  test_mode: bool) -> pd.DataFrame:
        """计算高级因子"""
        try:
            # 为每只股票单独计算因子
            factor_data_list = []
            
            symbols = raw_data.index.get_level_values('symbol').unique()
            
            for symbol in symbols:
                symbol_data = raw_data.loc[symbol].copy()
                
                # 计算所有高级因子
                symbol_with_factors = self.factor_pool.calculate_all_factors(symbol_data)
                
                # 重新添加symbol索引
                symbol_with_factors.index = pd.MultiIndex.from_product(
                    [[symbol], symbol_with_factors.index], 
                    names=['symbol', 'timestamp']
                )
                
                factor_data_list.append(symbol_with_factors)
            
            # 合并所有股票的数据
            combined_data = pd.concat(factor_data_list)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"高级因子计算失败: {e}")
            return raw_data
    
    def _select_top_factors(self, 
                          ic_analysis_results: Dict, 
                          test_mode: bool) -> Dict:
        """选择顶级因子"""
        factor_scores = {}
        
        # 汇总所有时间框架的因子表现
        for timeframe, timeframe_results in ic_analysis_results.items():
            for factor_name, factor_analysis in timeframe_results.items():
                basic_ic = factor_analysis.get('basic_ic', {})
                
                # 计算综合得分
                ic_value = abs(basic_ic.get('ic', 0))
                ic_ir = abs(basic_ic.get('ic_ir', 0)) if not pd.isna(basic_ic.get('ic_ir', 0)) else 0
                sample_size = basic_ic.get('sample_size', 0)
                
                # 衰减分析加分
                decay_curve = factor_analysis.get('decay_curve', {})
                decay_bonus = 0
                if 'best_lag' in decay_curve and decay_curve['best_lag']:
                    decay_bonus = abs(decay_curve['best_lag'].get('ic', 0)) * 0.2
                
                # 综合得分
                composite_score = ic_value * 0.4 + ic_ir * 0.4 + decay_bonus * 0.2
                
                if sample_size > 20:  # 最小样本要求
                    factor_key = f"{factor_name}_{timeframe}"
                    factor_scores[factor_key] = {
                        'factor_name': factor_name,
                        'timeframe': timeframe,
                        'composite_score': composite_score,
                        'ic': ic_value,
                        'ic_ir': ic_ir,
                        'sample_size': sample_size,
                        'analysis': factor_analysis
                    }
        
        # 按得分排序
        sorted_factors = sorted(
            factor_scores.items(), 
            key=lambda x: x[1]['composite_score'], 
            reverse=True
        )
        
        # 选择顶级因子
        top_n = 5 if test_mode else 15
        selected = dict(sorted_factors[:top_n])
        
        print(f"   选择了 {len(selected)} 个顶级因子")
        for factor_key, factor_info in list(selected.items())[:3]:  # 显示前3个
            print(f"     🏆 {factor_info['factor_name']} ({factor_info['timeframe']}): {factor_info['composite_score']:.3f}")
        
        return selected
    
    def _build_strategy_signals(self, 
                              factor_data_by_timeframe: Dict,
                              selected_factors: Dict,
                              test_mode: bool) -> Dict:
        """构建策略信号"""
        print("   构建多周期集成信号...")
        
        signals_by_timeframe = {}
        
        # 为每个时间框架构建信号
        for timeframe in factor_data_by_timeframe:
            timeframe_data = factor_data_by_timeframe[timeframe]
            
            # 获取该时间框架的选中因子
            timeframe_factors = [
                info['factor_name'] for key, info in selected_factors.items()
                if info['timeframe'] == timeframe
            ]
            
            if not timeframe_factors:
                continue
            
            # 构建集成信号
            signals = self._create_ensemble_signals(timeframe_data, timeframe_factors)
            signals_by_timeframe[timeframe] = signals
        
        # 多时间框架信号融合
        final_signals = self._fuse_multi_timeframe_signals(signals_by_timeframe)
        
        return {
            'signals_by_timeframe': signals_by_timeframe,
            'final_signals': final_signals,
            'signal_stats': self._calculate_signal_stats(final_signals)
        }
    
    def _create_ensemble_signals(self, 
                               factor_data: pd.DataFrame,
                               factor_names: List[str]) -> pd.DataFrame:
        """创建集成信号"""
        available_factors = [f for f in factor_names if f in factor_data.columns]
        
        if not available_factors:
            return pd.DataFrame()
        
        # 标准化因子
        factor_matrix = factor_data[available_factors]
        
        # 横截面标准化
        normalized_factors = factor_matrix.groupby(level='timestamp').transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # 等权重集成
        ensemble_signal = normalized_factors.mean(axis=1)
        
        # 转换为DataFrame
        signals_df = pd.DataFrame({
            'raw_signal': ensemble_signal,
            'signal_rank': ensemble_signal.groupby(level='timestamp').rank(pct=True),
            'signal_zscore': ensemble_signal.groupby(level='timestamp').transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        })
        
        return signals_df
    
    def _fuse_multi_timeframe_signals(self, signals_by_timeframe: Dict) -> pd.DataFrame:
        """多时间框架信号融合"""
        if not signals_by_timeframe:
            return pd.DataFrame()
        
        # 简单的时间框架权重
        timeframe_weights = {
            '15m': 0.1,
            '1h': 0.3, 
            '4h': 0.4,
            '1d': 0.2
        }
        
        weighted_signals = []
        
        for timeframe, signals in signals_by_timeframe.items():
            if timeframe in timeframe_weights and not signals.empty:
                weight = timeframe_weights[timeframe]
                weighted_signal = signals['signal_zscore'] * weight
                weighted_signals.append(weighted_signal)
        
        if weighted_signals:
            final_signal = sum(weighted_signals)
            
            # 构建最终信号DataFrame
            final_signals = pd.DataFrame({
                'final_signal': final_signal,
                'position_size': self._calculate_position_sizes(final_signal),
                'signal_strength': np.abs(final_signal)
            })
            
            return final_signals
        
        return pd.DataFrame()
    
    def _calculate_position_sizes(self, signals: pd.Series) -> pd.Series:
        """计算仓位大小"""
        # 按信号强度分配仓位
        def allocate_positions(group):
            # 选择信号最强的股票
            top_signals = group.nlargest(self.max_positions)
            
            # 等权重分配
            position_size = 1.0 / len(top_signals)
            position_size = min(position_size, self.max_single_weight)
            
            positions = pd.Series(0.0, index=group.index)
            positions.loc[top_signals.index] = position_size
            
            return positions
        
        return signals.groupby(level='timestamp').transform(allocate_positions)
    
    def _calculate_signal_stats(self, final_signals: pd.DataFrame) -> Dict:
        """计算信号统计"""
        if final_signals.empty:
            return {}
        
        return {
            'avg_signal_strength': final_signals['signal_strength'].mean(),
            'signal_volatility': final_signals['final_signal'].std(),
            'avg_positions': (final_signals['position_size'] > 0).groupby(level='timestamp').sum().mean(),
            'turnover_proxy': final_signals['position_size'].groupby(level='symbol').diff().abs().mean()
        }
    
    def _generate_comprehensive_report(self, 
                                     ic_analysis_results: Dict,
                                     selected_factors: Dict,
                                     strategy_signals: Dict) -> str:
        """生成综合报告"""
        report = ["# 🚀 终极因子策略综合报告\n"]
        
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**分析资金**: {self.capital:,.0f} 港币")
        report.append(f"**最大持仓**: {self.max_positions} 只股票\n")
        
        # IC分析摘要
        report.append("## 📊 IC分析摘要\n")
        
        for timeframe, timeframe_results in ic_analysis_results.items():
            valid_factors = [name for name, analysis in timeframe_results.items() 
                           if analysis.get('basic_ic', {}).get('ic', 0) != 0]
            
            if valid_factors:
                avg_ic = np.mean([abs(timeframe_results[name]['basic_ic']['ic']) 
                                for name in valid_factors])
                report.append(f"- **{timeframe}**: {len(valid_factors)}个有效因子, 平均|IC|={avg_ic:.3f}")
        
        # 顶级因子排行
        report.append("\n## 🏆 顶级因子排行\n")
        report.append("| 排名 | 因子名称 | 时间框架 | 综合得分 | IC | IC_IR |")
        report.append("|------|----------|----------|----------|-----|-------|")
        
        for rank, (factor_key, factor_info) in enumerate(selected_factors.items(), 1):
            report.append(
                f"| {rank} | {factor_info['factor_name']} | {factor_info['timeframe']} | "
                f"{factor_info['composite_score']:.3f} | {factor_info['ic']:.3f} | "
                f"{factor_info['ic_ir']:.2f} |"
            )
            
            if rank >= 10:  # 只显示前10名
                break
        
        # 策略信号统计
        signal_stats = strategy_signals.get('signal_stats', {})
        if signal_stats:
            report.append("\n## ⚡ 策略信号统计\n")
            report.append(f"- **平均信号强度**: {signal_stats.get('avg_signal_strength', 0):.3f}")
            report.append(f"- **信号波动率**: {signal_stats.get('signal_volatility', 0):.3f}")
            report.append(f"- **平均持仓数**: {signal_stats.get('avg_positions', 0):.1f}")
            report.append(f"- **换手率代理**: {signal_stats.get('turnover_proxy', 0):.3f}")
        
        # 投资建议
        report.append("\n## 💡 投资建议\n")
        
        if len(selected_factors) > 0:
            best_factor = list(selected_factors.values())[0]
            report.append(f"### 🎯 核心推荐")
            report.append(f"**最佳因子**: {best_factor['factor_name']} ({best_factor['timeframe']})")
            report.append(f"- 综合得分: {best_factor['composite_score']:.3f}")
            report.append(f"- 建议使用该因子作为主要交易信号")
            
            report.append(f"\n### 📈 策略建议")
            report.append(f"- **资金分配**: 等权重分配到{self.max_positions}只股票")
            report.append(f"- **再平衡**: 每日检查信号变化")
            report.append(f"- **风险控制**: 单只股票最大权重{self.max_single_weight:.1%}")
        
        report.append(f"\n---")
        report.append(f"*报告由终极因子策略框架自动生成*")
        
        return "\n".join(report)
    
    def _save_results(self, results: Dict):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/ultimate_strategy_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(results_dir, "ultimate_strategy_results.json")
        
        # 处理不能序列化的对象
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存报告
        report_file = os.path.join(results_dir, "comprehensive_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(results['comprehensive_report'])
        
        self.logger.info(f"结果已保存到: {results_dir}")
        
        return results_dir
    
    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return f"<DataFrame/Series shape: {getattr(obj, 'shape', 'unknown')}>"
        elif isinstance(obj, np.ndarray):
            return f"<ndarray shape: {obj.shape}>"
        elif pd.isna(obj) or obj is None:
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)


def main():
    """主函数 - 运行终极因子策略"""
    print("🎯 启动终极因子策略框架测试...")
    
    # 创建策略实例
    strategy = UltimateFactorStrategy(capital=300000)
    
    # 测试股票列表
    test_symbols = [
        '0700.HK', '0005.HK', '0388.HK', '0981.HK', '1211.HK'
    ]
    
    try:
        # 运行分析
        results = strategy.run_ultimate_analysis(
            symbols=test_symbols,
            test_mode=True  # 测试模式
        )
        
        print("\n🎉 终极因子策略测试完成!")
        print(f"   分析耗时: {results['execution_time']:.2f}秒")
        print(f"   顶级因子: {len(results['selected_factors'])}个")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
