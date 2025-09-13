#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Categorical数据类型修复器 - 专门解决'Categorical' with dtype category问题
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class CategoricalDtypeFixer:
    """专门修复Categorical数据类型问题"""
    
    def __init__(self):
        """初始化修复器"""
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志"""
        import logging
        logger = logging.getLogger("{}.CategoricalFixer".format(__name__))
        logger.setLevel(logging.INFO)
        return logger
    
    def fix_categorical_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        修复DataFrame中的所有Categorical列
        
        Args:
            df: 包含可能Categorical列的DataFrame
            
        Returns:
            修复后的DataFrame
        """
        print("🔧 开始修复Categorical数据类型...")
        
        if df.empty:
            return df
            
        fixed_df = df.copy()
        categorical_columns = []
        conversion_errors = []
        
        for col in fixed_df.columns:
            try:
                col_data = fixed_df[col]
                
                # 检查是否为Categorical类型
                if hasattr(col_data, 'dtype') and col_data.dtype.name == 'category':
                    print("   发现Categorical列: {}".format(col))
                    categorical_columns.append(col)
                    
                    # 尝试转换为数值
                    try:
                        # 方法1: 直接转换category codes为数值
                        if hasattr(col_data, 'cat'):
                            # 获取category codes
                            numeric_data = col_data.cat.codes.astype(float)
                            # 将-1（NaN的code）转换为真正的NaN
                            numeric_data = numeric_data.replace(-1, np.nan)
                            fixed_df[col] = numeric_data
                            print("   ✅ {}: 成功转换为数值 (使用category codes)".format(col))
                            
                        else:
                            # 方法2: 强制转换为数值
                            fixed_df[col] = pd.to_numeric(col_data, errors='coerce')
                            print("   ✅ {}: 成功转换为数值 (强制转换)".format(col))
                            
                    except Exception as convert_error:
                        print("   ⚠️ {}: 转换失败，尝试备用方法".format(col))
                        
                        # 方法3: 转换为字符串再转数值
                        try:
                            str_data = col_data.astype(str)
                            numeric_data = pd.to_numeric(str_data, errors='coerce')
                            fixed_df[col] = numeric_data
                            print("   ✅ {}: 备用方法成功".format(col))
                        except Exception as backup_error:
                            print("   ❌ {}: 所有转换方法均失败 - {}".format(col, backup_error))
                            conversion_errors.append(col)
                            # 删除无法转换的列
                            fixed_df = fixed_df.drop(columns=[col])
                
                # 确保所有列都是数值类型
                elif not pd.api.types.is_numeric_dtype(fixed_df[col]):
                    print("   发现非数值列: {}, 尝试转换".format(col))
                    try:
                        fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')
                        print("   ✅ {}: 非数值列转换成功".format(col))
                    except Exception as e:
                        print("   ❌ {}: 非数值列转换失败 - {}".format(col, e))
                        conversion_errors.append(col)
                        fixed_df = fixed_df.drop(columns=[col])
                        
            except Exception as e:
                print("   ❌ 处理列 {} 时出错: {}".format(col, e))
                conversion_errors.append(col)
                continue
        
        print("   修复结果:")
        print("   - 发现Categorical列: {}个".format(len(categorical_columns)))
        print("   - 转换失败列: {}个".format(len(conversion_errors)))
        print("   - 最终有效列: {}个".format(len(fixed_df.columns)))
        
        if conversion_errors:
            print("   转换失败的列: {}".format(conversion_errors))
            
        return fixed_df
    
    def validate_numeric_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        验证DataFrame所有列都是数值类型，并进行基础清洗
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            (validated_df, validation_report)
        """
        print("🔍 开始数值类型验证...")
        
        if df.empty:
            return df, {'status': 'empty', 'issues': []}
        
        issues = []
        validated_df = df.copy()
        
        # 基础列（保留）
        base_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        existing_base_columns = [col for col in base_columns if col in df.columns]
        factor_columns = [col for col in df.columns if col not in base_columns]
        
        valid_factors = []
        problematic_factors = []
        
        for col in factor_columns:
            try:
                col_data = validated_df[col]
                
                # 检查数据类型
                if not pd.api.types.is_numeric_dtype(col_data):
                    issues.append("列 {} 不是数值类型: {}".format(col, col_data.dtype))
                    problematic_factors.append(col)
                    continue
                
                # 检查是否全为NaN
                if col_data.dropna().empty:
                    issues.append("列 {} 全为NaN".format(col))
                    problematic_factors.append(col)
                    continue
                
                # 检查常量列
                unique_count = col_data.dropna().nunique()
                if unique_count <= 1:
                    issues.append("列 {} 为常量 (unique={})".format(col, unique_count))
                    problematic_factors.append(col)
                    continue
                
                # 检查变异性
                try:
                    col_std = col_data.dropna().std()
                    if pd.isna(col_std) or col_std < 1e-8:
                        issues.append("列 {} 变异性过低 (std={})".format(col, col_std))
                        problematic_factors.append(col)
                        continue
                except Exception as std_error:
                    issues.append("列 {} std计算失败: {}".format(col, std_error))
                    problematic_factors.append(col)
                    continue
                
                valid_factors.append(col)
                
            except Exception as e:
                issues.append("列 {} 验证失败: {}".format(col, e))
                problematic_factors.append(col)
                continue
        
        # 构建最终的验证DataFrame
        final_columns = existing_base_columns + valid_factors
        final_df = validated_df[final_columns].copy()
        
        validation_report = {
            'status': 'completed',
            'original_columns': len(df.columns),
            'final_columns': len(final_df.columns),
            'valid_factors': valid_factors,
            'problematic_factors': problematic_factors,
            'issues': issues,
            'success_rate': len(valid_factors) / len(factor_columns) if factor_columns else 1.0
        }
        
        print("   验证结果:")
        print("   - 原始列数: {}".format(validation_report['original_columns']))
        print("   - 有效列数: {}".format(validation_report['final_columns']))
        print("   - 成功率: {:.1%}".format(validation_report['success_rate']))
        
        if issues:
            print("   - 发现问题: {}个".format(len(issues)))
            for issue in issues[:5]:  # 只显示前5个问题
                print("     • {}".format(issue))
            if len(issues) > 5:
                print("     • ... 还有{}个问题".format(len(issues)-5))
        
        return final_df, validation_report
    
    def comprehensive_fix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        综合修复：Categorical类型 + 数据验证
        
        Args:
            df: 原始DataFrame
            
        Returns:
            (fixed_df, fix_report)
        """
        print("🛠️ 开始综合修复...")
        
        if df.empty:
            return df, {'status': 'empty'}
        
        # 阶段1: 修复Categorical类型
        fixed_df = self.fix_categorical_dataframe(df)
        
        # 阶段2: 数值验证和清洗
        validated_df, validation_report = self.validate_numeric_dataframe(fixed_df)
        
        # 生成综合报告
        comprehensive_report = {
            'status': 'completed',
            'original_shape': df.shape,
            'final_shape': validated_df.shape,
            'categorical_fix': {
                'found_categorical': len([col for col in df.columns 
                                        if hasattr(df[col], 'dtype') and df[col].dtype.name == 'category']),
                'conversion_attempted': True
            },
            'validation': validation_report,
            'data_quality': {
                'columns_retained': len(validated_df.columns),
                'columns_removed': len(df.columns) - len(validated_df.columns),
                'removal_rate': (len(df.columns) - len(validated_df.columns)) / len(df.columns),
                'final_usable': not validated_df.empty and len(validated_df.columns) > 5
            }
        }
        
        print("🎯 综合修复完成:")
        print("   原始: {}".format(comprehensive_report['original_shape']))
        print("   最终: {}".format(comprehensive_report['final_shape']))
        print("   移除率: {:.1%}".format(comprehensive_report['data_quality']['removal_rate']))
        print("   可用性: {}".format('✅' if comprehensive_report['data_quality']['final_usable'] else '❌'))
        
        return validated_df, comprehensive_report


def test_categorical_fixer():
    """测试Categorical修复器"""
    print("🧪 测试Categorical修复器...")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'categorical_factor': pd.Categorical(['A', 'B', 'A', 'C', 'B']),
        'numeric_factor': [1.1, 2.2, 3.3, 4.4, 5.5],
        'constant_factor': [1, 1, 1, 1, 1],
        'nan_factor': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    
    print("测试数据类型:")
    print(test_data.dtypes)
    
    fixer = CategoricalDtypeFixer()
    fixed_data, report = fixer.comprehensive_fix(test_data)
    
    print("\n修复后数据类型:")
    print(fixed_data.dtypes)
    
    print("\n修复报告:")
    for key, value in report.items():
        print("  {}: {}".format(key, value))
    
    return fixed_data, report


if __name__ == "__main__":
    test_categorical_fixer()
