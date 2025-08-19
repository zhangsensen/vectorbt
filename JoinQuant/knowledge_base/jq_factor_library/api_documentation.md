# 聚宽因子库接口文档

## 1. 因子库概述

聚宽因子库提供了丰富的量化因子，包括基础因子、技术因子、财务因子、市场因子等，可以直接用于因子研究和多因子模型构建。

## 2. 获取因子数据

### 2.1 获取单个因子数据

**调用方法**
```python
from jqfactor import get_factor_values
get_factor_values(securities, factors, start_date, end_date, if_adjust_price=True)
```

**参数**
- securities: 证券代码列表
- factors: 因子名称列表
- start_date: 开始日期
- end_date: 结束日期
- if_adjust_price: 是否使用后复权价格计算因子，默认为True

**返回值**
dict，key为因子名称，value为DataFrame格式的因子数据

**示例**
```python
# 获取单个因子数据
securities = ['000001.XSHE', '000002.XSHE', '600000.XSHG']
factors = ['pe_ratio']
factor_data = get_factor_values(securities, factors, '2018-01-01', '2018-12-31')

# 获取多个因子数据
factors = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio']
factor_data = get_factor_values(securities, factors, '2018-01-01', '2018-12-31')
```

### 2.2 获取所有因子列表

**调用方法**
```python
from jqfactor import get_all_factors
get_all_factors()
```

**返回值**
DataFrame，包含所有因子的详细信息

**示例**
```python
# 获取所有因子列表
all_factors = get_all_factors()
print(all_factors.columns)
# Index(['factor', 'cn_name', 'en_name', 'category', 'description', 'data_source'], dtype='object')

# 按分类筛选因子
financial_factors = all_factors[all_factors['category'] == '财务因子']
```

## 3. 因子分类

### 3.1 基础因子
- market_cap: 总市值
- circulating_market_cap: 流通市值
- pe_ratio: 市盈率
- pb_ratio: 市净率
- ps_ratio: 市销率
- pcf_ratio: 市现率

### 3.2 技术因子
- momentum: 动量因子
- volatility: 波动率因子
- turnover: 换手率因子
- beta: 贝塔因子

### 3.3 财务因子
- roe: 净资产收益率
- roa: 总资产收益率
- gross_profit_margin: 毛利率
- net_profit_margin: 净利率
- asset_turnover: 资产周转率
- debt_to_asset: 资产负债率

### 3.4 成长因子
- revenue_growth: 营收增长率
- net_profit_growth: 净利润增长率
- eps_growth: 每股收益增长率

## 4. 因子处理和分析

### 4.1 因子标准化

```python
from jqfactor import standardize
import pandas as pd

# 获取因子数据
factor_data = get_factor_values(securities, ['pe_ratio'], '2018-01-01', '2018-12-31')
pe_ratio = factor_data['pe_ratio']

# Z-score标准化
pe_ratio_standardized = standardize(pe_ratio, method='z_score')

# 分位数标准化
pe_ratio_quantile = standardize(pe_ratio, method='quantile')
```

### 4.2 因子中性化

```python
from jqfactor import neutralize

# 按行业和市值中性化
factor_neutralized = neutralize(factor_data, 
                               groupby='sw_l1',  # 按申万一级行业分组
                               weights='market_cap')  # 以市值为权重
```

### 4.3 因子有效性检验

```python
from jqfactor import get_factor_return
import numpy as np

# 计算因子收益率
factor_return = get_factor_return(factor_data, 
                                 securities, 
                                 '2018-01-01', 
                                 '2018-12-31',
                                 method='long_short')  # 多空组合方法

# 计算信息比率
def information_ratio(factor_return, benchmark_return=None):
    if benchmark_return is None:
        benchmark_return = 0
    excess_return = factor_return - benchmark_return
    return np.mean(excess_return) / np.std(excess_return) * np.sqrt(252)
```

## 5. 多因子模型构建

### 5.1 因子合成

```python
# 等权合成
def equal_weight_composite(factor_dict):
    factors_df = pd.DataFrame(factor_dict)
    composite_factor = factors_df.mean(axis=1)
    return composite_factor

# 加权合成
def weighted_composite(factor_dict, weights):
    factors_df = pd.DataFrame(factor_dict)
    composite_factor = (factors_df * weights).sum(axis=1)
    return composite_factor

# 因子合成示例
factors = ['pe_ratio', 'pb_ratio', 'momentum', 'volatility']
factor_data = get_factor_values(securities, factors, '2018-01-01', '2018-12-31')

# 标准化处理
standardized_factors = {}
for factor in factors:
    standardized_factors[factor] = standardize(factor_data[factor], method='z_score')

# 等权合成
composite_factor = equal_weight_composite(standardized_factors)

# 加权合成（假设权重）
weights = {'pe_ratio': 0.3, 'pb_ratio': 0.3, 'momentum': 0.2, 'volatility': 0.2}
weighted_factor = weighted_composite(standardized_factors, weights)
```

### 5.2 因子回归分析

```python
from jqfactor import factor_regression

# 因子回归分析
regression_result = factor_regression(securities, 
                                     factor_data, 
                                     '2018-01-01', 
                                     '2018-12-31',
                                     dependent_variable='return')  # 以收益率为因变量
```

## 6. 因子回测

### 6.1 分层回测

```python
from jqfactor import quantile_analysis

# 分层回测
quantile_result = quantile_analysis(securities,
                                   factor_data['pe_ratio'],
                                   '2018-01-01',
                                   '2018-12-31',
                                   quantiles=5,  # 分为5层
                                   periods=[1, 5, 10])  # 不同期限
```

### 6.2 多因子选股策略

```python
# 多因子选股策略示例
def multi_factor_strategy(securities, factors, start_date, end_date):
    # 获取因子数据
    factor_data = get_factor_values(securities, factors, start_date, end_date)
    
    # 因子标准化
    standardized_factors = {}
    for factor in factors:
        standardized_factors[factor] = standardize(factor_data[factor], method='z_score')
    
    # 因子合成
    composite_factor = equal_weight_composite(standardized_factors)
    
    # 选股：选择因子值最高的前10%股票
    threshold = composite_factor.quantile(0.9)
    selected_stocks = composite_factor[composite_factor >= threshold].index.tolist()
    
    return selected_stocks

# 使用示例
factors = ['pe_ratio', 'pb_ratio', 'momentum']
selected = multi_factor_strategy(securities, factors, '2018-01-01', '2018-12-31')
```

## 7. 因子数据质量检查

### 7.1 缺失值处理

```python
# 检查因子数据缺失情况
def check_factor_missing(factor_data):
    missing_ratio = factor_data.isnull().sum() / len(factor_data)
    return missing_ratio

# 填充缺失值
def fill_factor_missing(factor_data, method='median'):
    if method == 'median':
        return factor_data.fillna(factor_data.median())
    elif method == 'mean':
        return factor_data.fillna(factor_data.mean())
    elif method == 'forward_fill':
        return factor_data.fillna(method='ffill')
```

### 7.2 异常值处理

```python
# Winsorization处理异常值
def winsorize_factor(factor_data, limits=(0.01, 0.99)):
    def winsorize_series(series, lower, upper):
        lower_quantile = series.quantile(lower)
        upper_quantile = series.quantile(upper)
        return series.clip(lower=lower_quantile, upper=upper_quantile)
    
    if isinstance(factor_data, pd.DataFrame):
        return factor_data.apply(lambda x: winsorize_series(x, limits[0], limits[1]))
    else:
        return winsorize_series(factor_data, limits[0], limits[1])
```

## 8. 常用因子示例

### 8.1 价值因子
```python
value_factors = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio']
```

### 8.2 成长因子
```python
growth_factors = ['revenue_growth', 'net_profit_growth', 'eps_growth']
```

### 8.3 质量因子
```python
quality_factors = ['roe', 'roa', 'gross_profit_margin', 'net_profit_margin']
```

### 8.4 动量因子
```python
momentum_factors = ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']
```

## 9. 注意事项

1. **数据频率**：因子数据通常为日频数据，注意与价格数据的对齐
2. **时滞问题**：因子计算通常使用T日数据预测T+1日收益，注意避免未来函数
3. **因子覆盖度**：不同因子在不同股票上的覆盖度可能不同，需要处理缺失值
4. **因子稳定性**：定期检验因子的有效性，避免因子失效
5. **计算效率**：对于大量股票和长时间序列，考虑优化计算效率
