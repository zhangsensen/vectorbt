# 行业数据接口文档

## 1. 获取行业分类信息

### 1.1 获取行业列表

获取所有行业分类信息

**调用方法**
```python
from jqdata import *
get_industries(name, date=None)
```

**参数**
- name: 行业分类名称，可选值：'sw_l1'(申万一级行业)、'sw_l2'(申万二级行业)、'sw_l3'(申万三级行业)、'zjw'(证监会行业分类)、'jq_l1'(聚宽一级行业)、'jq_l2'(聚宽二级行业)
- date: 查询日期, 一个字符串或者[datetime.date]/[datetime.datetime]对象, 可以是None

**返回值**
[pandas.DataFrame], 包含以下列：
- industry_code: 行业编码
- industry_name: 行业名称
- start_date: 开始日期
- end_date: 结束日期

**示例**
```python
# 获取申万一级行业列表
get_industries('sw_l1')

# 获取指定日期的证监会行业分类
get_industries('zjw', date='2018-06-01')
```

### 1.2 获取行业成份股

获取在给定日期一个行业板块的所有股票

**调用方法**
```python
get_industry_stocks(industry_code, date=None)
```

**参数**
- industry_code: 行业编码
- date: 查询日期, 一个字符串或者[datetime.date]/[datetime.datetime]对象, 可以是None

**返回值**
返回股票代码的list

**示例**
```python
# 获取计算机/互联网行业的成分股
stocks = get_industry_stocks('I64')
```

### 1.3 查询股票所属行业

**调用方法**
```python
get_industry(security, date=None)
```

**参数**
- security：标的代码，类型为字符串，形式如"000001.XSHE"；或为包含标的代码字符串的列表
- date：查询的日期

**返回值**
返回结果是一个dict，key是传入的股票代码

**示例**
```python
# 获取贵州茅台的所属行业数据
d = get_industry("600519.XSHG", date="2018-06-01")

# 同时获取多只股票的所属行业信息
stock_list = ['000001.XSHE','000002.XSHE']
d = get_industry(security=stock_list, date="2018-06-01")
```

## 2. 行业数据获取

### 2.1 获取行业历史数据

获取行业指数的历史行情数据

**调用方法**
```python
get_price(security, start_date, end_date, frequency='1d', fields=None, skip_paused=False, fq='pre')
```

**参数**
- security: 行业指数代码，格式如'HY001.XSHG'
- start_date: 开始日期
- end_date: 结束日期
- frequency: 数据频率，默认为'1d'
- fields: 字段名或者 list, 可选
- skip_paused: 是否跳过停牌日期
- fq: 复权方式

**示例**
```python
# 获取金融指数的历史数据
get_price('HY007.XSHG', start_date='2018-01-01', end_date='2018-12-31')
```

### 2.2 获取行业财务数据

获取行业成份股的财务数据

**调用方法**
```python
get_fundamentals(query_object, date=None, statDate=None)
```

**参数**
- query_object: 查询对象
- date: 查询日期
- statDate: 财报统计日期

**示例**
```python
from jqdata import *
from jqdata import query, valuation, indicator

# 查询银行行业的平均市盈率
q = query(valuation.pe_ratio, 
          indicator.eps
    ).filter(valuation.code.in_(get_industry_stocks('HY493')))
df = get_fundamentals(q, date='2018-06-01')
