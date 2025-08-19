# 指数数据接口文档

## 1. 获取指数基础信息

### 1.1 获取指数列表

获取所有指数的基础信息

**调用方法**
```python
get_all_securities(types=['index'], date=None)
```

**参数**
- types：默认为index，表示获取指数数据
- date: 日期, 一个字符串或者 [datetime.datetime]/[datetime.date] 对象

**返回值**
[pandas.DataFrame], 包含以下列：
- display_name: 中文名称
- name: 缩写简称
- start_date: 上市日期
- end_date: 退市日期
- type: 类型，index(指数)

**示例**
```python
# 获取所有指数列表
get_all_securities(types=['index'])

# 获取指定日期的指数列表
get_all_securities(types=['index'], date='2018-06-01')
```

### 1.2 获取指数成份股

获取一个指数给定日期在平台可交易的成分股列表

**调用方法**
```python
get_index_stocks(index_symbol, date=None)
```

**参数**
- index_symbol: 指数代码
- date: 查询日期, 一个字符串或者[datetime.date]/[datetime.datetime]对象, 可以是None

**返回值**
返回股票代码的list

**示例**
```python
# 获取沪深300指数的成分股
stocks = get_index_stocks('000300.XSHG')

# 获取指定日期的上证50成分股
stocks = get_index_stocks('000016.XSHG', date='2018-06-01')
```

## 2. 获取指数行情数据

### 2.1 获取指数历史行情

获取指数的历史价格数据

**调用方法**
```python
get_price(security, start_date, end_date, frequency='1d', fields=None, skip_paused=False, fq='pre')
```

**参数**
- security: 指数代码，格式如'000001.XSHG'
- start_date: 开始日期
- end_date: 结束日期
- frequency: 数据频率，默认为'1d'
- fields: 字段名或者 list, 可选
- skip_paused: 是否跳过停牌日期
- fq: 复权方式

**返回值**
[pandas.DataFrame]格式的历史数据

**示例**
```python
# 获取上证指数的历史数据
get_price('000001.XSHG', start_date='2018-01-01', end_date='2018-12-31')

# 获取特定字段的指数数据
get_price('000300.XSHG', start_date='2018-01-01', end_date='2018-12-31', 
          fields=['open', 'close', 'high', 'low', 'volume', 'money'])
```

### 2.2 获取指数实时行情

获取指数的当前行情数据

**调用方法**
```python
get_current_data()
```

**返回值**
返回当前所有证券的行情数据

**示例**
```python
# 获取当前指数行情
current_data = get_current_data()
# 获取上证指数的当前价格
sz_price = current_data['000001.XSHG'].last_price
```

## 3. 指数财务数据

### 3.1 获取指数成份股财务数据

获取指数成份股的财务数据

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

# 查询沪深300成份股的财务数据
q = query(valuation, indicator).filter(valuation.code.in_(get_index_stocks('000300.XSHG')))
df = get_fundamentals(q, date='2018-06-01')
```

## 4. 指数衍生数据

### 4.1 获取指数技术指标

基于指数价格数据计算技术指标

**示例**
```python
import talib
import numpy as np

# 获取指数价格数据
price_data = get_price('000001.XSHG', start_date='2018-01-01', end_date='2018-12-31', 
                       fields=['close'], frequency='1d')

# 计算MACD指标
close_prices = price_data['close'].values
macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)

# 计算RSI指标
rsi = talib.RSI(close_prices, timeperiod=14)
```

### 4.2 获取指数资金流向

获取指数成份股的资金流向数据

**调用方法**
```python
get_money_flow(security_list, start_date=None, end_date=None, fields=None, count=None)
```

**示例**
```python
# 获取沪深300成份股的资金流向
stocks = get_index_stocks('000300.XSHG')
money_flow_data = get_money_flow(stocks, start_date='2018-01-01', end_date='2018-01-31')
