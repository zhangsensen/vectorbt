# 股票数据接口文档

## 注意事项

- run_query函数为了防止返回数据量过大，每次最多返回条数为4000行（之前是3000行）
- query函数的更多用法详见：sqlalchemy.orm.query.Query对象
- Query的简单教程
- 数据常见疑问汇总
- 更多API的可参考官方API文档

## 1. 获取股票概况

包含股票的上市时间、退市时间、代码、名称、是否是ST等。

### 1.1 获取单支股票数据

获取单支股票的信息

**调用方法**
```python
get_security_info(code)
```

**参数**
- code: 证券代码

**返回值**
一个对象，有如下属性：
- display_name: 中文名称
- name: 缩写简称
- start_date: 上市日期, [datetime.date] 类型
- end_date: 退市日期，[datetime.date] 类型, 如果没有退市则为2200-01-01
- type: 类型，stock(股票)，index(指数)，etf(ETF基金)，fja（分级A），fjb（分级B）
- parent: 分级基金的母基金代码

**示例**
```python
# 输出平安银行信息的中文名称
get_security_info('000001.XSHE').display_name
```

### 1.2 获取所有股票数据

获取平台支持的所有股票数据

**调用方法**
```python
get_all_securities(types=['stock'], date=None)
```

**参数**
- types：默认为stock，这里请在使用时注意防止未来函数
- date: 日期, 一个字符串或者 [datetime.datetime]/[datetime.date] 对象, 用于获取某日期还在上市的股票信息. 默认值为 None, 表示获取所有日期的股票信息

**返回值**
[pandas.DataFrame], 包含以下列：
- display_name: 中文名称
- name: 缩写简称
- start_date: 上市日期
- end_date: 退市日期，如果没有退市则为2200-01-01
- type: 类型，stock(股票)

**示例**
```python
# 将所有股票列表转换成数组
stocks = list(get_all_securities(['stock']).index)

# 获得2015年10月10日还在上市的所有股票列表
get_all_securities(date='2015-10-10')
```

### 1.3 判断股票是否是ST

得到多只股票在一段时间是否是ST

**调用方法**
```python
get_extras(info, security_list, start_date='2015-01-01', end_date='2015-12-31', df=True)
```

**参数**
- info: 'is_st'，是否股改, st,*st和退市整理期标的
- security_list: 股票列表
- start_date/end_date: 开始结束日期, 同[get_price]
- df: 返回[pandas.DataFrame]对象还是一个dict

**返回值**
- df=True: [pandas.DataFrame]对象, 列索引是股票代号, 行索引是[datetime.datetime]
- df=False: 一个dict, key是股票代号, value是[numpy.ndarray]

**示例**
```python
# 返回DataFrame格式
get_extras('is_st', ['000001.XSHE', '000018.XSHE'], start_date='2013-12-01', end_date='2013-12-03')

# 返回dict格式
get_extras('is_st', ['000001.XSHE', '000018.XSHE'], start_date='2015-12-01', end_date='2015-12-03', df=False)
```

## 2. 获取股票的融资融券信息

获取一只或者多只股票在一个时间段内的融资融券信息

**调用方法**
```python
get_mtss(security_list, start_date, end_date, fields=None)
```

**参数**
- security_list: 一只股票代码或者一个股票代码的 list
- start_date: 开始日期, 一个字符串或者 datetime.datetime/datetime.date 对象
- end_date: 结束日期, 一个字符串或者 datetime.date/datetime.datetime对象
- fields: 字段名或者 list, 可选. 默认为 None, 表示取全部字段

**字段说明**
- date: 日期
- sec_code: 股票代码
- fin_value: 融资余额(元）
- fin_buy_value: 融资买入额（元）
- fin_refund_value: 融资偿还额（元）
- sec_value: 融券余量（股）
- sec_sell_value: 融券卖出量（股）
- sec_refund_value: 融券偿还股（股）
- fin_sec_value: 融资融券余额（元）

**返回值**
返回一个 pandas.DataFrame 对象，默认的列索引为取得的全部字段. 如果给定了 fields 参数, 则列索引与给定的 fields 对应.

**示例**
```python
from jqdata import *
# 获取一只股票的融资融券信息
get_mtss('000001.XSHE', '2016-01-01', '2016-04-01')

# 获取多只股票的融资融券信息
get_mtss(['000001.XSHE', '000002.XSHE', '000099.XSHE'], '2015-03-25', '2016-01-25')
```

## 3. 股票分类信息

### 3.1 获取指数成份股

获取一个指数给定日期在平台可交易的成分股列表，支持近600种股票指数数据。

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
# 获取所有沪深300的股票, 设为股票池
stocks = get_index_stocks('000300.XSHG')
set_universe(stocks)
```

### 3.2 获取行业、概念成份股

**调用方法**
```python
# 获取行业板块成分股
get_industry_stocks(industry_code, date=None)

# 获取概念板块成分股
get_concept_stocks(concept_code, date=None)
```

**参数**
- industry_code: 行业编码
- concept_code: 概念编码
- date: 查询日期, 一个字符串或者[datetime.date]/[datetime.datetime]对象, 可以是None

**返回值**
返回股票代码的list

**示例**
```python
# 获取计算机/互联网行业的成分股
stocks = get_industry_stocks('I64')

# 获取风力发电概念板块的成分股
stocks = get_concept_stocks('GN036')
```

### 3.3 查询股票所属行业

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

## 4. 获取行情数据

交易类数据提供股票的交易行情数据。

**相关接口**
- get_price: 获取历史数据，可查询多个标的多个数据字段
- history: 获取历史数据，可查询多个标的单个数据字段
- attribute_history: 获取历史数据，可查询单个标的多个数据字段
- get_bars: 获取历史数据(包含快照数据)，可查询单个或多个标的多个数据字段
- get_current_data: 获取当前逻辑时间数据(策略专用)
- get_current_tick: 获取当前逻辑时间最新的 tick 数据(策略专用)
- get_ticks: 获取股票、期货、50ETF期权、股票指数及场内基金的tick 数据
- get_call_auction: 获取指定时间区间内集合竞价时的 tick 数据

## 5. 获取融资融券标的列表

### 5.1 获取融资标的列表

**调用方法**
```python
get_margincash_stocks(date)
```

**参数**
- date: 默认为None, 不指定时返回上交所、深交所最近一次披露的可融资标的列表

**返回值**
返回指定日期上交所、深交所披露的可融资标的列表的list

**示例**
```python
# 获取融资标的列表
margincash_stocks = get_margincash_stocks(date='2018-07-02')

# 判断平安银行是否在可融资列表
'000001.XSHE' in get_margincash_stocks(date='2018-07-02')
```

### 5.2 获取融券标的列表

**调用方法**
```python
get_marginsec_stocks(date)
```

**参数**
- date: 默认为None, 不指定时返回上交所、深交所最近一次披露的可融券标的列表

**返回值**
返回指定日期上交所、深交所披露的可融券标的列表的list

**示例**
```python
# 获取融券标的列表
marginsec_stocks = get_marginsec_stocks(date='2018-07-05')

# 判断平安银行是否在可融券列表
'000001.XSHE' in get_marginsec_stocks(date='2018-07-05')
```

## 6. 获取融资融券汇总数据

**调用方法**
```python
from jqdata import *
finance.run_query(query(finance.STK_MT_TOTAL).filter(finance.STK_MT_TOTAL.date=='2019-05-23').limit(n))
```

**描述**
记录上海交易所和深圳交易所的融资融券汇总数据

**参数**
- query(finance.STK_MT_TOTAL): 表示从finance.STK_MT_TOTAL这张表中查询融资融券汇总数据
- filter(finance.STK_MT_TOTAL.date==date): 指定筛选条件
- limit(n): 限制返回的数据条数

**字段说明**
- date: 交易日期
- exchange_code: 交易市场（XSHG-上海证券交易所；XSHE-深圳证券交易所）
- fin_value: 融资余额（元）
- fin_buy_value: 融资买入额（元）
- sec_volume: 融券余量（股）
- sec_value: 融券余量金额（元）
- sec_sell_volume: 融券卖出量（股）
- fin_sec_value: 融资融券余额（元）

**注意事项**
- 为了防止返回数据量过大, 每次最多返回4000行
- 不能进行连表查询

## 7. 获取股票资金流向数据

**调用方法**
```python
from jqdata import *
get_money_flow(security_list, start_date=None, end_date=None, fields=None, count=None)
```

**参数**
- security_list: 一只股票代码或者一个股票代码的 list
- start_date: 开始日期
- end_date: 结束日期
- count: 数量, 与 start_date 二选一
- fields: 字段名或者 list, 可选

**字段说明**
- date: 日期
- sec_code: 股票代码
- change_pct: 涨跌幅(%)
- net_amount_main: 主力净额(万) - 主力净额 = 超大单净额 + 大单净额
- net_pct_main: 主力净占比(%) - 主力净占比 = 主力净额 / 成交额
- net_amount_xl: 超大单净额(万) - 超大单：大于等于50万股或者100万元的成交单
- net_pct_xl: 超大单净占比(%) - 超大单净占比 = 超大单净额 / 成交额
- net_amount_l: 大单净额(万) - 大单：大于等于10万股或者20万元且小于50万股和100万元的成交单
- net_pct_l: 大单净占比(%) - 大单净占比 = 大单净额 / 成交额
- net_amount_m: 中单净额(万) - 中单：大于等于2万股或者4万元且小于10万股和20万元的成交单
- net_pct_m: 中单净占比(%) - 中单净占比 = 中单净额 / 成交额
- net_amount_s: 小单净额(万) - 小单：小于2万股和4万元的成交单
- net_pct_s: 小单净占比(%) - 小单净占比 = 小单净额 / 成交额

**示例**
```python
# 获取一只股票在一个时间段内的资金流量数据
get_money_flow('000001.XSHE', '2016-02-01', '2016-02-04')

# 获取多只股票在一个时间段内的资金流向数据
get_money_flow(['000001.XSHE', '000040.XSHE', '000099.XSHE'], '2010-01-01', '2010-01-30')
```

## 8. 获取龙虎榜数据

**调用方法**
```python
get_billboard_list(stock_list, start_date, end_date, count)
```

**参数**
- stock_list: 一个股票代码的 list。当值为 None 时，返回指定日期的所有股票
- start_date: 开始日期
- end_date: 结束日期
- count: 交易日数量

**返回值**
pandas.DataFrame，各 column 的含义：
- code: 股票代码
- day: 日期
- direction: ALL 表示『汇总』，SELL 表示『卖』，BUY 表示『买』
- abnormal_code: 异常波动类型
- abnormal_name: 异常波动名称
- sales_depart_name: 营业部名称
- rank: 0 表示汇总，1~5 对应买入金额或卖出金额排名第一到第五
- buy_value: 买入金额
- buy_rate: 买入金额占比
- sell_value: 卖出金额
- sell_rate: 卖出金额占比
- total_value: 总额
- net_value: 净额
- amount: 市场总成交额

## 9. 上市公司分红送股（除权除息）数据

**调用方法**
```python
from jqdata import finance
finance.run_query(query(finance.STK_XR_XD).filter(finance.STK_XR_XD.code==code).order_by(finance.STK_XR_XD.report_date).limit(n))
```

**描述**
记录由上市公司年报、中报、一季报、三季报统计出的分红转增情况

**字段说明**
包含董事会预案公告日期、股东大会预案公告日期、实施方案公告日期、送股比例、转增比例、派息比例等详细字段

**注意事项**
- 为了防止返回数据量过大, 每次最多返回4000行
- 不能进行连表查询
