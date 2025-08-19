# 技术分析指标接口文档

## 1. 聚宽内置技术指标

### 1.1 获取技术指标数据

聚宽平台集成了常用的技术分析指标，可以直接通过get_price函数获取

**调用方法**
```python
get_price(security, start_date, end_date, frequency='1d', fields=None, skip_paused=False, fq='pre')
```

**常用技术指标字段**
- ma5, ma10, ma20, ma30, ma60: 均线指标
- ema5, ema10, ema20, ema30, ema60: 指数均线
- rsi6, rsi12, rsi24: 相对强弱指标
- macd_dif, macd_dea, macd_histogram: MACD指标
- boll_upper, boll_mid, boll_lower: 布林带指标
- kdj_k, kdj_d, kdj_j: KDJ指标
- wr6, wr10: 威廉指标
- bias6, bias12, bias24: 乖离率指标
- cci: 商品通道指数
- atr: 平均真实波幅
- dma_dif, dma_ama: 平行线差指标

**示例**
```python
# 获取股票的均线数据
get_price('000001.XSHE', start_date='2018-01-01', end_date='2018-12-31', 
          fields=['close', 'ma5', 'ma10', 'ma20'])

# 获取MACD指标
get_price('000001.XSHE', start_date='2018-01-01', end_date='2018-12-31', 
          fields=['close', 'macd_dif', 'macd_dea', 'macd_histogram'])
```

## 2. 使用TA-Lib计算技术指标

### 2.1 安装和导入

```python
import talib
import numpy as np
```

### 2.2 趋势类指标

#### 移动平均线 (MA)
```python
# 简单移动平均线
ma5 = talib.SMA(close_prices, timeperiod=5)
ma10 = talib.SMA(close_prices, timeperiod=10)

# 指数移动平均线
ema12 = talib.EMA(close_prices, timeperiod=12)
ema26 = talib.EMA(close_prices, timeperiod=26)

# 加权移动平均线
wma30 = talib.WMA(close_prices, timeperiod=30)
```

#### MACD指标
```python
macd, macd_signal, macd_hist = talib.MACD(close_prices, 
                                          fastperiod=12, 
                                          slowperiod=26, 
                                          signalperiod=9)
```

#### 布林带 (BOLL)
```python
upper, middle, lower = talib.BBANDS(close_prices, 
                                    timeperiod=20, 
                                    nbdevup=2, 
                                    nbdevdn=2, 
                                    matype=0)
```

### 2.3 震荡类指标

#### RSI相对强弱指标
```python
rsi6 = talib.RSI(close_prices, timeperiod=6)
rsi12 = talib.RSI(close_prices, timeperiod=12)
rsi24 = talib.RSI(close_prices, timeperiod=24)
```

#### KDJ随机指标
```python
# KDJ需要使用STOCH函数计算
slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices,
                           fastk_period=9,
                           slowk_period=3,
                           slowk_matype=0,
                           slowd_period=3,
                           slowd_matype=0)
# J = 3*K - 2*D
slowj = 3 * slowk - 2 * slowd
```

#### 威廉指标 (WR)
```python
wr10 = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=10)
wr14 = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
```

### 2.4 成交量类指标

#### 成交量移动平均线 (VMA)
```python
vma5 = talib.SMA(volume_prices, timeperiod=5)
vma10 = talib.SMA(volume_prices, timeperiod=10)
```

#### 量比 (Volume Ratio)
```python
# 需要手动计算
volume_ma = talib.SMA(volume_prices, timeperiod=5)
volume_ratio = volume_prices / volume_ma
```

#### OBV能量潮指标
```python
obv = talib.OBV(close_prices, volume_prices)
```

### 2.5 其他常用指标

#### CCI顺势指标
```python
cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
```

#### ATR平均真实波幅
```python
atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
```

#### ADX趋向指标
```python
adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
```

## 3. 自定义技术指标计算

### 3.1 基础指标计算

```python
import pandas as pd
import numpy as np

# 获取基础数据
data = get_price('000001.XSHE', start_date='2018-01-01', end_date='2018-12-31', 
                 fields=['open', 'high', 'low', 'close', 'volume'])

# 计算涨跌幅
data['change_pct'] = (data['close'] / data['close'].shift(1) - 1) * 100

# 计算振幅
data['amplitude'] = (data['high'] - data['low']) / data['close'].shift(1) * 100

# 计算换手率（需要流通股本数据）
# turnover_rate = volume / circulating_capital * 100
```

### 3.2 复合指标计算

```python
# 计算乖离率 (BIAS)
def calculate_bias(close_prices, ma_period):
    ma = talib.SMA(close_prices, timeperiod=ma_period)
    bias = (close_prices - ma) / ma * 100
    return bias

bias6 = calculate_bias(data['close'].values, 6)
bias12 = calculate_bias(data['close'].values, 12)
bias24 = calculate_bias(data['close'].values, 24)

# 计算动量指标 (Momentum)
def calculate_momentum(close_prices, period):
    momentum = close_prices / np.roll(close_prices, period) * 100
    return momentum

momentum5 = calculate_momentum(data['close'].values, 5)
momentum10 = calculate_momentum(data['close'].values, 10)
```

## 4. 技术指标使用示例

### 4.1 多指标组合策略

```python
# 获取数据
security = '000001.XSHE'
data = get_price(security, start_date='2018-01-01', end_date='2018-12-31', 
                 fields=['open', 'high', 'low', 'close', 'volume'])

# 计算技术指标
close_prices = data['close'].values
high_prices = data['high'].values
low_prices = data['low'].values

# MACD指标
macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)

# RSI指标
rsi = talib.RSI(close_prices, timeperiod=14)

# 布林带
upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)

# 判断信号
data['macd_signal'] = np.where((macd > signal) & (macd.shift(1) <= signal.shift(1)), 1, 0)
data['rsi_signal'] = np.where((rsi < 30) & (rsi.shift(1) >= 30), 1, 0)
data['boll_signal'] = np.where(close_prices < lower, 1, 0)
```

### 4.2 技术指标可视化

```python
import matplotlib.pyplot as plt

# 绘制价格和技术指标
plt.figure(figsize=(12, 8))

# 价格和均线
plt.subplot(2, 1, 1)
plt.plot(data['close'], label='Close Price')
plt.plot(data['ma5'], label='MA5')
plt.plot(data['ma20'], label='MA20')
plt.legend()

# RSI指标
plt.subplot(2, 1, 2)
plt.plot(rsi, label='RSI')
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=30, color='g', linestyle='--')
plt.legend()

plt.show()
```

## 5. 注意事项

1. **数据质量**：确保输入数据的质量，避免缺失值和异常值影响计算结果
2. **参数选择**：根据不同的市场环境和交易品种选择合适的参数
3. **滞后性**：大部分技术指标都存在滞后性，需要结合其他分析方法
4. **过度拟合**：避免过度优化参数，导致在历史数据上表现良好但在实盘中失效
5. **计算效率**：对于大量数据的计算，考虑使用向量化操作提高效率
