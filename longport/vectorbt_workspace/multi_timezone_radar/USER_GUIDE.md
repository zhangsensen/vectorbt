# 📚 用户使用指南

## 🚀 快速开始

### 1. 环境准备

#### 安装依赖
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 配置文件
创建 `config.py` 文件：
```python
# config.py
LONGPORT_CONFIG = {
    'APP_KEY': 'your_app_key',
    'APP_SECRET': 'your_app_secret',
    'ACCESS_TOKEN': 'your_access_token'
}

# 系统配置
SYSTEM_CONFIG = {
    'data_dir': '/path/to/your/data',
    'test_ratio': 0.3,
    'transaction_cost': 0.001,
    'n_workers': 4,
    'memory_limit_gb': 4.0
}
```

### 2. 数据准备

#### 下载数据
```bash
# 运行数据下载器
python core/data_download/batch_downloader.py
```

#### 数据结构
```
data/
├── 1m/           # 1分钟数据
├── 5m/           # 5分钟数据
├── 15m/          # 15分钟数据
├── 30m/          # 30分钟数据
├── 1h/           # 1小时数据
└── 1d/           # 日线数据
    ├── 0005.HK.parquet
    ├── 0700.HK.parquet
    └── ...
```

### 3. 运行分析

#### 基本使用
```python
from core.hk_comprehensive_analysis import HKComprehensiveAnalysis
from core.out_of_sample_tester_fixed import OutOfSampleTester

# 1. 创建分析器
analyzer = HKComprehensiveAnalysis('/path/to/data')

# 2. 加载数据
data = analyzer.load_all_stocks_data()

# 3. 计算因子
factors = analyzer.calculate_technical_factors(data)

# 4. 运行测试
tester = OutOfSampleTester(data, test_ratio=0.3)
results = tester.test_factors(factors)

# 5. 查看结果
print("稳健因子:", results['robust_factors'])
```

#### 高级使用
```python
# 自定义配置
tester = OutOfSampleTester(
    data=data,
    test_ratio=0.3,
    transaction_cost=0.001,
    timeframe='1d',
    n_workers=8,
    memory_limit_gb=8.0
)

# 运行完整分析
results = tester.test_factors(factors)

# 分析详细结果
for factor_name, metrics in results['oos_metrics'].items():
    print(f"{factor_name}:")
    print(f"  IC: {metrics['ic']:.4f}")
    print(f"  Rank IC: {metrics['rank_ic']:.4f}")
    print(f"  Cost Adj IC: {metrics['cost_adjusted_ic']:.4f}")
```

## 📊 结果解读

### 1. 基础指标

#### 信息系数 (IC)
- **Raw IC**: 原始信息系数，衡量因子预测能力
- **Rank IC**: 排序信息系数，更稳健的预测能力衡量
- **Multi-period IC**: 多期信息系数，衡量长期预测能力
- **Cost-adjusted IC**: 成本调整后信息系数，考虑交易成本的影响

**解读标准**:
- `|IC| > 0.05`: 有预测能力
- `|IC| > 0.1`: 较强预测能力
- `|IC| > 0.2`: 很强预测能力

#### 稳健性评估
- **Decay Analysis**: 衰减分析
  - `mild`: 轻度衰减 (< 0.3)
  - `moderate`: 中度衰减 (0.3-0.6)
  - `severe`: 重度衰减 (> 0.6)

### 2. 多重检验结果

#### Bonferroni校正
- 严格的校正方法，控制假阳性率
- 适用于因子数量较少的情况

#### FDR校正
- 平衡假阳性控制和统计功效
- 适用于因子数量较多的情况

**解读标准**:
- `significant_after_correction = True`: 通过校正
- `corrected_p_value < 0.05`: 统计显著

### 3. 换手率分析

#### 换手率 (Turnover Rate)
- 衡量因子信号的稳定性
- 低换手率 = 稳定信号
- 高换手率 = 频繁变化

#### 自相关 (Autocorrelation)
- 衡量因子持续性
- 高自相关 = 信号持久
- 低自相关 = 信号变化快

#### 半衰期 (Half Life)
- 信号衰减到一半所需的时间
- 长半衰期 = 信号持久
- 短半衰期 = 信号快速衰减

## 🎯 最佳实践

### 1. 因子选择

#### 选择标准
1. **统计显著性**: 通过多重检验校正
2. **经济意义**: 有合理的金融逻辑
3. **稳定性**: 样本内外表现一致
4. **实用性**: 考虑交易成本和换手率

#### 避免陷阱
- **过拟合**: 避免在样本内过度优化
- **数据窥探**: 避免多次测试相同数据
- **幸存者偏差**: 确保包含退市股票
- **未来函数**: 避免使用未来信息

### 2. 参数调优

#### 测试比例
```python
# 推荐测试比例
test_ratios = [0.2, 0.3, 0.4]
for ratio in test_ratios:
    tester = OutOfSampleTester(data, test_ratio=ratio)
    results = tester.test_factors(factors)
    print(f"测试比例 {ratio}: 稳健因子 {len(results['robust_factors'])}")
```

#### 交易成本
```python
# 不同交易成本下的表现
costs = [0.0005, 0.001, 0.002, 0.005]
for cost in costs:
    tester = OutOfSampleTester(data, transaction_cost=cost)
    results = tester.test_factors(factors)
    print(f"交易成本 {cost}: 稳健因子 {len(results['robust_factors'])}")
```

### 3. 风险管理

#### 因子监控
```python
# 定期重新验证因子稳健性
def monitor_factors(data, factors, monitoring_period=30):
    tester = OutOfSampleTester(data, test_ratio=0.3)
    results = tester.test_factors(factors)
    
    # 检查因子衰减
    for factor_name in results['robust_factors']:
        decay = results['decay_analysis'].get(factor_name, {})
        if decay.get('decay_level') == 'severe':
            print(f"警告: {factor_name} 衰减严重!")
    
    return results
```

#### 集中度风险
```python
# 避免过度依赖少数因子
def check_concentration(results):
    robust_factors = results['robust_factors']
    if len(robust_factors) < 3:
        print("警告: 稳健因子数量过少!")
    
    # 检查因子相关性
    # TODO: 实现因子相关性分析
```

## 🔧 常见问题

### 1. 数据问题

#### Q: 数据下载失败
```bash
# 检查网络连接
ping open.longportapp.com

# 检查API配置
python -c "from config import LONGPORT_CONFIG; print(LONGPORT_CONFIG)"
```

#### Q: 数据格式错误
```python
# 检查数据文件
import pandas as pd
df = pd.read_parquet('data/1d/0700.HK.parquet')
print(df.columns)  # 应该包含 ['open', 'high', 'low', 'close', 'volume']
print(df.index.dtype)  # 应该是 datetime64[ns]
```

### 2. 计算问题

#### Q: 内存不足
```python
# 减少数据量
tester = OutOfSampleTester(
    data=data,
    memory_limit_gb=2.0,  # 降低内存限制
    n_workers=2  # 减少并行进程
)

# 或者分批处理
symbols = list(data.keys())
for i in range(0, len(symbols), 10):
    batch_data = {s: data[s] for s in symbols[i:i+10]}
    tester = OutOfSampleTester(batch_data)
    results = tester.test_factors(factors)
```

#### Q: 计算时间过长
```python
# 增加并行进程
tester = OutOfSampleTester(
    data=data,
    n_workers=8  # 增加工作进程数
)

# 或者减少时间范围
# 筛选最近3个月的数据
```

### 3. 结果问题

#### Q: 没有稳健因子
```python
# 检查数据质量
print(f"股票数量: {len(data)}")
for symbol, df in data.items():
    print(f"{symbol}: {len(df)} 条记录")

# 检查因子计算
factors = analyzer.calculate_technical_factors(data)
print(f"因子数量: {len(factors)}")

# 放宽标准
# 修改阈值或调整参数
```

#### Q: 因子表现不一致
```python
# 检查样本内外一致性
for factor_name in results['oos_metrics'].keys():
    is_ic = results['is_metrics'][factor_name]['ic']
    oos_ic = results['oos_metrics'][factor_name]['ic']
    decay = abs(is_ic - oos_ic)
    print(f"{factor_name}: IC衰减 {decay:.4f}")
```

## 📈 进阶使用

### 1. 自定义因子

#### 添加新因子
```python
def calculate_custom_factor(df):
    """
    计算自定义因子
    
    Args:
        df: 价格数据DataFrame
        
    Returns:
        因子值序列
    """
    # 示例: 价格动量因子
    momentum = df['close'].pct_change(10)
    return momentum

# 集成到系统中
class CustomAnalysis(HKComprehensiveAnalysis):
    def calculate_technical_factors(self, data):
        factors = super().calculate_technical_factors(data)
        
        # 添加自定义因子
        custom_factors = {}
        for symbol, df in data.items():
            custom_factor = calculate_custom_factor(df)
            # ... 处理因子格式 ...
        
        factors.update(custom_factors)
        return factors
```

### 2. 多资产分析

#### 同时分析港股和美股
```python
# 加载不同市场数据
hk_data = analyzer.load_market_data('HK')
us_data = analyzer.load_market_data('US')

# 分别计算因子
hk_factors = analyzer.calculate_technical_factors(hk_data)
us_factors = analyzer.calculate_technical_factors(us_data)

# 合并分析
combined_data = {**hk_data, **us_data}
combined_factors = {**hk_factors, **us_factors}

# 运行测试
results = tester.test_factors(combined_factors)
```

### 3. 实时监控

#### 创建监控脚本
```python
import schedule
import time

def daily_analysis():
    """每日分析任务"""
    print("开始每日分析...")
    
    # 加载数据
    data = analyzer.load_all_stocks_data()
    
    # 计算因子
    factors = analyzer.calculate_technical_factors(data)
    
    # 运行测试
    tester = OutOfSampleTester(data)
    results = tester.test_factors(factors)
    
    # 生成报告
    generate_report(results)
    
    # 发送警报
    send_alerts(results)

# 设置定时任务
schedule.every().day.at("16:00").do(daily_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 📚 参考资料

### 1. 相关文献
- "Active Portfolio Management" by Grinold and Kahn
- "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- "Quantitative Trading: How to Build Your Own Algorithmic Trading Business" by Ernie Chan

### 2. 技术文档
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [TA-Lib Documentation](https://mrjbq7.github.io/ta-lib/)

### 3. 在线资源
- [LongPort OpenAPI](https://open.longportapp.com/)
- [QuantConnect Documentation](https://www.quantconnect.com/docs/)
- [Kaggle Quant Finance](https://www.kaggle.com/learn/finance)

---

## 📞 技术支持

### 获取帮助
- **GitHub Issues**: [项目Issues页面](https://github.com/your-repo/issues)
- **文档**: 查看 `docs/` 目录下的详细文档
- **示例**: 参考 `examples/` 目录下的使用案例

### 报告问题
如果您遇到问题，请提供以下信息：
1. 操作系统和Python版本
2. 错误信息和堆栈跟踪
3. 数据样本和配置参数
4. 重现步骤

### 功能请求
欢迎提交新功能请求，请包含：
1. 功能描述和用途
2. 预期行为
3. 使用场景示例

---

**📝 最后更新**: 2024年1月  
**🎯 文档目的**: 帮助用户快速上手和使用系统  
**📖 适用版本**: v2.0.0+