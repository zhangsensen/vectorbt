# 用户详细教程 (Detailed User Tutorial)

## 🎯 系统简介

这个港股智能分析系统可以帮助您：
- 自动分析54只港股的技术指标
- 为每只股票给出投资评分
- 提供买卖建议
- 生成详细的分析报告

## 🚀 完整使用流程

### 准备阶段

#### 1. 确保环境就绪
```bash
# 检查Python版本 (需要3.8或更高)
python --version

# 如果没有Python，请先安装Python
```

#### 2. 安装必要的软件包
```bash
# 进入项目目录
cd /Users/zhangshenshen/longport/vectorized_factor_analyzer_v2

# 安装依赖
pip install -r requirements.txt
```

#### 3. 准备数据文件
将您的港股数据文件放入 `data/` 文件夹中。数据文件应该包含：
- **必须的列**: open, high, low, close, volume
- **支持的格式**: .parquet, .csv
- **建议的时间周期**: 1分钟, 5分钟, 15分钟, 30分钟, 1小时, 4小时, 1天

### 运行分析

#### 方式一：简单运行 (推荐新手)
```bash
# 复制这段代码到终端运行
python -c "
import sys
sys.path.append('.')
from strategies.cta_eval_v3 import CTA_Evaluator_V3

print('正在启动港股分析系统...')
analyzer = CTA_Evaluator_V3()
results = analyzer.run_full_analysis()
print('分析完成！')
print(f'总共分析了 {len(results)} 个股票因子组合')
"
```

#### 方式二：自定义运行
```bash
# 创建一个简单的运行脚本
echo "
import sys
sys.path.append('.')
from strategies.cta_eval_v3 import CTA_Evaluator_V3

# 创建分析器
analyzer = CTA_Evaluator_V3()

# 运行分析并保存结果
results = analyzer.run_full_analysis()

# 显示前10个最佳结果
print('前10个最佳投资机会：')
for i, result in enumerate(list(results)[:10]):
    print(f'{i+1}. {result[0]} - {result[1]} - 评分: {result[2]:.2f}')
" > run_analysis.py

# 运行脚本
python run_analysis.py
```

### 查看和理解结果

#### 结果文件位置
- **主要结果**: `results/` 文件夹
- **运行日志**: `logs/` 文件夹
- **分析报告**: `.json` 和 `.md` 文件

#### 如何解读评分
```
90-100分: 强烈推荐买入 ⭐⭐⭐⭐⭐
80-89分:  推荐买入       ⭐⭐⭐⭐
70-79分:  观望           ⭐⭐⭐
60-69分:  谨慎           ⭐⭐
0-59分:   不推荐         ⭐
```

#### 技术指标解读

**RSI指标**
- 高于70: 股票可能超买，考虑卖出
- 低于30: 股票可能超卖，考虑买入

**MACD指标**
- 金叉: 买入信号
- 死叉: 卖出信号

**价格位置**
- 80%以上: 价格处于高位
- 20%以下: 价格处于低位

## 🛠️ 常见问题解决

### 安装问题
**问题**: `pip install` 失败
**解决**: 
```bash
# 尝试升级pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 内存问题
**问题**: 程序运行时内存不足
**解决**:
1. 关闭其他占用内存的程序
2. 减少分析的股票数量
3. 增加虚拟内存

### 数据问题
**问题**: 数据文件格式不正确
**解决**:
```python
# 检查数据文件的列名
import pandas as pd
df = pd.read_parquet('data/your_file.parquet')
print(df.columns)
```

## 📊 高级功能

### 自定义分析参数
```python
# 创建分析器时可以传入自定义参数
analyzer = CTA_Evaluator_V3(
    timeframes=['1d', '4h', '1h'],  # 只分析这些时间周期
    stock_symbols=['700.HK', '9988.HK']  # 只分析这些股票
)
```

### 生成自定义报告
```python
# 运行分析后生成详细报告
report = analyzer.generate_report(results)
print(report)
```

## 🎯 投资建议

### 如何使用系统结果
1. **关注高分股票**: 优先考虑80分以上的股票
2. **结合多个时间周期**: 同一股票在多个周期都得分较高更可靠
3. **注意风险**: 不要把所有资金投入一只股票
4. **定期更新**: 建议每周或每月重新运行分析

### 风险提醒
- 过去表现不代表未来收益
- 技术分析仅供参考，基本面分析同样重要
- 投资有风险，请根据自己的风险承受能力决策

## 📞 获取更多帮助

如果还有问题：
1. 查看 `logs/` 文件夹中的错误日志
2. 检查数据文件格式
3. 确认所有依赖包已正确安装
4. 参考主README文件的详细说明

祝您投资顺利！🚀