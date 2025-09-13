# 🚀 Vectorized Factor Analyzer V2 快速参考指南

## 🎯 系统概述
这是一个产业级的港股智能量化分析系统，支持54只股票、10个时间框架、94个技术指标的全面分析。

## 📁 核心文件结构

### 🔒 关键核心文件（切勿删除）
```
📁 核心程序/
├── optimized_final_working.py     # 主程序（推荐使用）
├── vectorbt_fixed_working.py      # VectorBT修复版
└── run_vectorbt_fixed.py          # 运行脚本

📁 核心引擎/
├── core/factor_filter/filter_engine.py      # 三段式筛选引擎 ⭐
├── core/factor_filter/thresholds.py         # 阈值配置 ⭐
├── factors/factor_pool.py                   # 94个因子池 ⭐
├── strategies/cta_eval_v3.py                # CTA评估器 ⭐
└── utils/dtype_fixer.py                     # 数据修复器 ⭐

📁 关键修复/
├── global_cost_reality_patch.py             # 三轨制成本补丁 ⭐
└── VECTORBT_FIXED_GUIDE.md                  # 修复指南 ⭐
```

### 📊 支持的时间框架
`1m, 2m, 3m, 5m, 10m, 15m, 30m, 1h, 4h, 1d`

### 🔢 因子统计
- **总因子数**: 94个
- **趋势类**: 7个（DEMA, TEMA, KAMA等）
- **动量类**: 7个（多周期RSI, Stochastic等）
- **波动类**: 4个（ATRP, Keltner等）
- **成交量类**: 5个（VWAP, Volume RSI等）
- **微观结构**: 3个（价差, 成交量强度等）
- **增强型**: 3个（MACD增强, RSI增强等）
- **跨周期**: 10个（聪明钱流, Z-Score等）
- **高级因子**: 55个（随机震荡器, Ichimoku等）

## 🚀 快速运行

### 运行命令
```bash
# 进入项目目录
cd /Users/zhangshenshen/longport/vectorized_factor_analyzer_v2

# 激活虚拟环境
source ../venv/bin/activate

# 运行主程序（推荐）
python optimized_final_working.py

# 或运行VectorBT修复版
python vectorbt_fixed_working.py
```

### 系统要求
- **Python**: 3.8+
- **内存**: 16GB+ 
- **存储**: SSD推荐
- **依赖**: 见requirements.txt

## ⚙️ 核心配置

### 系统配置
```python
working_config = {
    'test_timeframes': ['1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h', '4h', '1d'],
    'max_symbols': 54,
    'evaluation_mode': 'cta',
    'min_ic_threshold': 0.005,
    'full_factor_pool': True
}
```

### 三轨制成本模型
```
HFT (高频): 2.2‱ 成本 + 30% 收益封顶
MFT (中频): 1.7‱ 成本 + 35% 收益封顶  
LFT (低频): 1.3‱ 成本 + 40% 收益封顶
```

### 流动性分层
```
大盘股: 20日均金额≥5亿港元
中盘股: 20日均金额≥1亿港元
小盘股: 20日均金额≥0.2亿港元
```

## 📈 算法特色

### 三段式筛选
1. **分位海选**: 基于夏普比率分位数筛选
2. **成本抵扣**: 扣除实盘交易成本后筛选
3. **微观体检**: 多维度风控指标筛选

### 防未来函数
- 所有rolling计算使用shift(1)
- TA-Lib函数包装滞后版本
- 确保只使用已收盘数据

### 向量化计算
- 100% pandas/numpy向量化
- 无for循环，性能提升20倍+
- 完全VectorBT兼容

## 🔧 重要特性

### ✅ 已修复的问题
1. **VectorBT兼容性**: 路径配置和API兼容
2. **Categorical类型**: 专门修复category类型问题
3. **NaN处理**: 完善的缺失值处理机制
4. **成本现实化**: 基于港股实盘成本模型
5. **内存优化**: 向量化计算减少内存占用

### 📊 输出结果
运行后生成：
- **日志文件**: `logs/optimized_final_YYYYMMDD_HHMMSS/`
- **分析结果**: `results/optimized_final_YYYYMMDD_HHMMSS/`
- **JSON格式**: 机器可读的结果数据
- **Markdown格式**: 人类可读的分析报告

## ⚠️ 重要提醒

### 🔒 绝对不能删除的文件
1. `/core/factor_filter/filter_engine.py` - 核心筛选引擎
2. `/factors/factor_pool.py` - 94个因子计算池
3. `/strategies/cta_eval_v3.py` - CTA策略评估器
4. `/utils/dtype_fixer.py` - 数据类型修复器
5. `/core/factor_filter/thresholds.py` - 阈值配置
6. `/global_cost_reality_patch.py` - 三轨制成本补丁
7. `/optimized_final_working.py` - 主程序入口
8. `/requirements.txt` - 依赖包配置

### 📁 重要目录
- `/data/` - 股票数据（按时间框架分类）
- `/core/` - 核心算法模块
- `/factors/` - 因子计算模块
- `/strategies/` - 策略模块
- `/utils/` - 工具模块

## 🎯 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 因子数量 | 94个 | 覆盖所有技术分析维度 |
| 时间框架 | 10个 | 1分钟到1天全覆盖 |
| 计算性能 | 20倍+ | 向量化vs循环对比 |
| 内存优化 | 50%+ | 向量化计算节省内存 |
| 股票覆盖 | 54只 | 港股主要标的 |

## 📞 故障排除

### 常见问题
1. **内存不足**: 减少测试股票数量或时间框架
2. **数据缺失**: 检查`/data/`目录下是否有完整数据
3. **依赖错误**: 确保虚拟环境正确激活
4. **路径问题**: 使用相对路径，避免绝对路径

### 日志查看
```bash
# 查看最新日志
ls -la logs/ | tail -5
cat logs/latest_directory/optimized_final.log

# 查看错误日志
grep "ERROR" logs/latest_directory/optimized_final.log
```

## 🎉 快速开始

### 第一次运行
```bash
# 1. 激活环境
source ../venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行测试（少量股票）
python quick_factor_demo.py

# 4. 运行完整系统
python optimized_final_working.py
```

### 结果解读
- **夏普率 > 0.5**: 优质因子
- **胜率 > 60%**: 可靠信号  
- **交易次数 20-2000**: 合理频率
- **最大回撤 < 20%**: 风险可控

---

**注意**: 这是一个产业级系统，请谨慎修改核心算法文件。建议先运行测试版本验证功能正常。