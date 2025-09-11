# 快速开始指南 (Quick Start Guide)

## 🎯 三步运行系统

### 第1步：进入项目目录
```bash
cd /Users/zhangshenshen/longport/vectorized_factor_analyzer_v2
```

### 第2步：安装依赖 (如果还没安装)
```bash
pip install -r requirements.txt
```

### 第3步：运行分析
复制以下命令到终端运行：
```bash
python -c "
import sys
sys.path.append('.')
from strategies.cta_eval_v3 import CTA_Evaluator_V3
import pandas as pd

print('开始分析港股...')
analyzer = CTA_Evaluator_V3()
results = analyzer.run_full_analysis()
print(f'分析完成！共分析了 {len(results)} 个组合')
"
```

## 📊 查看结果

分析完成后：
- **结果文件**: 查看 `results/` 文件夹
- **运行日志**: 查看 `logs/` 文件夹
- **最佳股票**: 评分最高的股票推荐

## 🔧 如果遇到问题

### 问题1：提示 "No module named 'xxx'"
**解决**: 运行 `pip install xxx` 安装缺失的模块

### 问题2：内存不足
**解决**: 
1. 关闭其他程序释放内存
2. 或者减少分析的股票数量

### 问题3：数据文件格式错误
**解决**: 确保数据文件包含必要的列：open, high, low, close, volume

## 📞 需要帮助？

查看主README文件获取更详细的使用说明。