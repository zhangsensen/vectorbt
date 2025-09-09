# 港股分析系统日志使用说明

## 📋 日志系统概述

港股多时间框架分析系统现在配备了完整的日志记录功能，所有的分析过程和调试信息都会被详细记录到日志文件中，便于问题排查和系统监控。

## 🗂️ 日志文件位置

日志文件存储在以下位置：
```
/Users/zhangshenshen/longport/vectorbt_workspace/multi_timezone_radar/core/logs/
```

日志文件命名格式：
```
hk_analysis_YYYYMMDD_HHMMSS.log
```

例如：`hk_analysis_20250908_000140.log`

## 📊 日志级别

系统使用以下日志级别：

- **DEBUG** (调试): 详细的调试信息，包括因子计算过程、数据格式检查等
- **INFO** (信息): 一般信息，如分析开始、时间框架处理、稳健因子识别等
- **WARNING** (警告): 警告信息，如数据不足、IC值异常等
- **ERROR** (错误): 错误信息，如分析失败、数据处理错误等

## 🔧 日志内容包含

### 1. 主程序日志 (hk_comprehensive_analysis.py)
- 分析开始和结束时间
- 各时间框架处理进度
- 数据加载情况
- 因子计算统计
- 结果输出信息

### 2. OOS测试日志 (out_of_sample_tester_fixed.py)
- 样本内/样本外测试过程
- 因子调试信息（DEBUG级别）
- IC值计算详情
- 稳健因子识别结果
- 衰减分析信息

## 📝 典型日志条目示例

```
2025-09-08 00:01:40,427 - __main__ - INFO - 开始港股全时间框架分析...
2025-09-08 00:01:40,428 - __main__ - INFO - Processing 1m timeframe...
2025-09-08 00:01:40,925 - __main__ - INFO - Loaded 54 stocks for 1m timeframe
2025-09-08 00:01:41,123 - out_of_sample_tester_fixed - INFO - 🧪 开始OOS测试...
2025-09-08 00:01:41,123 - out_of_sample_tester_fixed - INFO -    因子数量: 10
2025-09-08 00:01:41,123 - out_of_sample_tester_fixed - INFO -    数据股票数: 54
2025-09-08 00:01:41,456 - out_of_sample_tester_fixed - DEBUG -    🔍 调试 RSI:
2025-09-08 00:01:41,456 - out_of_sample_tester_fixed - DEBUG -       因子类型: <class 'pandas.core.series.Series'>
2025-09-08 00:01:41,456 - out_of_sample_tester_fixed - DEBUG -       因子形状: (2168119,)
2025-09-08 00:01:42,789 - out_of_sample_tester_fixed - INFO -    ✅ RSI: IC=-0.0906, 股票数=37
```

## 🔍 问题排查指南

### 1. 查找最新日志文件
```bash
ls -la /Users/zhangshenshen/longport/vectorbt_workspace/multi_timezone_radar/core/logs/ | tail -5
```

### 2. 实时查看日志
```bash
tail -f /Users/zhangshenshen/longport/vectorbt_workspace/multi_timezone_radar/core/logs/hk_analysis_*.log
```

### 3. 搜索特定错误
```bash
grep -i "error" /Users/zhangshenshen/longport/vectorbt_workspace/multi_timezone_radar/core/logs/hk_analysis_*.log
```

### 4. 查看特定时间框架的分析
```bash
grep "1m timeframe" /Users/zhangshenshen/longport/vectorbt_workspace/multi_timezone_radar/core/logs/hk_analysis_*.log
```

## 🚨 常见问题及解决方案

### 1. 日志文件未创建
- **原因**: 权限问题或路径错误
- **解决**: 检查logs目录权限和路径

### 2. 调试信息缺失
- **原因**: 日志级别设置过高
- **解决**: 确保日志级别设置为DEBUG

### 3. OOS测试信息缺失
- **原因**: OutOfSampleTester日志配置问题
- **解决**: 检查out_of_sample_tester_fixed.py中的logger配置

## 📈 日志分析建议

1. **性能监控**: 观察各时间框架的处理时间
2. **数据质量**: 检查数据加载和因子计算的警告信息
3. **因子表现**: 分析稳健因子的识别结果
4. **错误追踪**: 关注ERROR级别的日志信息

## 🔄 日志清理

建议定期清理旧的日志文件以节省空间：
```bash
find /Users/zhangshenshen/longport/vectorbt_workspace/multi_timezone_radar/core/logs/ -name "*.log" -mtime +30 -delete
```

---

**注意**: 所有日志信息都会同时输出到控制台和日志文件，方便实时监控和后续分析。