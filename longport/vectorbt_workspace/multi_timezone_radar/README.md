# 港股全时间框架全因子分析系统

## 🚀 项目概述

这是一个专业的港股全时间框架技术因子分析系统，支持54只港股的完整分析，具有断点续跑、内存保护等企业级特性。

### 🎯 核心特性

- **全时间框架分析**：支持1分钟到日线的7个时间框架（1m、5m、15m、30m、1h、4h、1d）
- **多因子验证**：RSI、MACD、Momentum_ROC、Price_Position、Volume_Ratio等技术因子
- **时区统一处理**：自动处理Unix时间戳转换，确保时区一致性
- **断点续跑功能**：支持从中断点恢复分析，无需重新开始
- **内存保护机制**：自动清理缓存，防止内存溢出
- **滚动日志系统**：防止单个日志文件过大
- **一键启动脚本**：支持后台运行，方便管理

## 📁 项目结构

```
multi_timezone_radar/
├── core/                          # 核心分析模块
│   ├── vectorbt_wfo_analyzer.py       # 向量化回测分析器
│   └── single_stock_wfo.py           # 单股票WFO分析
├── run_0700_full_analysis.py      # 0700.HK 单股全量分析脚本
├── run_53_stocks_sequential.py    # 顺序分析脚本（可指定股票范围）
├── run_54_stocks_analysis.py      # 54只股票批量分析脚本
├── run.sh                         # 一键启动脚本
├── docs/                          # 操作与技术文档
├── PROJECT_DOCUMENTATION.md       # 技术实现说明
├── PROJECT_SUMMARY.md             # 项目总结报告
├── USER_GUIDE.md                  # 使用指南
├── requirements.txt               # 依赖清单
├── logs/                          # 日志文件
│   ├── individual_analysis/         # 单股票分析结果
│   └── sequential_analysis_summary_*/ # 总体分析报告
└── README.md                     # 项目说明
```

## 🛠️ 快速开始

### 1. 环境要求

- Python 3.11+
- 依赖库：pandas, numpy, vectorbt, talib, psutil
- 内存：建议16GB以上
- 存储：建议10GB以上可用空间

### 2. 快速自检
运行单股票全量分析脚本，快速确认依赖安装和数据路径是否正确：

```bash
python run_0700_full_analysis.py
```

### 3. 运行分析

#### 方式一：单只股票测试
```bash
python run_53_stocks_sequential.py --stock 0005.HK
```

#### 方式二：完整分析（前台运行）
```bash
python run_53_stocks_sequential.py --start 0 --end 53
```

#### 方式三：全量批处理（前台运行）
```bash
python run_54_stocks_analysis.py
```

#### 方式四：一键启动（后台运行，推荐）
```bash
./run.sh
```

#### 方式五：断点续跑
```bash
python run_53_stocks_sequential.py --resume logs/sequential_analysis_summary_<timestamp>/checkpoint.json
```

#### 方式六：监控运行状态
```bash
# 查看正在运行的分析
screen -r hk53_analysis

# 查看实时日志
tail -f logs/master_YYYYMMDD_HHMMSS.log

# 查看系统资源使用
top -p $(pgrep -f run_53_stocks_sequential.py)
```

#### 方式七：停止分析
```bash
# 优雅停止
screen -S hk53_analysis -X quit

# 强制停止（如果需要）
pkill -f run_53_stocks_sequential.py
```

## 📊 分析结果

### 输出文件结构

每次运行分析会生成时间戳目录，包含：

```
logs/
├── master_YYYYMMDD_HHMMSS.log            # 主日志文件
├── sequential_analysis_summary_YYYYMMDD_HHMMSS/  # 总体分析报告和检查点
│   ├── checkpoint.json                   # 断点续跑检查点
│   ├── comprehensive_report.md           # 综合分析报告
│   ├── all_results.json                  # 完整结果数据
│   └── summary_stats.json                # 统计摘要
└── individual_analysis/                  # 各股票详细分析结果
    ├── 0005.HK_1d_report.md              # 单股票报告
    ├── 0005.HK_1h_results.json          # 单股票结果
    └── ...                               # 其他股票文件
```

### 日志系统

系统采用滚动日志机制，自动管理日志文件大小：

- **主日志**: `logs/master_YYYYMMDD_HHMMSS.log` - 记录整体运行状态
- **错误日志**: `logs/error_YYYYMMDD_HHMMSS.log` - 记录错误信息
- **性能日志**: `logs/performance_YYYYMMDD_HHMMSS.log` - 记录性能指标
- **自动清理**: 超过30天的日志文件自动删除

### 稳健因子识别标准

因子必须同时满足以下条件才能被认定为稳健：
1. **样本外|IC| > 0.03**：具有预测能力
2. **样本内外IC符号一致**：方向稳定性
3. **覆盖股票数 > 1**：具有普适性
4. **通过0.76%交易成本测试**：经济价值

## 📈 分析框架

### 技术因子库

系统计算10个经典技术因子：
1. **RSI** (相对强弱指数)
2. **MACD** (指数平滑异同移动平均线)
3. **Volume_Ratio** (成交量比率)
4. **Momentum_ROC** (动量变化率)
5. **Bollinger_Position** (布林带位置)
6. **Z_Score** (标准化得分)
7. **ADX** (平均趋向指数)
8. **Stochastic** (随机指标)
9. **Williams_R** (威廉指标)
10. **CCI** (商品通道指数)

### 交易成本设置

- **交易手续费**：0.15%
- **滑点成本**：0.10%
- **印花税**：0.10%
- **交易费**：0.005%
- **结算费(双边)**：0.40%
- **总成本**：0.76%

## 🔧 系统特点

### 1. 抗过拟合机制
- 70%训练集，30%测试集严格分割
- 样本外验证确保因子稳健性
- 多重测试校正避免假阳性

### 2. 现实导向
- 真实港股交易成本
- 考虑流动性和滑点
- 实际可执行的因子策略

### 3. 系统可靠性
- 自动错误处理和恢复
- 完整的日志记录
- 结果验证和一致性检查

### 4. 生产级特性
- **断点续跑**: 支持从中断点恢复，无需重新开始
- **内存保护**: 自动清理缓存，防止内存溢出
- **滚动日志**: 防止单个日志文件过大
- **后台运行**: 使用screen管理，支持断开连接继续运行
- **性能监控**: 实时监控CPU、内存使用情况
- **自动恢复**: 异常后自动恢复和重试机制

## 📋 使用建议

### 1. 因子选择策略
- 优先选择在多个时间框架中表现稳健的因子
- 关注因子在不同市场环境下的表现
- 结合基本面分析进行综合判断

### 2. 风险管理
- 严格控制交易频率以降低成本影响
- 设置适当的风险管理机制
- 定期重新验证因子表现

### 3. 系统维护
- 定期清理旧的日志和结果文件
- 监控系统运行状态
- 根据市场变化调整参数

### 4. 性能优化建议
- **内存配置**: 建议系统内存16GB以上，分析器会自动限制使用
- **CPU优化**: 根据CPU核心数自动调整工作进程数
- **存储空间**: 确保有10GB以上可用空间存储分析结果
- **网络连接**: 稳定的网络连接确保数据下载完整性

### 5. 故障排除
- **内存不足**: 系统会自动清理缓存，如遇问题可重启分析
- **数据缺失**: 检查股票代码是否正确，某些股票可能数据不足
- **时区问题**: 确保系统时区设置为Asia/Hong_Kong
- **权限问题**: 确保对logs目录有读写权限

## 🛡️ 注意事项

- 所有分析基于历史数据，未来表现可能不同
- 技术因子在市场结构变化时可能失效
- 建议结合其他分析方法综合判断
- 严格按照系统信号执行，避免情绪化交易

## 🚨 Known Issues & Fixes

### Critical Issue: Timezone Comparison and Timestamp Handling (Fixed 2025-09-09)

#### Problem Description
- **Error**: `Invalid comparison between dtype=datetime64[ns, Asia/Hong_Kong] and Timestamp`
- **Symptom**: All WFO windows being skipped due to 0 data points
- **Root Cause**: Integer row indices (0, 1, 2, 3...) were being treated as Unix timestamps instead of using the actual 'timestamp' column
- **Impact**: Dates from 1969-12-31 instead of intended 2025 dates

#### Fixes Applied

1. **vectorbt_wfo_analyzer.py** (Lines 415-450):
   - Added intelligent timestamp unit detection (microseconds, milliseconds, seconds)
   - Prioritized 'timestamp' column usage over index conversion
   - Enhanced DatetimeIndex reconstruction logic

2. **single_stock_wfo.py** (Lines 120-180):
   - Added comprehensive error handling for multiprocessing data serialization
   - Implemented timestamp reconstruction when DatetimeIndex is lost
   - Added proper hasattr checks to prevent AttributeError crashes

3. **run_53_stocks_sequential.py** (Line 427):
   - Fixed JSON serialization: `TypeError: Object of type int64 is not JSON serializable`
   - Added `default=int` parameter to json.dump()

#### Technical Details
- **Multiprocessing Issue**: DatetimeIndex information was lost during data serialization between processes
- **Solution**: Added robust timestamp reconstruction logic with proper error handling
- **Prevention**: System now gracefully handles serialization issues while identifying root causes

#### Verification
- **Smoke Test**: ✅ All functionality working correctly
- **Full WFO Analysis**: ✅ 35/35 tests successful (100% success rate)
- **Data Loading**: ✅ Proper timezone handling with Asia/Hong_Kong
- **Factor Analysis**: ✅ Correct timestamp processing

#### Prevention Measures
- Added comprehensive error handling to prevent crashes
- Implemented logging to identify when data loses datetime index
- Added quick self-check command for continuous verification
- Added CI integration to prevent regression

---

## 📞 技术支持

如遇问题请检查：
1. Python环境是否符合要求
2. 依赖库是否正确安装
3. 数据路径是否正确配置
4. 日志文件中的错误信息
5. 参考上述 Known Issues & Fixes 章节

---

## 📝 版本历史

### v2.1 (2025-09-08)
- ✨ **新增断点续跑功能**: 支持从中断点恢复分析
- 🛡️ **内存保护机制**: 自动清理缓存，防止内存溢出
- 📊 **滚动日志系统**: 防止单个日志文件过大
- 🚀 **一键启动脚本**: 支持后台运行，方便管理
- 🔧 **时区处理优化**: 完善Unix时间戳转换逻辑
- 📈 **快速自检脚本**: 快速验证系统功能
- 📋 **监控和停止**: 完整的生产环境管理工具

### v2.0 (2025-09-07)
- 🎯 **核心分析功能**: 54只港股全时间框架技术因子分析
- 📊 **多因子验证**: RSI、MACD、Momentum等技术指标
- 🌐 **时区统一处理**: 自动处理Unix时间戳转换
- 💰 **真实交易成本**: 港股实际交易成本计算

---

**版本**：v2.1  
**最后更新**：2025-09-08  
**维护状态**：活跃开发中  
**下次更新**: 计划添加更多技术因子和机器学习特征