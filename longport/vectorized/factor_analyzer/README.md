# 港股智能多因子分析系统

面向 54 只港股的自动化多时间框架信号探查工具。系统完全向量化实现，
涵盖 72 个核心因子，并通过共享的 `MultiTimeframeDataLoader` 自动完成
不同时间尺度的数据拼接与重采样。

## 📦 项目结构

```
factor_analyzer/
├── factors/                # 因子池实现（72 个因子分类整理）
├── strategies/             # CTA 回测与评估逻辑
├── utils/                  # 数据类型修复、防未来函数等工具
├── optimized_final_working.py  # 推荐入口（CTA + 多时间框架报告）
├── quick_verification.py       # 成本现实化补丁快速自检
├── requirements.txt            # 依赖列表
└── README.md                   # 本说明文档
```

运行入口会自动创建 `logs/optimized_final_*/` 与
`results/optimized_final_*/` 目录，无需提前手动生成。

## 🧠 因子池总览

| 分类 | 数量 | 说明 |
| --- | --- | --- |
| 趋势因子 | 7 | DEMA、TEMA、KAMA、TRIX、Aroon、ADX 等趋势跟随指标 |
| 动量因子 | 7 | RSI(2/14/100)、Stoch RSI、CCI、ROC、WillR 等动量信号 |
| 波动率因子 | 5 | ATRP、Keltner Position、布林收缩、Parkinson 波动率等 |
| 成交量因子 | 5 | VWAP 偏离、Volume RSI、AD Line、CMF、量能偏离等 |
| 微观结构 | 3 | 高低价差、量能强度、价格效率 |
| 增强型因子 | 3 | MACD/RSI/ATR 的成交量增强版本 |
| 跨周期信号 | 10 | Smart Money Flow、Z-Score、订单流不平衡、Vol Term Structure 等 |
| 高级因子 | 35 | 随机震荡器、Ichimoku、抛物线 SAR、协整、配对交易、异常检测及其他技术指标 |
| 工程化衍生 | 12 | 横截面标准化、非线性变换、状态分层、交互项 |

所有计算均通过 `safe_*` 包装函数和 `roll_closed` 机制严格避开未来数据。
当样本不足 60 根 K 线时，系统会自动关闭 `vol_term_structure`、
`drawdown_volatility`、`skewness_60` 等长窗口信号。

## 📂 数据准备

1. 下载或整理原始 OHLCV 数据，并按照时间框架放置：
   ```
   /path/to/data_root/
   ├── 1m/    0700.HK.parquet
   ├── 2m/    ...
   ├── 3m/
   ├── 5m/
   ├── 1d/
   └── （无需手动准备 10m 以上数据）
   ```
2. `MultiTimeframeDataLoader` 会自动重采样生成 `10m、15m、30m、1h、2h、4h`
   等更高维度数据，保持索引 UTC 且无重复。

## 🚀 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r longport/vectorized/factor_analyzer/requirements.txt

python -m longport.vectorized.factor_analyzer.optimized_final_working \
  --data-dir /path/to/data_root \
  --capital 300000
```

可通过 `--timeframes 1m 5m 1d` 指定自定义时间框架列表。

运行结束后可在 `results/optimized_final_<timestamp>/` 中找到 JSON 结果和
Markdown 报告，同时 `logs/optimized_final_<timestamp>/` 会保留完整调试日志。

## 📊 输出解读

- **Timeframe Summary**：每个时间框架的有效因子数量、均值/方差、样本覆盖。
- **CTA Results**：针对每个因子的夏普率、胜率、盈亏比和交易次数。
- **Top 10 因子榜**：自动筛选成本后夏普率表现最佳的十个组合。

## 🛠 常见问题

| 问题 | 解决方案 |
| --- | --- |
| 没有生成某个时间框架结果 | 确认原始 1m/2m/3m/5m/1d 数据存在且索引为 `DatetimeIndex` |
| 因子数量少于预期 | 检查样本量是否达到 60 根 K 线阈值，或查看日志中的质量控制提示 |
| 运行时报错 `ImportError` | 重新执行 `pip install -r requirements.txt` 并确认已安装 TA-Lib |

## 📞 支持

- 运行日志：`logs/optimized_final_<timestamp>/optimized_final.log`
- 快速自检：`python -m longport.vectorized.factor_analyzer.quick_verification`
- 需要更多定制？参考 `factors/` 与 `strategies/` 中的实现进行扩展。
