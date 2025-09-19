# 🚀 向量化投资组合分析器

面向多股票、多时间框架、多因子的组合评估平台。系统沿用因子分析器
的 `MultiTimeframeDataLoader`，在 CTA 与 IC 两种模式之间灵活切换，实现
大规模因子筛选与成本敏感回测。

## 🗂 项目结构

```
portfolio_analyzer/
├── analysis/              # 多股票组合与IC分析脚本
├── core/                  # 主入口（main_analyzer.py 等）
├── factors/               # 因子池与工程化工具
├── strategies/            # CTA 评估器与回测逻辑
├── tests/                 # 公用组件的单元测试
├── utils/                 # 类型修复、日志工具
├── requirements.txt       # 与因子分析器共享
└── README.md
```

所有入口脚本都会自动在本目录下创建 `logs/` 与 `results/` 时间戳文件夹，
仓库中不再保留历史运行产物，以保持整洁。

## 🧩 支持的时间框架

统一支持 `1m, 2m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 1d`。
只需准备 1m/2m/3m/5m/1d 原始 parquet 文件，其余周期由
`MultiTimeframeDataLoader` 自动重采样生成。

## ⚙️ 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r longport/vectorized/factor_analyzer/requirements.txt

python -m longport.vectorized.portfolio_analyzer.core.main_analyzer \
  --data-dir /path/to/data_root \
  --capital 300000
```

如需限制分析维度，可附加 `--timeframes 1m 15m 1d` 参数。

运行完成后，`results/final_working_<timestamp>/` 内包含 JSON 输出与 Markdown
报告，`logs/final_working_<timestamp>/` 记录完整的评估过程。

## 📈 输出指标

- **CTA 模式**：夏普率、胜率、盈亏比、交易次数、成本测试结果。
- **IC 模式**：信息系数、IC_IR、正向命中率、样本覆盖度。
- **Top 因子榜**：自动提取表现最好的 10 个因子/时间框架组合。

## 🧪 测试

运行 `pytest` 可以验证共享数据加载器与数据类型修复工具的行为：

```bash
pytest longport/vectorized/portfolio_analyzer/tests
```

测试在缺少 `pandas`/`numpy` 时会自动跳过，因此在安装完科学计算栈之后
再执行可获得完整覆盖率。

## 🔧 常见问题

| 问题 | 解决方案 |
| --- | --- |
| 运行提示找不到数据文件 | 确认 `--data-dir` 指向的目录下存在相应时间框架的 parquet 文件 |
| CTA 模式无有效因子 | 检查日志中的交易成本说明或缩短评估时间范围 |
| IC 模式指标为 NaN | 使用 `CategoricalDtypeFixer` 清洗原始数据，或检查因子输出是否为空 |

欢迎在 `core/main_analyzer.py` 中调整配置以适配不同的股票池、时间框架或
评估模式。
