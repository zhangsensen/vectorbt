# 港股短线交易策略发现系统（简化版）

本仓库现在以 `hk_factor_discovery` 为核心，实现了面向香港市场的两阶段向量化策略发现流程：

1. **阶段一 – 单因子探索**：遍历 72 个技术/统计因子与 11 个时间框架（1m→1d）的组合，利用完全向量化的回测引擎评估夏普率、稳定性、胜率等关键指标，并把结果写入 SQLite 数据库。
2. **阶段二 – 多因子组合**：在阶段一结果的基础上挑选表现最佳的因子，生成 2~3 个因子的组合，计算组合收益序列并输出排序后的最优策略列表，同样支持数据库留存与复用。

## 仓库结构

```
hk_factor_discovery/
├── config.py                  # 时间框架等核心配置
├── data_loader.py             # 支持 1m/2m/3m/5m/1d 原始数据与高阶重采样
├── database.py                # 因子结果与组合策略的 SQLite 持久化
├── factors/                   # 72 个因子定义（趋势/动量/波动率等八大类）
├── phase1/                    # 单因子探索器与向量化回测引擎
├── phase2/                    # 多因子组合与优化逻辑
├── utils/                     # 绩效指标、成本模型等通用组件
└── tests/                     # Pytest 覆盖核心流程
```

历史上的 `longport/vectorized` 与 `vectorbt_workspace` 模块仍然保留以便追溯，但新的策略开发推荐直接使用 `hk_factor_discovery` 提供的 API 与 CLI。

## 快速开始

1. **安装依赖**（推荐 Python 3.10+）
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas
   ```

2. **准备数据目录**
   - 将 `1m`、`2m`、`3m`、`5m`、`1d` 的 OHLCV Parquet/CSV 数据放在 `symbol/timeframe.parquet` 结构下。
   - 其余时间框架（10m、15m、30m、1h、2h、4h）由 `HistoricalDataLoader` 自动进行向量化重采样，无需额外文件。

3. **运行命令行入口**
   ```bash
   python -m hk_factor_discovery.main --symbol 0700.HK --data-root /path/to/data_root
   ```
   - `--phase phase1` 或 `--phase phase2` 可以只执行单阶段。
   - `--reset` 会清空 SQLite 数据库重新探索。

## 编程接口

```python
from hk_factor_discovery import HistoricalDataLoader, SingleFactorExplorer, MultiFactorCombiner
from hk_factor_discovery.factors import all_factors

loader = HistoricalDataLoader(data_provider=my_provider)
factors = all_factors()
explorer = SingleFactorExplorer("0700.HK", data_loader=loader, factors=factors)
phase1_results = explorer.explore_all_factors()

combiner = MultiFactorCombiner("0700.HK", phase1_results)
strategies = combiner.discover_strategies()
print(strategies[0]["strategy_name"], strategies[0]["sharpe_ratio"])
```

## 测试

仓库包含覆盖因子注册、探索器、组合器与数据库读写的 Pytest 测试：

```bash
pytest hk_factor_discovery/tests
```

若环境暂时未安装 `pandas` 或 `numpy`，测试会自动跳过相关用例。建议在准备好依赖后完整执行一次以验证因子池与组合逻辑。

## 关键设计亮点

- **向量化实现**：所有因子计算与回测步骤均基于 `numpy/pandas`，避免显式循环确保性能。
- **香港市场成本建模**：内置印花税、佣金、滑点与最小费用的组合成本模型，便于快速评估真实净收益。
- **结果持久化**：SQLite 架构遵循需求文档提供的三张核心表，可直接供后续分析或外部可视化使用。
- **可扩展性**：通过配置类与注册中心轻松扩展时间框架、因子或组合策略逻辑，满足未来多市场扩张需求。
