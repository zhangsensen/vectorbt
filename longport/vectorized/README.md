# Vectorized Analytics Suite

The `longport.vectorized` package hosts every reusable component that powers the
Hong Kong multi-factor research workflows.  Each analyzer leans on the shared
`common` utilities so adding a new strategy only requires wiring an entry point
around the existing building blocks.

## Shared Utilities

### MultiTimeframeDataLoader
- Normalises OHLCV columns and enforces UTC timestamps.
- Supports the unified timeframe list used across the project:
  `1m, 2m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 1d`.
- Reads raw parquet files when they exist and resamples higher intervals from
  the lowest available timeframe (typically 1m/2m/3m/5m/1d).
- Caches results in memory so repeated factor evaluations stay lightweight.

Import it once and keep it close to your analyzers:

```python
from longport.vectorized.common import MultiTimeframeDataLoader

data_loader = MultiTimeframeDataLoader("/path/to/data_root")
prices_30m = data_loader.load("0700.HK", "30m")
```

### Advanced Factor Pool
The factor analyzer exposes 72 fully vectorized factors grouped in three
families:

| 家族 | 说明 |
| --- | --- |
| 核心因子池 | 趋势、动量、波动率、成交量、微观结构、增强型、跨周期七大类基础指标 |
| 高级因子 | 随机震荡器、Ichimoku、抛物线SAR、协整、配对交易、异常检测、其他技术指标 |
| 工程化因子 | 横截面标准化、非线性变换、状态分层、交互项等二次加工信号 |

All factors rely on `safe_*` helper functions to avoid look-ahead bias and keep
the computation pipeline purely vectorized.

## Directory Overview

```
vectorized/
├── common/              # Shared helpers like MultiTimeframeDataLoader
├── factor_analyzer/     # 多因子信号分析与报告生成
└── portfolio_analyzer/  # 向量化组合评估与CTA复核
```

Each analyzer package exports its main runner so you can integrate it directly
from Python without touching filesystem paths:

```python
from longport.vectorized.factor_analyzer import OptimizedFinalWorking

analyzer = OptimizedFinalWorking(data_dir="/path/to/data_root")
results = analyzer.run_optimized_test()
```

## Command Line Usage

Both analyzers include a small CLI wrapper.  Execute them with the `-m` flag so
Python keeps the package context intact:

```bash
python -m longport.vectorized.factor_analyzer.optimized_final_working --data-dir /path/to/data_root
python -m longport.vectorized.portfolio_analyzer.core.main_analyzer --data-dir /path/to/data_root
```
Add `--capital` or `--timeframes` to override defaults when exploring custom
scenarios.

## Testing

Unit tests live under `longport/vectorized/portfolio_analyzer/tests/` and focus
on the shared utilities.  Run them from the repository root once the optional
scientific stack is available:

```bash
pytest
```
