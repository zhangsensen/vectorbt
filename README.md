# Vectorized Multi-Factor Research Suite

This repository collects the Hong Kong equities research tooling that was
previously scattered across several stand-alone projects.  The current layout
keeps every analyzer in a dedicated Python package while sharing the same
vectorized data-loading utilities and documentation style.

## Repository Layout

```
longport/
├── vectorized/                  # Unified multi-factor research package
└── vectorbt_workspace/
    └── multi_timezone_radar/    # Legacy timezone radar kept for reference
```

Additional notebooks and experiments that target JoinQuant live in the
`JoinQuant/` directory.  They are intentionally isolated so that the main
vectorized toolkit can stay dependency-light.

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r longport/vectorized/factor_analyzer/requirements.txt
   ```
   The portfolio analyzer shares the same requirement file, so a single
   installation step prepares both entry points.

2. **Prepare data folders**
   Place your OHLCV parquet files into timeframe-named subfolders:
   ```
   data_root/
   ├── 1m/
   ├── 2m/
   ├── 3m/
   ├── 5m/
   ├── 1d/
   └── …
   ```
   Raw market data is only needed for `1m`, `2m`, `3m`, `5m`, and `1d`.  Higher
   intervals are generated automatically by the shared
   `MultiTimeframeDataLoader` through vectorized resampling.

3. **Run an analyzer**
   ```bash
   python -m longport.vectorized.factor_analyzer.optimized_final_working \
       --data-dir /path/to/data_root
   ```
   Optional flags such as `--capital 500000` or
   `--timeframes 1m 5m 1d` can be supplied to fine-tune a run.  The portfolio
   analyzer exposes a similar interface under
   `longport.vectorized.portfolio_analyzer.core.main_analyzer`.

## Next Steps

Each package has its own README with detailed workflows, factor pool
explanations, and troubleshooting tips.  Start with
`longport/vectorized/README.md` for an overview of the shared utilities, then
follow the analyzer-specific guides to customize factor pools or extend the
strategy pipeline.
