"""Command line entry for the HK factor discovery workflow."""
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - handled via runtime error
    pd = None

from .config import DEFAULT_TIMEFRAMES
from .data_loader import HistoricalDataLoader
from .database import DatabaseManager
from .factors import all_factors
from .phase1.backtest_engine import SimpleBacktestEngine
from .phase1.explorer import SingleFactorExplorer
from .phase2.combiner import MultiFactorCombiner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="港股因子探索系统")
    parser.add_argument("--symbol", required=True, help="股票代码，如: 0700.HK")
    parser.add_argument("--phase", choices=["phase1", "phase2", "both"], default="both")
    parser.add_argument("--reset", action="store_true", help="重置数据库")
    parser.add_argument(
        "--data-root",
        help="可选的本地数据目录，目录下按 symbol/timeframe.parquet 存放",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if pd is None:
        raise ModuleNotFoundError("pandas is required to run the discovery workflow")

    parser = _build_parser()
    args = parser.parse_args(argv)

    db = DatabaseManager()
    if args.reset:
        db.reset_database()

    loader = HistoricalDataLoader(data_root=args.data_root)

    if args.phase in {"phase1", "both"}:
        print("🔍 开始阶段1: 单因子探索")
        explorer = SingleFactorExplorer(args.symbol, data_loader=loader)
        phase1_results = explorer.explore_all_factors()
        for row in phase1_results.values():
            row["exploration_date"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        db.save_exploration_results(phase1_results.values())
        print(f"✅ 完成 {len(phase1_results)} 个因子探索")
    else:
        stored = db.load_exploration_results(args.symbol)
        if not stored:
            raise RuntimeError("没有阶段1结果，请先运行 phase1")
        factor_map = {factor.name: factor for factor in all_factors()}
        explorer = SingleFactorExplorer(
            args.symbol,
            data_loader=loader,
            factors=list(factor_map.values()),
            backtest_engine=SimpleBacktestEngine(args.symbol),
        )
        phase1_results = {}
        for row in stored:
            factor = factor_map.get(row.factor_name)
            if factor is None:
                continue
            data = loader.load(args.symbol, row.timeframe)
            phase1_results[f"{row.timeframe}_{row.factor_name}"] = explorer.explore_single_factor(
                row.timeframe, factor, data
            )

    if args.phase in {"phase2", "both"}:
        print("🧩 开始阶段2: 多因子组合")
        combiner = MultiFactorCombiner(
            args.symbol,
            phase1_results,
            data_loader=loader,
            timeframes=DEFAULT_TIMEFRAMES,
        )
        strategies = combiner.discover_strategies()
        for strategy in strategies:
            strategy["creation_date"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        db.save_combination_strategies(strategies)
        print(f"✅ 发现 {len(strategies)} 个优质策略")

    print("🎉 系统运行完成！")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
