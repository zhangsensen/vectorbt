"""Single factor exploration workflow."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional

try:  # pragma: no cover
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

from ..config import DEFAULT_TIMEFRAMES
from ..data_loader import HistoricalDataLoader
from ..factors import all_factors
from .backtest_engine import SimpleBacktestEngine


class SingleFactorExplorer:
    """Explore 72 factors across multiple timeframes."""

    def __init__(
        self,
        symbol: str,
        timeframes: Optional[Iterable[str]] = None,
        factors: Optional[Iterable] = None,
        data_loader: Optional[HistoricalDataLoader] = None,
        backtest_engine: Optional[SimpleBacktestEngine] = None,
    ) -> None:
        if pd is None:
            raise ModuleNotFoundError("pandas is required for factor exploration")
        self.symbol = symbol
        self.timeframes = list(timeframes) if timeframes is not None else list(DEFAULT_TIMEFRAMES)
        self.factors = list(factors) if factors is not None else all_factors()
        self.data_loader = data_loader
        if self.data_loader is None:
            raise ValueError("data_loader must be provided for SingleFactorExplorer")
        self.backtest_engine = backtest_engine or SimpleBacktestEngine(symbol)

    def explore_all_factors(self) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        for timeframe in self.timeframes:
            data = self.data_loader.load(self.symbol, timeframe)
            for factor in self.factors:
                key = f"{timeframe}_{factor.name}"
                results[key] = self.explore_single_factor(timeframe, factor, data)
        return results

    def explore_single_factor(self, timeframe: str, factor, data: Optional["pd.DataFrame"] = None) -> Dict[str, object]:
        if data is None:
            data = self.data_loader.load(self.symbol, timeframe)
        signals = factor.generate_signals(self.symbol, timeframe, data)
        backtest = self.backtest_engine.backtest_factor(data, signals)
        return {
            "symbol": self.symbol,
            "timeframe": timeframe,
            "factor": factor.name,
            "sharpe_ratio": backtest["sharpe_ratio"],
            "stability": backtest["stability"],
            "trades_count": backtest["trades_count"],
            "win_rate": backtest["win_rate"],
            "profit_factor": backtest["profit_factor"],
            "max_drawdown": backtest["max_drawdown"],
            "information_coefficient": backtest["information_coefficient"],
            "returns": backtest["returns"],
            "equity_curve": backtest["equity_curve"],
            "exploration_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
