"""Utilities for loading multi-timeframe OHLCV data with resampling support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Tuple

import pandas as pd


def _timeframe_to_pandas_freq(timeframe: str) -> str:
    """Convert a timeframe string like ``"5m"`` to a pandas frequency code."""

    if not timeframe:
        raise ValueError("Timeframe must be a non-empty string")

    value_part = timeframe[:-1]
    unit = timeframe[-1].lower()
    if not value_part.isdigit():
        raise ValueError(f"Invalid timeframe value: {timeframe}")

    multiplier = int(value_part)
    unit_map = {"m": "T", "h": "H", "d": "D"}
    if unit not in unit_map:
        raise ValueError(f"Unsupported timeframe unit: {timeframe}")

    return f"{multiplier}{unit_map[unit]}"


@dataclass(frozen=True)
class LoadMetadata:
    """Metadata describing how a symbol/timeframe dataset was produced."""

    symbol: str
    target_timeframe: str
    source_timeframe: Optional[str]
    resampled: bool
    resample_rule: Optional[str] = None


class MultiTimeframeDataLoader:
    """Load OHLCV parquet files and synthesise higher timeframes when required."""

    BASE_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    SUPPORTED_TIMEFRAMES: Tuple[str, ...] = (
        "1m",
        "2m",
        "3m",
        "5m",
        "10m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "1d",
    )
    COLUMN_MAPPING: Mapping[str, str] = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
        "Turnover": "volume",
    }
    DEFAULT_RESAMPLE_MAP: Mapping[str, Tuple[str, ...]] = {
        "10m": ("5m", "2m", "1m"),
        "15m": ("5m", "3m", "1m"),
        "30m": ("5m", "3m", "1m"),
        "1h": ("5m", "3m", "1m"),
        "2h": ("5m", "3m", "1m"),
        "4h": ("5m", "3m", "1m"),
    }

    def __init__(
        self,
        data_dir: str | Path,
        *,
        resample_map: Optional[Mapping[str, Iterable[str]]] = None,
        cache: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.resample_map: Mapping[str, Tuple[str, ...]] = {
            key: tuple(value)
            for key, value in (resample_map or self.DEFAULT_RESAMPLE_MAP).items()
        }
        self.cache_enabled = cache
        self._cache: MutableMapping[Tuple[str, str], Tuple[pd.DataFrame, LoadMetadata]] = {}

    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear any cached dataframes."""

        self._cache.clear()

    # ------------------------------------------------------------------
    def load(
        self, symbol: str, timeframe: str, *, return_metadata: bool = False
    ) -> pd.DataFrame | Tuple[pd.DataFrame, LoadMetadata]:
        """Return data for ``symbol`` at ``timeframe``.

        If the requested timeframe is unavailable on disk the loader attempts to
        synthesise it from the best matching lower timeframe according to
        ``resample_map``.
        """

        cache_key = (symbol, timeframe)
        if self.cache_enabled and cache_key in self._cache:
            cached_df, cached_meta = self._cache[cache_key]
            if return_metadata:
                return cached_df.copy(), cached_meta
            return cached_df.copy()

        metadata = LoadMetadata(
            symbol=symbol,
            target_timeframe=timeframe,
            source_timeframe=timeframe,
            resampled=False,
            resample_rule=None,
        )

        df = self._load_direct(symbol, timeframe)
        if df.empty:
            df, metadata = self._resample_from_base(symbol, timeframe)

        if df.empty:
            if return_metadata:
                return df, metadata
            return df

        if self.cache_enabled:
            self._cache[cache_key] = (df, metadata)

        if return_metadata:
            return df.copy(), metadata
        return df.copy()

    # ------------------------------------------------------------------
    def _load_direct(self, symbol: str, timeframe: str) -> pd.DataFrame:
        timeframe_dir = self.data_dir / timeframe
        if not timeframe_dir.exists():
            return pd.DataFrame()

        file_path = timeframe_dir / f"{symbol}.parquet"
        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(file_path)
        return self._sanitise_dataframe(df)

    # ------------------------------------------------------------------
    def _resample_from_base(
        self, symbol: str, timeframe: str
    ) -> Tuple[pd.DataFrame, LoadMetadata]:
        metadata = LoadMetadata(
            symbol=symbol,
            target_timeframe=timeframe,
            source_timeframe=None,
            resampled=False,
            resample_rule=None,
        )

        base_timeframes = self.resample_map.get(timeframe, ())
        if not base_timeframes:
            return pd.DataFrame(), metadata

        target_rule = _timeframe_to_pandas_freq(timeframe)

        for base_tf in base_timeframes:
            base_df = self._load_direct(symbol, base_tf)
            if base_df.empty:
                continue

            resampled = self._resample_dataframe(base_df, target_rule)
            if resampled.empty:
                continue

            metadata = LoadMetadata(
                symbol=symbol,
                target_timeframe=timeframe,
                source_timeframe=base_tf,
                resampled=True,
                resample_rule=target_rule,
            )
            return resampled, metadata

        return pd.DataFrame(), metadata

    # ------------------------------------------------------------------
    def _sanitise_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        else:
            df.index = pd.to_datetime(df.index)

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        if any(col in df.columns for col in self.COLUMN_MAPPING):
            df = df.rename(columns=self.COLUMN_MAPPING)

        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        available_cols = [col for col in self.BASE_COLUMNS if col in df.columns]
        df = df[available_cols]

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=required_cols, how="any")

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        return df

    # ------------------------------------------------------------------
    def _resample_dataframe(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        agg_map = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg_map["volume"] = "sum"

        resampled = (
            df.resample(rule, label="right", closed="right")
            .agg(agg_map)
            .dropna(subset=["open", "high", "low", "close"], how="any")
        )

        if "volume" in resampled.columns:
            resampled["volume"] = resampled["volume"].fillna(0)

        return resampled


__all__ = ["MultiTimeframeDataLoader", "LoadMetadata"]
