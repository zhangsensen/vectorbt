"""SQLite persistence helpers for factor exploration results."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

DEFAULT_DB_PATH = Path("hk_factor_results.sqlite")


@dataclass
class FactorResult:
    symbol: str
    timeframe: str
    factor_name: str
    sharpe_ratio: float
    stability: float
    trades_count: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    information_coefficient: float
    exploration_date: str


@dataclass
class StrategyResult:
    symbol: str
    strategy_name: str
    factor_combination: List[str]
    sharpe_ratio: float
    stability: float
    trades_count: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    average_information_coefficient: float
    creation_date: str


class DatabaseManager:
    """Persist factor exploration and strategy results in SQLite."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self._ensure_schema()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS factor_exploration_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    stability REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    information_coefficient REAL NOT NULL DEFAULT 0,
                    exploration_date TEXT NOT NULL,
                    UNIQUE(symbol, timeframe, factor_name)
                );

                CREATE TABLE IF NOT EXISTS combination_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    factor_combination TEXT NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    stability REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    average_information_coefficient REAL NOT NULL DEFAULT 0,
                    creation_date TEXT NOT NULL,
                    UNIQUE(symbol, strategy_name)
                );

                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT
                );
                """
            )
            self._ensure_column(cur, "factor_exploration_results", "information_coefficient", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(
                cur,
                "combination_strategies",
                "average_information_coefficient",
                "REAL NOT NULL DEFAULT 0",
            )
            conn.commit()

    @staticmethod
    def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, definition: str) -> None:
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def reset_database(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                DELETE FROM factor_exploration_results;
                DELETE FROM combination_strategies;
                DELETE FROM system_config;
                VACUUM;
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    def save_exploration_results(self, results: Iterable[Mapping[str, object]]) -> None:
        with self._connect() as conn:
            rows = [
                (
                    r["symbol"],
                    r["timeframe"],
                    r["factor"],
                    r["sharpe_ratio"],
                    r["stability"],
                    r["trades_count"],
                    r["win_rate"],
                    r["profit_factor"],
                    r["max_drawdown"],
                    r.get("information_coefficient", 0.0),
                    r["exploration_date"],
                )
                for r in results
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO factor_exploration_results (
                    symbol, timeframe, factor_name, sharpe_ratio, stability,
                    trades_count, win_rate, profit_factor, max_drawdown,
                    information_coefficient, exploration_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def load_exploration_results(self, symbol: str) -> List[FactorResult]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT symbol, timeframe, factor_name, sharpe_ratio, stability,
                       trades_count, win_rate, profit_factor, max_drawdown,
                       information_coefficient, exploration_date
                FROM factor_exploration_results
                WHERE symbol = ?
                ORDER BY timeframe, sharpe_ratio DESC
                """,
                (symbol,),
            )
            rows = cur.fetchall()
        return [FactorResult(*row) for row in rows]

    def save_combination_strategies(self, strategies: Iterable[Mapping[str, object]]) -> None:
        with self._connect() as conn:
            rows = [
                (
                    s["symbol"],
                    s["strategy_name"],
                    json.dumps(s["factors"]),
                    s["sharpe_ratio"],
                    s["stability"],
                    s["trades_count"],
                    s["win_rate"],
                    s["profit_factor"],
                    s["max_drawdown"],
                    s.get("average_information_coefficient", 0.0),
                    s["creation_date"],
                )
                for s in strategies
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO combination_strategies (
                    symbol, strategy_name, factor_combination, sharpe_ratio,
                    stability, trades_count, win_rate, profit_factor, max_drawdown,
                    average_information_coefficient, creation_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def load_combination_strategies(self, symbol: str) -> List[StrategyResult]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT symbol, strategy_name, factor_combination, sharpe_ratio, stability,
                       trades_count, win_rate, profit_factor, max_drawdown,
                       average_information_coefficient, creation_date
                FROM combination_strategies
                WHERE symbol = ?
                ORDER BY sharpe_ratio DESC
                """,
                (symbol,),
            )
            rows = cur.fetchall()
        return [
            StrategyResult(row[0], row[1], json.loads(row[2]), *row[3:])
            for row in rows
        ]
