from datetime import datetime

import pytest

pytest.importorskip("pandas")

from hk_factor_discovery.database import DatabaseManager


def test_database_roundtrip(tmp_path):
    db_path = tmp_path / "results.sqlite"
    manager = DatabaseManager(db_path)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    exploration_rows = [
        {
            "symbol": "0700.HK",
            "timeframe": "1m",
            "factor": "factor_a",
            "sharpe_ratio": 1.2,
            "stability": 0.8,
            "trades_count": 10,
            "win_rate": 0.55,
            "profit_factor": 1.4,
            "max_drawdown": 0.1,
            "exploration_date": now,
        }
    ]
    manager.save_exploration_results(exploration_rows)

    loaded = manager.load_exploration_results("0700.HK")
    assert loaded
    assert loaded[0].factor_name == "factor_a"

    strategies = [
        {
            "symbol": "0700.HK",
            "strategy_name": "factor_a+factor_b",
            "factors": ["factor_a", "factor_b"],
            "sharpe_ratio": 1.1,
            "stability": 0.7,
            "trades_count": 5,
            "win_rate": 0.6,
            "profit_factor": 1.3,
            "max_drawdown": 0.12,
            "creation_date": now,
        }
    ]
    manager.save_combination_strategies(strategies)
    loaded_strategies = manager.load_combination_strategies("0700.HK")
    assert loaded_strategies
    assert loaded_strategies[0].factor_combination == ["factor_a", "factor_b"]
