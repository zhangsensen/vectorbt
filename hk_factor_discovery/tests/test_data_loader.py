import pytest

pd = pytest.importorskip("pandas")
pd_testing = pytest.importorskip("pandas.testing")

from hk_factor_discovery.data_loader import HistoricalDataLoader


def _sample_frame():
    index = pd.date_range("2024-01-01 09:30", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 101.5, 102.0],
            "low": [99.5, 100.0, 100.5],
            "close": [100.2, 100.7, 101.1],
            "volume": [1_000, 1_100, 1_050],
        },
        index=index,
    )


def test_load_raw_supports_timeframe_first_layout(tmp_path, monkeypatch):
    dataframe = _sample_frame()
    timeframe_dir = tmp_path / "1m"
    timeframe_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = timeframe_dir / "0700.HK.parquet"
    parquet_path.touch()

    def fake_read_parquet(path, *_, **__):
        assert path == parquet_path
        return dataframe

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    loader = HistoricalDataLoader(data_root=tmp_path)
    loaded = loader.load("0700.HK", "1m")

    pd_testing.assert_frame_equal(loaded, dataframe)


def test_load_raw_supports_symbol_first_layout(tmp_path, monkeypatch):
    dataframe = _sample_frame()
    symbol_dir = tmp_path / "0700.HK"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = symbol_dir / "1m.parquet"
    parquet_path.touch()

    def fake_read_parquet(path, *_, **__):
        assert path == parquet_path
        return dataframe

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    loader = HistoricalDataLoader(data_root=tmp_path)
    loaded = loader.load("0700.HK", "1m")

    pd_testing.assert_frame_equal(loaded, dataframe)
