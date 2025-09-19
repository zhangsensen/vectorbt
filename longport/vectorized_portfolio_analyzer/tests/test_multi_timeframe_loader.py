import pytest

pandas = pytest.importorskip("pandas")
pd_testing = pytest.importorskip("pandas.testing")
np = pytest.importorskip("numpy")

from longport.common.data_loader import MultiTimeframeDataLoader


@pytest.fixture()
def sample_dataframes(tmp_path):
    symbol = "0700.HK"
    base_dir = tmp_path

    direct_dir = base_dir / "2m"
    direct_dir.mkdir(parents=True, exist_ok=True)
    direct_path = direct_dir / f"{symbol}.parquet"
    direct_path.touch()

    direct_index = pandas.date_range("2024-01-01 09:30", periods=4, freq="2T")
    direct_df = pandas.DataFrame(
        {
            "Open": np.arange(100, 104, dtype=float),
            "High": np.arange(101, 105, dtype=float),
            "Low": np.arange(99, 103, dtype=float),
            "Close": np.arange(100.5, 104.5, dtype=float),
            "Volume": np.arange(1_000, 1_004, dtype=float),
        },
        index=direct_index,
    )

    base_dir_5m = base_dir / "5m"
    base_dir_5m.mkdir(parents=True, exist_ok=True)
    base_path = base_dir_5m / f"{symbol}.parquet"
    base_path.touch()

    base_index = pandas.date_range("2024-01-01 09:30", periods=6, freq="5T")
    base_df = pandas.DataFrame(
        {
            "Open": np.linspace(200, 205, 6),
            "High": np.linspace(201, 206, 6),
            "Low": np.linspace(199, 204, 6),
            "Close": np.linspace(200.5, 205.5, 6),
            "Volume": np.linspace(2_000, 2_500, 6),
        },
        index=base_index,
    )

    data_map = {
        str(direct_path): direct_df,
        str(base_path): base_df,
    }

    def fake_read_parquet(path, *args, **kwargs):
        try:
            return data_map[str(path)]
        except KeyError as exc:
            raise FileNotFoundError(str(path)) from exc

    return symbol, direct_df, base_df, fake_read_parquet


def test_load_direct_timeframe(monkeypatch, tmp_path, sample_dataframes):
    symbol, direct_df, _, fake_reader = sample_dataframes
    monkeypatch.setattr(pandas, "read_parquet", fake_reader)

    loader = MultiTimeframeDataLoader(tmp_path)
    loaded, meta = loader.load(symbol, "2m", return_metadata=True)

    assert not meta.resampled
    assert meta.source_timeframe == "2m"
    pd_testing.assert_frame_equal(loaded, direct_df.rename(columns=str.lower))


def test_resample_from_lower_timeframe(monkeypatch, tmp_path, sample_dataframes):
    symbol, _, base_df, fake_reader = sample_dataframes
    monkeypatch.setattr(pandas, "read_parquet", fake_reader)

    loader = MultiTimeframeDataLoader(tmp_path)
    loaded, meta = loader.load(symbol, "10m", return_metadata=True)

    assert meta.resampled
    assert meta.source_timeframe == "5m"
    assert meta.resample_rule == "10T"

    expected = (
        base_df.rename(columns=str.lower)
        .resample("10T", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    pd_testing.assert_frame_equal(loaded, expected)


def test_returns_empty_when_no_data(tmp_path):
    loader = MultiTimeframeDataLoader(tmp_path)
    df, meta = loader.load("0700.HK", "10m", return_metadata=True)

    assert df.empty
    assert meta.source_timeframe is None
    assert not meta.resampled
