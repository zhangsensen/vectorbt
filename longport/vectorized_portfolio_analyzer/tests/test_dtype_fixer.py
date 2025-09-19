import pytest

pandas = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from longport.vectorized_portfolio_analyzer.utils.dtype_fixer import (
    CategoricalDtypeFixer,
)


def test_comprehensive_fix_converts_and_filters_columns():
    df = pandas.DataFrame(
        {
            "close": [100, 101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "categorical_factor": pandas.Categorical(["A", "B", "A", "C", "B"]),
            "numeric_factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant_factor": [1, 1, 1, 1, 1],
            "nan_factor": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    fixer = CategoricalDtypeFixer()
    fixed_df, report = fixer.comprehensive_fix(df)

    # 保留基础行情列以及成功转换的因子列
    assert set(["close", "volume", "categorical_factor", "numeric_factor"]) <= set(
        fixed_df.columns
    )

    # 常量列和全NaN列应被移除
    assert "constant_factor" not in fixed_df.columns
    assert "nan_factor" not in fixed_df.columns

    # 转换后的分类列应为数值型并包含非NaN数据
    assert fixed_df["categorical_factor"].dtype.kind in {"f", "i"}
    assert fixed_df["categorical_factor"].notna().all()

    validation = report["validation"]
    assert set(validation["valid_factors"]) == {"categorical_factor", "numeric_factor"}
    assert set(validation["problematic_factors"]) == {
        "constant_factor",
        "nan_factor",
    }
