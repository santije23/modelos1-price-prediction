"""Tests para la ingeniería de features."""

import numpy as np
import pandas as pd

from bike_demand.features.build_features import (
    add_cyclic_features,
    add_rush_hour_flag,
    prepare_dataframe,
)

def test_add_cyclic_features():
    df = pd.DataFrame({"hr": [0, 6, 12, 18], "mnth": [1] * 4})
    out = add_cyclic_features(df, ["hr"], {"hr": 24})
    assert "hr_sin" in out.columns and "hr_cos" in out.columns
    # hora 0 -> sin=0, cos=1
    assert np.isclose(out.loc[0, "hr_sin"], 0.0)
    assert np.isclose(out.loc[0, "hr_cos"], 1.0)

def test_add_rush_hour_flag():
    df = pd.DataFrame({
        "hr": [7, 8, 9, 12, 17, 23],
        "workingday": [1, 1, 1, 1, 1, 1],
    })
    out = add_rush_hour_flag(df)
    assert out["is_rush_hour"].tolist() == [1, 1, 1, 0, 1, 0]

def test_prepare_dataframe_drops_leakage():
    df = pd.DataFrame({
        "instant": [1], "dteday": pd.to_datetime(["2011-01-01"]),
        "hr": [10], "mnth": [1], "workingday": [1],
        "casual": [3], "registered": [10], "cnt": [13],
    })
    X, y = prepare_dataframe(
        df, drop_columns=["instant", "dteday"], leakage_columns=["casual", "registered"],
        cyclic_columns=["hr", "mnth"], periods={"hr": 24, "mnth": 12}, target="cnt",
    )
    assert "casual" not in X.columns
    assert "registered" not in X.columns
    assert y.iloc[0] == 13
