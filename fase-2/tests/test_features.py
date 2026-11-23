"""Pruebas de ingeniería de variables y fuga de información."""

import numpy as np
import pandas as pd

from bike_demand.features.build_features import (
    add_cyclic_features,
    add_rush_hour_flag,
    prepare_dataframe,
)


def test_add_cyclic_features():
    df = pd.DataFrame({
        "hr": [0, 6, 12, 18]
    })

    out = add_cyclic_features(
        df,
        ["hr"],
        {"hr": 24},
    )

    assert np.isclose(out.loc[0, "hr_sin"], 0.0)
    assert np.isclose(out.loc[0, "hr_cos"], 1.0)


def test_add_rush_hour_flag():
    df = pd.DataFrame({
        "hr": [7, 8, 9, 12, 17, 23],
        "workingday": [1] * 6,
    })

    assert add_rush_hour_flag(df)[
        "is_rush_hour"
    ].tolist() == [1, 1, 1, 0, 1, 0]


def test_prepare_dataframe_uses_only_predictors_and_no_target():
    df = pd.DataFrame({
        "season": [1],
        "yr": [0],
        "mnth": [1],
        "hr": [10],
        "holiday": [0],
        "weekday": [6],
        "workingday": [0],
        "weathersit": [1],
        "temp": [.2],
        "hum": [.8],
        "windspeed": [0.0],
        "cnt": [13],
        "casual": [3],
        "registered": [10],
    })

    X, y = prepare_dataframe(
        df,
        target="cnt",
    )

    assert "cnt" not in X.columns
    assert "casual" not in X.columns
    assert "registered" not in X.columns

    assert y.iloc[0] == 13
    assert len(X.columns) == 16
