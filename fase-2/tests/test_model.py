"""Tests de entrenamiento y predicción con un dataset sintético."""

from pathlib import Path

import pandas as pd
import pytest

from bike_demand.models.predict import load_model, predict
from bike_demand.models.train import train_model


@pytest.fixture
def synthetic_config(tmp_path):
    return {
        "project": {"name": "test", "seed": 42},
        "data": {
            "raw_path": str(tmp_path / "raw.csv"),
            "processed_path": str(tmp_path / "proc.csv"),
            "target": "cnt",
            "leakage_columns": ["casual", "registered"],
            "test_size": 0.25,
        },
        "features": {"cyclic_columns": ["hr", "mnth"], "drop_columns": ["instant", "dteday"]},
        "model": {"type": "RandomForestRegressor", "params": {"n_estimators": 5, "random_state": 42}},
        "output": {
            "model_path": str(tmp_path / "model.joblib"),
            "metrics_path": str(tmp_path / "metrics.json"),
        },
    }


@pytest.fixture
def synthetic_df(tmp_path, synthetic_config):
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "instant": range(1, n + 1),
        "dteday": pd.date_range("2011-01-01", periods=n, freq="h"),
        "season": rng.integers(1, 5, n), "yr": rng.integers(0, 2, n),
        "mnth": rng.integers(1, 13, n), "hr": rng.integers(0, 24, n),
        "holiday": rng.integers(0, 2, n), "weekday": rng.integers(0, 7, n),
        "workingday": rng.integers(0, 2, n), "weathersit": rng.integers(1, 4, n),
        "temp": rng.random(n), "atemp": rng.random(n),
        "hum": rng.random(n), "windspeed": rng.random(n),
        "casual": rng.integers(0, 50, n), "registered": rng.integers(0, 100, n),
    })
    df["cnt"] = df["casual"] + df["registered"]
    df.to_csv(synthetic_config["data"]["raw_path"], index=False)
    return df


def test_train_model_creates_artifacts(synthetic_config, synthetic_df):
    metrics = train_model(synthetic_config)
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert Path(synthetic_config["output"]["model_path"]).exists()
    assert Path(synthetic_config["output"]["metrics_path"]).exists()


def test_predict_returns_series(synthetic_config, synthetic_df):
    train_model(synthetic_config)
    pipeline = load_model(synthetic_config["output"]["model_path"])
    assert pipeline is not None
    preds = predict(synthetic_df.head(5), synthetic_config)
    assert len(preds) == 5
