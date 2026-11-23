"""Fixtures compartidas: un modelo pequeño y determinista en una carpeta temporal."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

FASE3 = Path(__file__).resolve().parents[1]
REPO_ROOT = FASE3.parent

for path in (FASE3, REPO_ROOT / "fase-2" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api.settings import Settings  # noqa: E402
from bike_demand.models.train import train_model  # noqa: E402

# Primera fila de data/example_prediction.csv.
VALID_RECORD = {
    "season": 1,
    "yr": 1,
    "mnth": 6,
    "hr": 8,
    "holiday": 0,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.55,
    "hum": 0.65,
    "windspeed": 0.20,
}


def _synthetic_data(path: Path, rows: int = 240) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "instant": range(1, rows + 1),
        "dteday": pd.date_range("2011-01-01", periods=rows, freq="h"),
        "season": rng.integers(1, 5, rows),
        "yr": rng.integers(0, 2, rows),
        "mnth": rng.integers(1, 13, rows),
        "hr": rng.integers(0, 24, rows),
        "holiday": rng.integers(0, 2, rows),
        "weekday": rng.integers(0, 7, rows),
        "workingday": rng.integers(0, 2, rows),
        "weathersit": rng.integers(1, 5, rows),
        "temp": rng.random(rows),
        "atemp": rng.random(rows),
        "hum": rng.random(rows),
        "windspeed": rng.random(rows),
        "casual": rng.integers(0, 50, rows),
        "registered": rng.integers(0, 100, rows),
    })
    df["cnt"] = df["casual"] + df["registered"]
    df.to_csv(path, index=False)


@pytest.fixture
def small_settings(tmp_path) -> Settings:
    """Settings que apuntan a un modelo v1 pequeño entrenado con datos sintéticos."""

    data_path = tmp_path / "raw.csv"
    _synthetic_data(data_path)

    models_dir = tmp_path / "models"
    train_config = {
        "project": {"name": "test", "seed": 42},
        "data": {"raw_path": str(data_path), "target": "cnt", "test_size": 0.25},
        "features": {"cyclic_columns": ["hr", "mnth"]},
        "model": {
            "type": "RandomForestRegressor",
            "params": {"n_estimators": 8, "max_depth": 5, "random_state": 42, "n_jobs": 1},
        },
        "output": {"models_dir": str(models_dir)},
    }

    # load_config exige encontrar la raíz (carpetas fase-1/ y data/).
    (tmp_path / "fase-1").mkdir()
    (tmp_path / "data").mkdir()
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(train_config), encoding="utf-8")

    train_model({**train_config, "_repo_root": str(tmp_path)}, version="v1")

    return Settings(model_version="v1", models_dir=models_dir, train_config=config_path)
