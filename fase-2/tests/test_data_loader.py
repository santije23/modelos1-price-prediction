"""Tests para el loader y la validación de datos."""

import pandas as pd
import pytest

from bike_demand.data.loader import load_data
from bike_demand.data.validation import validate_schema, validate_target


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "instant": [1, 2],
        "dteday": pd.to_datetime(["2011-01-01", "2011-01-01"]),
        "season": [1, 1], "yr": [0, 0], "mnth": [1, 1], "hr": [0, 1],
        "holiday": [0, 0], "weekday": [6, 6], "workingday": [0, 0],
        "weathersit": [1, 1], "temp": [0.24, 0.22], "atemp": [0.2879, 0.2727],
        "hum": [0.81, 0.80], "windspeed": [0.0, 0.0],
        "casual": [3, 8], "registered": [13, 32], "cnt": [16, 40],
    })


def test_validate_schema_ok(sample_df):
    validate_schema(sample_df)


def test_validate_schema_missing_column(sample_df):
    with pytest.raises(ValueError):
        validate_schema(sample_df.drop(columns=["cnt"]))


def test_validate_target_ok(sample_df):
    validate_target(sample_df)


def test_validate_target_with_nan(sample_df):
    sample_df.loc[0, "cnt"] = None
    with pytest.raises(ValueError):
        validate_target(sample_df)


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "no_existe.csv")
