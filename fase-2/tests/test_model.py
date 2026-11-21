"""Pruebas deterministas de entrenamiento, serialización y versionamiento."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor

from bike_demand.models.predict import (
    load_model,
    predict_dataframe,
)

from bike_demand.models.train import train_model

from bike_demand.features.build_features import (
    prepare_dataframe,
    build_pipeline,
)


@pytest.fixture
def synthetic_config(tmp_path):
    return {
        "project": {
            "name": "test",
            "seed": 42,
        },
        "_repo_root": str(tmp_path),
        "data": {
            "raw_path": str(tmp_path / "raw.csv"),
            "target": "cnt",
            "leakage_columns": ["casual", "registered"],
            "test_size": .25,
        },
        "features": {
            "cyclic_columns": ["hr", "mnth"],
            "drop_columns": ["instant", "dteday", "atemp"],
        },
        "model": {
            "type": "RandomForestRegressor",
            "params": {
                "n_estimators": 8,
                "max_depth": 5,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "output": {
            "models_dir": str(tmp_path / "models"),
        },
    }


@pytest.fixture
def synthetic_df(tmp_path, synthetic_config):
    rng = np.random.default_rng(0)
    n = 120

    df = pd.DataFrame({
        "instant": range(1, n + 1),
        "dteday": pd.date_range(
            "2011-01-01",
            periods=n,
            freq="h",
        ),
        "season": rng.integers(1, 5, n),
        "yr": rng.integers(0, 2, n),
        "mnth": rng.integers(1, 13, n),
        "hr": rng.integers(0, 24, n),
        "holiday": rng.integers(0, 2, n),
        "weekday": rng.integers(0, 7, n),
        "workingday": rng.integers(0, 2, n),
        "weathersit": rng.integers(1, 5, n),
        "temp": rng.random(n),
        "atemp": rng.random(n),
        "hum": rng.random(n),
        "windspeed": rng.random(n),
        "casual": rng.integers(0, 50, n),
        "registered": rng.integers(0, 100, n),
    })

    df["cnt"] = df["casual"] + df["registered"]

    df.loc[2, "hum"] = np.nan
    df.loc[3, "windspeed"] = np.nan

    df.to_csv(
        synthetic_config["data"]["raw_path"],
        index=False,
    )

    return df


def test_train_saves_model_metrics_and_metadata(
    synthetic_config,
    synthetic_df,
):
    result = train_model(
        synthetic_config,
        version="v1",
    )

    folder = Path(result["version_dir"])

    assert (folder / "modelo.joblib").exists()
    assert (folder / "metrics.json").exists()

    metadata = json.loads(
        (folder / "metadata.json").read_text()
    )

    assert metadata["version"] == "v1"
    assert metadata["target"] == "cnt"

    assert (
        metadata["split"]["strategy"]
        == "GroupShuffleSplit by dteday"
    )

    assert metadata["predictor_schema"]["temp"] == {
        "type": "number",
        "nullable": True,
    }

    assert "scikit-learn" in metadata["dependencies"]


def test_null_values_are_imputed_and_prediction_works_without_target(
    synthetic_config,
    synthetic_df,
):
    result = train_model(
        synthetic_config,
        version="v1",
    )

    pipeline = load_model(
        result["version_dir"]
    )

    input_df = synthetic_df.drop(
        columns=[
            "cnt",
            "casual",
            "registered",
            "instant",
            "dteday",
            "atemp",
        ]
    ).head(5).copy()

    input_df.loc[input_df.index[0], "hum"] = np.nan

    predictions = predict_dataframe(
        input_df,
        pipeline,
    )

    assert len(predictions) == 5
    assert predictions.notna().all()
    assert (predictions >= 0).all()


def test_in_memory_and_reloaded_pipeline_are_equivalent(
    synthetic_config,
    synthetic_df,
):
    X, y = prepare_dataframe(
        synthetic_df,
        target="cnt",
    )

    pipeline = build_pipeline(
        model=RandomForestRegressor(
            n_estimators=8,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        )
    )

    pipeline.fit(X, y)

    before = pipeline.predict(
        X.head(6)
    )

    folder = (
        Path(synthetic_config["output"]["models_dir"])
        / "manual"
    )

    folder.mkdir(parents=True)

    joblib.dump(
        pipeline,
        folder / "modelo.joblib",
    )

    after = load_model(folder).predict(
        X.head(6)
    )

    np.testing.assert_allclose(
        before,
        after,
        rtol=1e-12,
        atol=1e-12,
    )


def test_duplicate_version_is_rejected_without_modifying_existing_files(
    synthetic_config,
    synthetic_df,
):
    result = train_model(
        synthetic_config,
        version="v1",
    )

    model = Path(result["version_dir"]) / "modelo.joblib"

    original = model.read_bytes()

    with pytest.raises(
        FileExistsError,
        match="ya existe",
    ):
        train_model(
            synthetic_config,
            version="v1",
        )

    assert model.read_bytes() == original


def test_two_versions_can_coexist(
    synthetic_config,
    synthetic_df,
):
    first = train_model(
        synthetic_config,
        version="v1",
    )

    second = train_model(
        synthetic_config,
        version="v2",
    )

    assert Path(first["model_path"]).exists()
    assert Path(second["model_path"]).exists()

    assert load_model(first["version_dir"]) is not None
    assert load_model(second["version_dir"]) is not None
