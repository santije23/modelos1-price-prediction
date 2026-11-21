"""Carga de versiones y predicción por filas."""

from pathlib import Path
from typing import Any
import json

import joblib
import pandas as pd

from bike_demand.features.build_features import (
    prepare_dataframe,
    PREDICTOR_COLUMNS,
)


def resolve_model_path(model_path: str | Path) -> Path:
    """Resuelve la ruta de un pipeline guardado."""

    path = Path(model_path)

    if path.is_dir():
        path = path / "modelo.joblib"

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {path}"
        )

    return path


def load_model(model_path: str | Path):
    """Carga un pipeline serializado con Joblib."""

    path = resolve_model_path(model_path)

    return joblib.load(path)


def predict_dataframe(
    df: pd.DataFrame,
    pipeline,
) -> pd.Series:
    """
    Genera una predicción por fila.

    No requiere la variable objetivo cnt.
    """

    missing = [
        column
        for column in PREDICTOR_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Faltan columnas predictoras obligatorias: "
            + ", ".join(missing)
        )

    X, _ = prepare_dataframe(
        df,
        target="cnt",
    )

    predictions = pipeline.predict(X)

    return pd.Series(
        predictions,
        index=df.index,
        name="prediccion",
    ).clip(lower=0)


def predict(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """Predice utilizando el modelo indicado en la configuración."""

    model_path = config["output"]["model_path"]

    pipeline = load_model(model_path)

    return predict_dataframe(df, pipeline)


def load_version(version_dir: str | Path):
    """
    Carga el modelo y los metadatos de una versión.

    Devuelve:
        pipeline: modelo entrenado.
        metadata: información de la versión.
    """

    directory = Path(version_dir)

    if not directory.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de versión: {directory}"
        )

    model = load_model(directory)

    metadata_path = directory / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"La versión no contiene metadata.json: {directory}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return model, metadata
