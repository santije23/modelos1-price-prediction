"""Inferencia con un modelo entrenado."""

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from bike_demand.features.build_features import prepare_dataframe


def load_model(model_path: str | Path):
    """Carga un pipeline guardado con joblib."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {path}")
    return joblib.load(path)


def predict(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Genera predicciones para un DataFrame de entrada.

    Args:
        df: DataFrame con columnas crudas.
        config: Configuración cargada desde YAML.

    Returns:
        Serie con las predicciones.
    """
    pipeline = load_model(config["output"]["model_path"])
    periods = {"hr": 24, "mnth": 12}
    X, _ = prepare_dataframe(
        df,
        drop_columns=config["features"]["drop_columns"],
        leakage_columns=config["data"]["leakage_columns"],
        cyclic_columns=config["features"]["cyclic_columns"],
        periods=periods,
        target=config["data"]["target"],
    )
    return pd.Series(pipeline.predict(X))
