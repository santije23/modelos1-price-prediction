"""Validación del esquema del dataset."""

from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
    "windspeed", "casual", "registered", "cnt",
]


def validate_schema(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Verifica que el DataFrame tenga todas las columnas requeridas.

    Args:
        df: DataFrame a validar.
        required: Columnas que deben estar presentes.

    Raises:
        ValueError: Si falta alguna columna requerida.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def validate_target(df: pd.DataFrame, target: str = "cnt") -> None:
    """Verifica que la columna objetivo exista y no tenga nulos.

    Args:
        df: DataFrame a validar.
        target: Nombre de la columna objetivo.

    Raises:
        ValueError: Si la columna objetivo falta o contiene nulos.
    """
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe.")
    if df[target].isna().any():
        raise ValueError(f"La columna objetivo '{target}' contiene nulos.")
