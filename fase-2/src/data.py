"""Lectura y validación de datos. No transforma ni ajusta nada."""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


class DataValidationError(ValueError):
    """Error previsible en los datos (archivo, columnas o tipos)."""


def load_data(path):
    """Lee un CSV y devuelve un DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"No se encontró el archivo: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DataValidationError(f"El archivo está vacío: {path}")
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise DataValidationError(f"No se pudo leer {path} como CSV: {e}")
    if df.empty:
        raise DataValidationError(f"El archivo no contiene filas de datos: {path}")
    return df


def validate_columns(df, required_cols):
    """Verifica que existan todas las columnas requeridas."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Faltan columnas requeridas: {missing}. "
            f"Columnas recibidas: {list(df.columns)}"
        )


def validate_numeric_types(df, cols):
    """Verifica que las columnas sean numéricas (los nulos están permitidos)."""
    invalid = [c for c in cols if not is_numeric_dtype(df[c])]
    if invalid:
        detalle = {c: str(df[c].dtype) for c in invalid}
        raise DataValidationError(
            f"Tipos de datos incompatibles (se esperaba numérico): {detalle}"
        )


def validate_target(df, target):
    """El objetivo debe existir, ser numérico y no tener nulos (solo entrenamiento)."""
    validate_columns(df, [target])
    validate_numeric_types(df, [target])
    n_nulls = int(df[target].isnull().sum())
    if n_nulls > 0:
        raise DataValidationError(
            f"La variable objetivo '{target}' tiene {n_nulls} valores nulos."
        )


def select_features(df, feature_cols):
    """Devuelve solo las features, en el orden del entrenamiento.

    Las columnas adicionales se ignoran y el orden original del CSV no importa.
    """
    validate_columns(df, feature_cols)
    return df[feature_cols].copy()