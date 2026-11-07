"""Ingeniería de features y construcción del Pipeline de sklearn."""

from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def add_cyclic_features(df: pd.DataFrame, columns: List[str], periods: dict) -> pd.DataFrame:
    """Convierte columnas cíclicas en seno/coseno.

    Args:
        df: DataFrame de entrada.
        columns: Columnas a transformar (p. ej. ["hr", "mnth"]).
        periods: Periodo de cada columna (p. ej. {"hr": 24, "mnth": 12}).

    Returns:
        DataFrame con las nuevas columnas sin/cos.
    """
    df = df.copy()
    for col in columns:
        period = periods[col]
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df

def add_rush_hour_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Añade un flag de hora punta (7-9 y 17-19 en días laborables)."""
    df = df.copy()
    is_rush = df["hr"].isin([7, 8, 9, 17, 18, 19])
    df["is_rush_hour"] = (is_rush & (df["workingday"] == 1)).astype(int)
    return df

def prepare_dataframe(
    df: pd.DataFrame,
    drop_columns: List[str],
    leakage_columns: List[str],
    cyclic_columns: List[str],
    periods: dict,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica toda la transformación previa al modelo.

    Returns:
        (X, y) listos para el Pipeline de sklearn.
    """
    df = df.copy()
    df = add_cyclic_features(df, cyclic_columns, periods)
    df = add_rush_hour_flag(df)

    cols_to_drop = [c for c in drop_columns + leakage_columns if c in df.columns]
    if target in df.columns:
        cols_to_drop.append(target)

    X = df.drop(columns=cols_to_drop)
    y = df[target] if target in df.columns else None
    return X, y

def build_pipeline(numeric_features: List[str], model) -> Pipeline:
    """Construye un Pipeline con escalado y modelo."""
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
