"""Ingeniería de variables y pipeline compartido de entrenamiento e inferencia."""

from typing import List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "temp",
    "hum",
    "windspeed",
    "hr_sin",
    "hr_cos",
    "mnth_sin",
    "mnth_cos",
]

CATEGORICAL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "is_rush_hour",
]

PREDICTOR_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]

PERIODS = {
    "hr": 24,
    "mnth": 12,
}


def add_cyclic_features(
    df: pd.DataFrame,
    columns: List[str],
    periods: dict,
) -> pd.DataFrame:
    """Convierte variables temporales en representaciones cíclicas."""

    result = df.copy()

    for col in columns:
        if col not in result.columns:
            raise ValueError(
                f"No se encontró la columna temporal '{col}'."
            )

        period = periods[col]

        result[f"{col}_sin"] = np.sin(
            2 * np.pi * result[col] / period
        )

        result[f"{col}_cos"] = np.cos(
            2 * np.pi * result[col] / period
        )

    return result


def add_rush_hour_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica las horas pico durante días laborales."""

    result = df.copy()

    rush = result["hr"].isin([7, 8, 9, 17, 18, 19])

    result["is_rush_hour"] = (
        rush & (result["workingday"] == 1)
    ).astype(int)

    return result


def prepare_dataframe(
    df: pd.DataFrame,
    drop_columns: List[str] | None = None,
    leakage_columns: List[str] | None = None,
    cyclic_columns: List[str] | None = None,
    periods: dict | None = None,
    target: str = "cnt",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Prepara las variables predictoras.

    La variable objetivo se devuelve separadamente y nunca
    forma parte de la matriz X.
    """

    drop_columns = drop_columns or [
        "instant",
        "dteday",
        "atemp",
        "casual",
        "registered",
    ]

    leakage_columns = leakage_columns or [
        "casual",
        "registered",
    ]

    cyclic_columns = cyclic_columns or ["hr", "mnth"]
    periods = periods or PERIODS

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

    # Seleccionar exclusivamente los predictores permitidos.
    result = df.loc[:, PREDICTOR_COLUMNS].copy()

    # Convertir entradas numéricas y rechazar texto incompatible.
    for column in PREDICTOR_COLUMNS:
        converted = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        invalid = result[column].notna() & converted.isna()

        if invalid.any():
            raise ValueError(
                f"La columna '{column}' contiene valores no numéricos."
            )

        result[column] = converted

    # Las variables temporales se transforman antes del pipeline.
    result = add_cyclic_features(
        result,
        cyclic_columns,
        periods,
    )

    result = add_rush_hour_flag(result)

    X = result[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y = (
        df[target].copy()
        if target in df.columns
        else None
    )

    return X, y


def build_pipeline(
    numeric_features: List[str] | None = None,
    model=None,
) -> Pipeline:
    """
    Construye un pipeline completo.

    Las reglas de imputación, escalamiento y codificación
    se aprenden exclusivamente durante el entrenamiento.
    """

    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])

    if model is None:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
