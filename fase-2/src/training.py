"""Partición, construcción y ajuste de modelos. No lee ni escribe archivos."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLS, GROUP_COL, TARGET
from src.data import DataValidationError
from src.preprocessing import build_preprocessor

MODEL_NAMES = ("ridge", "random_forest")


def split_data(df, test_size=0.2, seed=42,
               feature_cols=None, target=TARGET, group_col=GROUP_COL):
    """Separa train/test agrupando por día: un día completo va a un solo conjunto.

    Devuelve X_train, X_test, y_train, y_test e info (tamaños y días por conjunto).
    """
    feature_cols = FEATURE_COLS if feature_cols is None else feature_cols

    if not 0 < test_size < 1:
        raise DataValidationError(f"test_size debe estar entre 0 y 1, se recibió {test_size}")
    if df[group_col].isnull().any():
        raise DataValidationError(f"La columna de grupo '{group_col}' tiene valores nulos.")
    if df[group_col].nunique() < 2:
        raise DataValidationError(
            f"Se necesitan al menos 2 grupos distintos en '{group_col}' para separar train y test."
        )

    X = df[feature_cols]
    y = df[target]
    groups = df[group_col]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    info = {
        "strategy": f"GroupShuffleSplit por '{group_col}'",
        "test_size": test_size,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_groups_train": int(groups.iloc[train_idx].nunique()),
        "n_groups_test": int(groups.iloc[test_idx].nunique()),
    }
    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx], info)


def build_model(name, seed=42, params=None):
    """Devuelve un Pipeline (preprocesamiento + modelo) SIN ajustar."""
    params = dict(params or {})

    if name == "ridge":
        estimator = Ridge(**{"alpha": 1.0, **params})
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            **{"n_estimators": 200, "n_jobs": -1, **params, "random_state": seed}
        )
    else:
        raise ValueError(f"Modelo desconocido: '{name}'. Opciones: {MODEL_NAMES}")

    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])


def fit_model(model, X_train, y_train):
    """Ajusta el pipeline completo SOLO con datos de entrenamiento."""
    model.fit(X_train, y_train)
    return model