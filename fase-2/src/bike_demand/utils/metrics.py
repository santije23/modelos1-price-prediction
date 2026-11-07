"""Métricas de evaluación del modelo."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """Calcula MAE, RMSE y R².

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.

    Returns:
        Diccionario con las métricas.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
