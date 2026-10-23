"""Cálculo de métricas. No entrena, no lee ni escribe archivos."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    """MAE, RMSE y R² como floats (serializables a JSON)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model, X, y):
    """Predice con un modelo ya ajustado y devuelve sus métricas."""
    return compute_metrics(y, model.predict(X))


def compare_with_baseline(model_metrics, baseline_metrics):
    """Cambio relativo del modelo frente al baseline.

    mae_reduction_pct y rmse_reduction_pct: positivo = el modelo reduce el error.
    r2_difference: positivo = el modelo explica más variabilidad.
    """
    return {
        "mae_reduction_pct": float(
            (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100
        ),
        "rmse_reduction_pct": float(
            (baseline_metrics["rmse"] - model_metrics["rmse"]) / baseline_metrics["rmse"] * 100
        ),
        "r2_difference": float(model_metrics["r2"] - baseline_metrics["r2"]),
    }


def evaluate_models(model, baseline, X_test, y_test, X_train=None, y_train=None):
    """Evalúa modelo y baseline sobre la MISMA partición y devuelve el dict de metrics.json.

    Si se pasan X_train y y_train, se incluye también el desempeño en train
    (útil para ver el sobreajuste que mencionaste en el notebook).
    """
    test_model = evaluate_model(model, X_test, y_test)
    test_baseline = evaluate_model(baseline, X_test, y_test)

    metrics = {
        "model": {"test": test_model},
        "baseline": {"test": test_baseline},
        "comparison": compare_with_baseline(test_model, test_baseline),
    }

    if X_train is not None and y_train is not None:
        metrics["model"]["train"] = evaluate_model(model, X_train, y_train)
        metrics["baseline"]["train"] = evaluate_model(baseline, X_train, y_train)

    return metrics