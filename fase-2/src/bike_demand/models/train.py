"""Entrenamiento del modelo."""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from bike_demand.data.loader import load_data
from bike_demand.data.validation import validate_schema, validate_target
from bike_demand.features.build_features import build_pipeline, prepare_dataframe
from bike_demand.utils.metrics import compute_metrics


def train_model(config: Dict[str, Any]) -> Dict[str, float]:
    """Entrena el modelo, guarda artefactos y devuelve las métricas.

    Args:
        config: Diccionario de configuración cargado desde YAML.

    Returns:
        Diccionario con MAE, RMSE y R² en test.
    """
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    out_cfg = config["output"]

    df = load_data(data_cfg["raw_path"])
    validate_schema(df)
    validate_target(df, data_cfg["target"])

    periods = {"hr": 24, "mnth": 12}
    X, y = prepare_dataframe(
        df,
        drop_columns=feat_cfg["drop_columns"],
        leakage_columns=data_cfg["leakage_columns"],
        cyclic_columns=feat_cfg["cyclic_columns"],
        periods=periods,
        target=data_cfg["target"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=config["project"]["seed"]
    )

    model = RandomForestRegressor(**model_cfg["params"])
    pipeline = build_pipeline(numeric_features=list(X.columns), model=model)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    Path(out_cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out_cfg["metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_cfg["model_path"])
    with open(out_cfg["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
