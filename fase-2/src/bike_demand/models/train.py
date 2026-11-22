"""Entrenamiento y almacenamiento inmutable de versiones."""

from __future__ import annotations

import json
import os
import platform
import subprocess

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from bike_demand.data.loader import load_data
from bike_demand.data.validation import (
    validate_schema,
    validate_target,
)

from bike_demand.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PREDICTOR_COLUMNS,
    build_pipeline,
    prepare_dataframe,
)

from bike_demand.utils.metrics import compute_metrics


def _git_commit(repo_root: Path) -> str:
    """Obtiene el commit asociado al entrenamiento."""

    supplied = os.getenv("GIT_COMMIT")

    if supplied:
        return supplied

    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except Exception:
        return "unknown"


def _relative_to_root(path: str | Path, root: Path) -> str:
    """Devuelve la ruta relativa a la raíz del repositorio cuando es posible."""

    candidate = Path(path)

    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def train_model(
    config: dict[str, Any],
    version: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Entrena y guarda una versión independiente del modelo."""

    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]

    root = Path(
        config.get("_repo_root", Path.cwd())
    )

    base_dir = (
        Path(output_dir)
        if output_dir
        else Path(config["output"].get("models_dir", "fase-2/models"))
    )

    if not base_dir.is_absolute():
        base_dir = root / base_dir

    # Validar el identificador.
    if version:
        if not version.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "El identificador de versión solo puede contener "
                "letras, números, '-' y '_'."
            )

        version_dir = base_dir / version

        if version_dir.exists():
            raise FileExistsError(
                f"La versión '{version}' ya existe; "
                "no se modificó ningún artefacto."
            )

    else:
        version_dir = base_dir / datetime.now(
            timezone.utc
        ).strftime("v%Y%m%dT%H%M%S%fZ")

    # Cargar y validar datos.
    df = load_data(data_cfg["raw_path"])

    validate_schema(df)

    validate_target(
        df,
        data_cfg["target"],
    )

    X, y = prepare_dataframe(
        df,
        drop_columns=feat_cfg.get("drop_columns"),
        leakage_columns=data_cfg.get("leakage_columns"),
        cyclic_columns=feat_cfg.get("cyclic_columns"),
        target=data_cfg["target"],
    )

    # Separación agrupada por fecha.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=config["project"].get("seed", 42),
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=df["dteday"].astype(str),
        )
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Parámetros del modelo.
    params = dict(
        model_cfg.get("params", {})
    )

    params.setdefault(
        "random_state",
        config["project"].get("seed", 42),
    )

    params.setdefault("n_jobs", -1)

    # Construir y entrenar el pipeline.
    pipeline = build_pipeline(
        model=RandomForestRegressor(**params)
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    # Evaluación.
    predictions = pipeline.predict(X_test)

    metrics = compute_metrics(
        y_test,
        predictions,
    )

    metrics["train_rows"] = int(len(train_idx))
    metrics["test_rows"] = int(len(test_idx))

    # Crear la carpeta sin sobrescribir versiones anteriores.
    try:
        version_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    except FileExistsError as exc:
        raise FileExistsError(
            f"La versión '{version_dir.name}' ya existe; "
            "no se modificó ningún artefacto."
        ) from exc

    model_path = version_dir / "modelo.joblib"
    metrics_path = version_dir / "metrics.json"
    metadata_path = version_dir / "metadata.json"

    # Guardar pipeline completo.
    joblib.dump(
        pipeline,
        model_path,
        compress=3,
    )

    metrics_payload = {
        "version": version_dir.name,
        "problem_type": "regression",
        **metrics,
    }

    metadata = {
        "version": version_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_type": "regression",
        "target": data_cfg["target"],
        "algorithm": model_cfg.get(
            "type",
            "RandomForestRegressor",
        ),
        "model_params": params,
        "seed": config["project"].get("seed", 42),

        "data": {
            "path": _relative_to_root(data_cfg["raw_path"], root),
            "rows": int(len(df)),
            "source": "Capital Bikeshare hourly dataset",
        },

        "split": {
            "strategy": "GroupShuffleSplit by dteday",
            "test_size": data_cfg.get("test_size", 0.2),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        },

        "predictor_columns": PREDICTOR_COLUMNS,

        "predictor_schema": {
            name: {
                "type": "number",
                "nullable": True,
            }
            for name in PREDICTOR_COLUMNS
        },

        "numeric_features_after_engineering": NUMERIC_FEATURES,
        "categorical_features_after_engineering": CATEGORICAL_FEATURES,
        "nullable_predictors": PREDICTOR_COLUMNS,

        "metrics_file": "metrics.json",
        "model_file": "modelo.joblib",

        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,

        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "PyYAML": yaml.__version__,
        },

        "dependencies_file": "requirements.txt",
        "code_commit": _git_commit(root),
    }

    metrics_path.write_text(
        json.dumps(
            metrics_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        **metrics_payload,
        "version_dir": str(version_dir),
        "model_path": str(model_path),
    }
