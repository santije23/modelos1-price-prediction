"""Guardado y carga de versiones del modelo. Cada versión vive en su propia carpeta."""

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import DATA_SOURCE, REQUIREMENTS_FILE

MODEL_FILE = "modelo.joblib"
METRICS_FILE = "metrics.json"
METADATA_FILE = "metadata.json"
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StorageError(Exception):
    """Error previsible al guardar o cargar una versión."""


class VersionExistsError(StorageError):
    """Se intentó crear una versión que ya existe."""


def validate_version_id(version):
    """Solo letras, números, punto, guion y guion bajo (evita rutas peligrosas)."""
    if not VERSION_PATTERN.match(str(version)):
        raise StorageError(
            f"Identificador de versión inválido: '{version}'. "
            "Usa letras, números, '.', '-' o '_' (ej. v1, v2, 2026-10-22)."
        )


def get_git_commit():
    """Hash del commit actual. Dentro de Docker (sin .git) se toma de GIT_COMMIT."""
    env_value = os.environ.get("GIT_COMMIT")
    if env_value:
        return env_value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5, cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "desconocido"


def build_metadata(*, model_name, pipeline, target, X_train, seed, split_info,
                   data_path, n_rows, data_source=DATA_SOURCE,
                   problem_type="regression", requirements_file=REQUIREMENTS_FILE):
    """Arma el contenido de metadata.json (sin version ni fecha: las agrega save_version)."""
    return {
        "problem_type": problem_type,
        "target": target,
        "algorithm": model_name,
        "model_params": pipeline.named_steps["model"].get_params(),
        "seed": seed,
        "features": {
            "names": list(X_train.columns),
            "dtypes": {c: str(t) for c, t in X_train.dtypes.items()},
        },
        "data": {
            "source": data_source,
            "file": str(data_path),
            "n_rows": int(n_rows),
        },
        "split": split_info,
        "evaluation_metrics": ["mae", "rmse", "r2"],
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "requirements_file": requirements_file,
        },
        "git_commit": get_git_commit(),
    }


def _write_json(path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False, default=str)


def save_version(output_dir, version, model, metrics, metadata):
    """Guarda modelo, métricas y metadatos en output_dir/version/.

    Si la versión ya existe, falla sin tocar nada: nunca sobrescribe.
    Si algo falla a mitad del guardado, borra la carpeta incompleta.
    """
    validate_version_id(version)
    version_dir = Path(output_dir) / version
    version_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        version_dir.mkdir()  # falla si ya existe
    except FileExistsError:
        raise VersionExistsError(
            f"La versión '{version}' ya existe en {version_dir}. "
            "Usa otro identificador; las versiones no se sobrescriben."
        )

    full_metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metadata,
    }
    paths = {
        "version_dir": version_dir,
        "model": version_dir / MODEL_FILE,
        "metrics": version_dir / METRICS_FILE,
        "metadata": version_dir / METADATA_FILE,
    }

    try:
        joblib.dump(model, paths["model"], compress=3)
        _write_json(paths["metrics"], metrics)
        _write_json(paths["metadata"], full_metadata)
    except Exception as e:
        shutil.rmtree(version_dir, ignore_errors=True)
        raise StorageError(f"No se pudo guardar la versión '{version}': {e}") from e

    return paths


def resolve_model_path(path):
    """Acepta la carpeta de la versión o la ruta directa a modelo.joblib."""
    path = Path(path)
    if path.is_dir():
        path = path / MODEL_FILE
    if not path.is_file():
        raise StorageError(f"No se encontró el modelo: {path}")
    return path


def load_version(path):
    """Carga una versión y devuelve (modelo, metadata). No reentrena ni reajusta nada."""
    model_path = resolve_model_path(path)
    metadata_path = model_path.parent / METADATA_FILE
    if not metadata_path.is_file():
        raise StorageError(
            f"Falta {METADATA_FILE} junto al modelo ({metadata_path}). "
            "La versión está incompleta."
        )

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise StorageError(f"No se pudo cargar el modelo {model_path}: {e}") from e
    if not hasattr(model, "predict"):
        raise StorageError(f"El archivo {model_path} no contiene un modelo válido.")

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise StorageError(f"{METADATA_FILE} está dañado: {e}") from e

    return model, metadata