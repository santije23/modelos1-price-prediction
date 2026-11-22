"""Configuración de la API: versión activa y rutas compartidas con fase 2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "fase-3" / "configs" / "api.yaml"


@dataclass(frozen=True)
class Settings:
    """Parámetros con los que arranca la API."""

    model_version: str
    models_dir: Path
    train_config: Path

    @property
    def model_dir(self) -> Path:
        """Carpeta de la versión configurada para predecir."""
        return self.models_dir / self.model_version


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Lee ``api.yaml`` y aplica las variables de entorno.

    Variables admitidas (tienen prioridad sobre el YAML):
        API_CONFIG     ruta a otro archivo YAML de la API.
        MODEL_VERSION  versión que atiende /predict (p. ej. v1).
        MODELS_DIR     carpeta de versiones.
        TRAIN_CONFIG   configuración de entrenamiento para /train.
    """

    path = _resolve(config_path or os.getenv("API_CONFIG", DEFAULT_CONFIG))

    if not path.is_file():
        raise FileNotFoundError(f"No se encontró la configuración de la API: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    for key in ("model_version", "models_dir", "train_config"):
        if key not in raw:
            raise ValueError(f"Falta '{key}' en {path}")

    return Settings(
        model_version=os.getenv("MODEL_VERSION", str(raw["model_version"])),
        models_dir=_resolve(os.getenv("MODELS_DIR", raw["models_dir"])),
        train_config=_resolve(os.getenv("TRAIN_CONFIG", raw["train_config"])),
    )
