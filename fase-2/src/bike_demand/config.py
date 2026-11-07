"""Carga y validación de la configuración del proyecto."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Carga el YAML de configuración y resuelve rutas relativas a la raíz del repo.

    Args:
        config_path: Ruta al archivo YAML de configuración.

    Returns:
        Diccionario con la configuración.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si falta alguna clave requerida.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = ["data", "features", "model", "output"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Falta la clave requerida en config: '{key}'")

    # Resolver rutas relativas a la raíz del repositorio (dos niveles arriba de configs/)
    repo_root = config_path.parent.parent.parent
    config["_repo_root"] = str(repo_root)
    for section, key in [("data", "raw_path"), ("data", "processed_path"),
                         ("output", "model_path"), ("output", "metrics_path")]:
        raw = config[section][key]
        if not Path(raw).is_absolute():
            config[section][key] = str(repo_root / raw)

    return config
