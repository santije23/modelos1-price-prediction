"""Carga YAML y resuelve rutas desde la raíz del repositorio."""

from pathlib import Path
from typing import Any
import yaml


def find_repo_root(config_path: Path) -> Path:
    """Localiza la raíz del repositorio a partir de la configuración."""
    for parent in config_path.parents:
        if (parent / "fase-1").is_dir() and (parent / "data").is_dir():
            return parent

    raise FileNotFoundError(
        "No se pudo identificar la raíz del repositorio. "
        "Se esperaban las carpetas 'fase-1/' y 'data/'."
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Carga y valida la configuración YAML."""
    path = Path(config_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("La configuración YAML debe contener un objeto.")

    required_sections = (
        "project",
        "data",
        "features",
        "model",
        "output",
    )

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Falta la sección requerida en config: '{section}'"
            )

    root = find_repo_root(path)
    config["_repo_root"] = str(root)

    # Resolver rutas relativas desde la raíz del repositorio.
    for section, key in (
        ("data", "raw_path"),
        ("output", "models_dir"),
    ):
        if key not in config[section]:
            raise ValueError(
                f"Falta '{key}' en la sección '{section}'."
            )

        value = Path(config[section][key])

        if not value.is_absolute():
            value = root / value

        config[section][key] = str(value.resolve())

    return config
