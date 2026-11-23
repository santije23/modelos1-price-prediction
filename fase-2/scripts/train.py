"""CLI de entrenamiento versionado."""

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from bike_demand.config import load_config
from bike_demand.models.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y guarda una versión inmutable del modelo."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Configuración YAML.",
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Identificador único, por ejemplo v1 o v2.",
    )

    parser.add_argument(
        "--output-dir",
        help="Directorio de versiones; por defecto el configurado en YAML.",
    )

    args = parser.parse_args()

    try:
        result = train_model(
            load_config(args.config),
            version=args.version,
            output_dir=args.output_dir,
        )

    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))

    print(f"Entrenamiento completado: {result['version']}")
    print(f"Artefactos: {result['version_dir']}")

    for key in ("mae", "rmse", "r2"):
        print(f"{key.upper()}: {result[key]:.4f}")


if __name__ == "__main__":
    main()
