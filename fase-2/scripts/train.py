"""Script CLI para entrenar el modelo."""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bike_demand.config import load_config  # noqa: E402
from bike_demand.models.train import train_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo de demanda de bicicletas.")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración.")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = train_model(config)
    print("Entrenamiento completado. Métricas en test:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
