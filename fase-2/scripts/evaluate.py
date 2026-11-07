"""Script CLI para evaluar un modelo guardado sobre el set de test."""

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bike_demand.config import load_config  # noqa: E402
from bike_demand.data.loader import load_data  # noqa: E402
from bike_demand.features.build_features import prepare_dataframe  # noqa: E402
from bike_demand.models.predict import load_model  # noqa: E402
from bike_demand.utils.metrics import compute_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evalúa el modelo sobre el set de test.")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración.")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_data(config["data"]["raw_path"])
    periods = {"hr": 24, "mnth": 12}
    X, y = prepare_dataframe(
        df,
        drop_columns=config["features"]["drop_columns"],
        leakage_columns=config["data"]["leakage_columns"],
        cyclic_columns=config["features"]["cyclic_columns"],
        periods=periods,
        target=config["data"]["target"],
    )
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["project"]["seed"]
    )
    pipeline = load_model(config["output"]["model_path"])
    metrics = compute_metrics(y_test, pipeline.predict(X_test))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
