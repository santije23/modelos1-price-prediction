"""Evalúa una versión sobre la misma partición agrupada por día."""

import argparse
import json
import sys

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from bike_demand.config import load_config
from bike_demand.data.loader import load_data
from bike_demand.features.build_features import prepare_dataframe
from bike_demand.models.predict import load_model
from bike_demand.utils.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa una versión con el split agrupado por fecha."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Carpeta de versión o modelo.joblib.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    df = load_data(
        config["data"]["raw_path"]
    )

    X, y = prepare_dataframe(
        df,
        cyclic_columns=config["features"].get("cyclic_columns"),
        target=config["data"]["target"],
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["seed"],
    )

    _, test_idx = next(
        splitter.split(
            X,
            y,
            groups=df["dteday"].astype(str),
        )
    )

    pipeline = load_model(args.model)

    predictions = pipeline.predict(
        X.iloc[test_idx]
    )

    metrics = compute_metrics(
        y.iloc[test_idx],
        predictions,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
