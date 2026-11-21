"""CLI para predecir por lotes desde un CSV."""

import argparse
import json
import sys

from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from bike_demand.models.predict import (
    load_model,
    predict_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera una predicción por fila de un CSV."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Carpeta de versión o archivo modelo.joblib.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV con los 11 predictores; cnt no es necesario.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="CSV de salida.",
    )

    args = parser.parse_args()

    source = Path(args.input)

    if not source.is_file():
        parser.error(
            f"No existe el CSV de entrada: {source}"
        )

    try:
        df = pd.read_csv(source)

        pipeline = load_model(args.model)

        predictions = predict_dataframe(
            df,
            pipeline,
        )

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as exc:
        parser.error(str(exc))

    result = pd.DataFrame({
        "fila": range(1, len(df) + 1),
        "prediccion": predictions.to_numpy(),
    })

    version_dir = (
        Path(args.model)
        if Path(args.model).is_dir()
        else Path(args.model).parent
    )

    metadata_path = version_dir / "metadata.json"

    if metadata_path.is_file():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        result["version"] = metadata["version"]

    destination = Path(args.output)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        destination,
        index=False,
    )

    print(f"Predicciones: {len(result)}")
    print(f"Archivo: {destination.resolve()}")


if __name__ == "__main__":
    main()

