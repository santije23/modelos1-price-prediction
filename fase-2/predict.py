"""Genera predicciones con una versión guardada del modelo. No reentrena nada.

Ejemplo (desde fase-2/):
    python predict.py --model models/v1/modelo.joblib \
        --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv

Salida: CSV con 'fila' (posición en el CSV de entrada, desde 1) y 'prediccion'
(número estimado de bicicletas alquiladas en esa hora).
"""

import argparse
import sys

from src.data import DataValidationError, load_data
from src.prediction import (build_output, get_expected_features,
                            predict_dataframe, write_predictions)
from src.storage import StorageError, load_version


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Genera predicciones de demanda horaria (cnt) con una versión "
                    "guardada del modelo."
    )
    parser.add_argument("--model", required=True,
                        help="Carpeta de la versión (ej. models/v1) o ruta a modelo.joblib")
    parser.add_argument("--input", required=True,
                        help="CSV con las variables de entrada (no necesita la variable objetivo)")
    parser.add_argument("--output", required=True,
                        help="Ruta del CSV de salida (ej. outputs/predicciones.csv)")
    parser.add_argument("--id-col", default=None,
                        help="Columna del CSV de entrada a copiar como identificador "
                             "en la salida (opcional, ej. instant)")
    return parser.parse_args(argv)


def run(args):
    model, metadata = load_version(args.model)
    df = load_data(args.input)

    predictions = predict_dataframe(model, metadata, df)
    result = build_output(df, predictions, id_col=args.id_col)
    out_path = write_predictions(result, args.output)

    features = get_expected_features(metadata)
    n_imputed = int(df[features].isnull().any(axis=1).sum())
    print(f"Versión del modelo : {metadata.get('version', 'desconocida')}")
    print(f"Filas predichas    : {len(result)} (con valores nulos imputados: {n_imputed})")
    print(f"Predicción mín/media/máx: {predictions.min():.1f} / "
          f"{predictions.mean():.1f} / {predictions.max():.1f}")
    print(f"Resultados en      : {out_path}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except (DataValidationError, StorageError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())