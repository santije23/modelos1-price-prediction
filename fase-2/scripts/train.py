"""Entrena el baseline (Ridge) y el modelo (Random Forest) y guarda una versión.

Ejemplo (desde fase-2/):
    python train.py --data data/processed/bike_share_with_nans.csv --target cnt \
        --output-dir models --version v1 --seed 42
"""

import argparse
import sys
from pathlib import Path

from src.config import FEATURE_COLS, GROUP_COL, TARGET
from src.data import (DataValidationError, load_data, validate_columns,
                      validate_numeric_types, validate_target)
from src.evaluation import evaluate_models
from src.storage import (StorageError, VersionExistsError, build_metadata,
                         save_version, validate_version_id)
from src.training import build_model, fit_model, split_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Entrena el baseline (Ridge) y el modelo (Random Forest) "
                    "y guarda una versión identificable en --output-dir/--version."
    )
    parser.add_argument("--data", required=True,
                        help="Ruta al CSV con datos etiquetados")
    parser.add_argument("--target", default=TARGET,
                        help="Nombre de la variable objetivo (default: %(default)s)")
    parser.add_argument("--output-dir", default="models",
                        help="Directorio donde se guardan las versiones (default: %(default)s)")
    parser.add_argument("--version", required=True,
                        help="Identificador de la versión, ej. v1. No se sobrescribe si ya existe")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: %(default)s)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporción de días para prueba (default: %(default)s)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Número de árboles del Random Forest (default: %(default)s)")
    parser.add_argument("--min-samples-leaf", type=int, default=1,
                        help="Mínimo de muestras por hoja del Random Forest (default: %(default)s)")
    return parser.parse_args(argv)


def check_version_available(output_dir, version):
    """Falla ANTES de entrenar si la versión ya existe."""
    validate_version_id(version)
    version_dir = Path(output_dir) / version
    if version_dir.exists():
        raise VersionExistsError(
            f"La versión '{version}' ya existe en {version_dir}. "
            "Usa otro identificador; las versiones no se sobrescriben."
        )


def print_summary(version, metrics, paths):
    test_model = metrics["model"]["test"]
    test_base = metrics["baseline"]["test"]
    comp = metrics["comparison"]
    print(f"\n=== Resumen de la versión '{version}' (métricas en prueba) ===")
    print(f"{'':16}{'MAE':>9}{'RMSE':>9}{'R2':>9}")
    print(f"{'Baseline Ridge':16}{test_base['mae']:9.2f}{test_base['rmse']:9.2f}{test_base['r2']:9.4f}")
    print(f"{'Random Forest':16}{test_model['mae']:9.2f}{test_model['rmse']:9.2f}{test_model['r2']:9.4f}")
    print(f"Reducción de MAE vs baseline: {comp['mae_reduction_pct']:.1f}% | "
          f"reducción de RMSE: {comp['rmse_reduction_pct']:.1f}%")
    print("\nArchivos generados:")
    for key in ("model", "metrics", "metadata"):
        print(f"  {paths[key]}")


def run(args):
    check_version_available(args.output_dir, args.version)

    # 1. Lectura y validación
    df = load_data(args.data)
    validate_columns(df, FEATURE_COLS + [args.target, GROUP_COL])
    validate_target(df, args.target)
    validate_numeric_types(df, FEATURE_COLS)

    # 2. Partición agrupada por día (ANTES de ajustar cualquier transformación)
    X_train, X_test, y_train, y_test, split_info = split_data(
        df, test_size=args.test_size, seed=args.seed, target=args.target
    )

    # 3. Entrenamiento: modelo y baseline con la misma partición
    rf_params = {"n_estimators": args.n_estimators,
                 "min_samples_leaf": args.min_samples_leaf}
    model = fit_model(build_model("random_forest", args.seed, rf_params), X_train, y_train)
    baseline = fit_model(build_model("ridge", args.seed), X_train, y_train)

    # 4. Evaluación sobre datos no usados para ajustar
    metrics = evaluate_models(model, baseline, X_test, y_test, X_train, y_train)

    # 5. Guardado
    metadata = build_metadata(
        model_name="random_forest", pipeline=model, target=args.target,
        X_train=X_train, seed=args.seed, split_info=split_info,
        data_path=args.data, n_rows=len(df),
    )
    paths = save_version(args.output_dir, args.version, model, metrics, metadata)

    print_summary(args.version, metrics, paths)
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