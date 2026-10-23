"""Validación, predicción y escritura de resultados sobre datos nuevos.

No entrena ni reajusta nada: usa el modelo (con su preprocesamiento) tal como se guardó.
"""

import os
from pathlib import Path

import pandas as pd

from src.data import (select_features, validate_columns,
                      validate_numeric_types)
from src.storage import StorageError

PREDICTION_COL = "prediccion"
ROW_COL = "fila"


def get_expected_features(metadata):
    """Nombres de las variables que la versión espera, en el orden del entrenamiento."""
    try:
        return list(metadata["features"]["names"])
    except (KeyError, TypeError):
        raise StorageError(
            "metadata.json no contiene el esquema de variables (features.names). "
            "La versión está incompleta."
        )


def predict_dataframe(model, metadata, df):
    """Valida el DataFrame y devuelve una predicción por fila, en el mismo orden."""
    features = get_expected_features(metadata)
    validate_columns(df, features)
    validate_numeric_types(df, features)
    X = select_features(df, features)

    predictions = model.predict(X)
    if len(predictions) != len(df):  # defensa: nunca perder filas en silencio
        raise RuntimeError(
            f"El modelo devolvió {len(predictions)} predicciones para {len(df)} filas."
        )
    return pd.Series(predictions, index=df.index, name=PREDICTION_COL)


def build_output(df, predictions, id_col=None):
    """Tabla de resultados: fila de entrada (1, 2, ...), [identificador] y prediccion."""
    result = pd.DataFrame({
        ROW_COL: range(1, len(df) + 1),
        PREDICTION_COL: predictions.to_numpy(),
    })
    if id_col is not None:
        validate_columns(df, [id_col])
        result.insert(1, id_col, df[id_col].to_numpy())
    return result


def write_predictions(result, path):
    """Escribe el CSV de forma segura: nunca deja un archivo a medias."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise StorageError(f"No se pudo escribir el archivo de salida {path}: {e}") from e
    return path