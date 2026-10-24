"""Construcción del preprocesamiento. Solo construye: el ajuste se hace en el entrenamiento."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CAT_COLS, NUM_COLS


def build_preprocessor(num_cols=None, cat_cols=None):
    """Devuelve un ColumnTransformer SIN ajustar.

    - Numéricas: imputación por mediana + estandarización.
    - Categóricas: imputación por moda + one-hot
      (las categorías no vistas en el entrenamiento se ignoran: todo en 0).
    - Cualquier otra columna que llegue se descarta (remainder="drop").
    """
    num_cols = NUM_COLS if num_cols is None else num_cols
    cat_cols = CAT_COLS if cat_cols is None else cat_cols

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )