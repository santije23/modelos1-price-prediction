"""Carga de datos crudos y procesados."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Carga un CSV de datos de bike sharing.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el DataFrame está vacío.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {path}")

    df = pd.read_csv(path, parse_dates=["dteday"])
    if df.empty:
        raise ValueError(f"El archivo de datos está vacío: {path}")

    return df
