"""API de la fase 3 con el registro de solicitudes habilitado.

Desde la raíz del repositorio:
    uvicorn app:app --app-dir fase-4 --port 8000

El registro se guarda en fase-4/logs/predictions_log.csv, o en la ruta
indicada por la variable de entorno PREDICTION_LOG_PATH.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FASE4 = Path(__file__).resolve().parent
REPO_ROOT = FASE4.parent

for path in (REPO_ROOT / "fase-3", REPO_ROOT / "fase-2" / "src", FASE4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api.main import create_app  # noqa: E402
from monitoring.logger import CsvPredictionLogger  # noqa: E402

DEFAULT_LOG = FASE4 / "logs" / "predictions_log.csv"
LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", DEFAULT_LOG))

app = create_app(prediction_logger=CsvPredictionLogger(LOG_PATH))
