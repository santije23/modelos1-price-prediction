"""Registro persistente de cada llamada a POST /predict en un CSV."""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

FIELDS = ["fecha_hora", "estado_http", "version", "prediccion"]


class CsvPredictionLogger:
    """Agrega una fila por solicitud; nunca sobrescribe el historial.

    Cumple el protocolo ``PredictionLogger`` de ``fase-3/api/main.py``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        status_code: int,
        version: str | None,
        prediction: float | None,
    ) -> None:
        row = {
            "fecha_hora": datetime.now().isoformat(timespec="seconds"),
            "estado_http": int(status_code),
            "version": version or "",
            "prediccion": "" if prediction is None else f"{prediction:.4f}",
        }

        with self._lock:
            is_new = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
