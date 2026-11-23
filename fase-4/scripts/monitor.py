"""Métricas de operación a partir del registro de /predict.

Uso, desde la raíz del repositorio:
    python fase-4/scripts/monitor.py
    python fase-4/scripts/monitor.py --log fase-4/logs/sample_predictions_log.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "logs" / "predictions_log.csv"


@dataclass(frozen=True)
class Summary:
    total: int
    failed: int

    @property
    def failure_rate(self) -> float | None:
        return None if self.total == 0 else 100 * self.failed / self.total


def summarize(path: Path) -> Summary:
    """Cuenta solicitudes y fallos (códigos HTTP 400-599)."""
    if not path.is_file():
        return Summary(total=0, failed=0)

    total = failed = 0
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            status = (row.get("estado_http") or "").strip()
            if not status:
                continue
            total += 1
            if 400 <= int(status) <= 599:
                failed += 1
    return Summary(total=total, failed=failed)


def format_rate(summary: Summary) -> str:
    rate = summary.failure_rate
    if rate is None:
        return "sin solicitudes"
    return f"{rate:.1f}".replace(".", ",") + " %"


def main() -> None:
    parser = argparse.ArgumentParser(description="Total de solicitudes y porcentaje de fallos de /predict.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="CSV del registro.")
    args = parser.parse_args()

    summary = summarize(args.log)
    print(f"Registro: {args.log}")
    print(f"Total de solicitudes: {summary.total}")
    print(f"Solicitudes fallidas (4xx/5xx): {summary.failed}")
    print(f"Porcentaje de fallos: {format_rate(summary)}")


if __name__ == "__main__":
    main()
