"""Cliente de prueba de la API (solo usa la biblioteca estándar).

Ejemplos, desde la raíz del repositorio:
    python fase-3/client/predict_client.py                      # /health + fila 1
    python fase-3/client/predict_client.py --row 2              # otra fila del CSV
    python fase-3/client/predict_client.py --drop temp          # quita un campo (espera 422)
    python fase-3/client/predict_client.py --null hum           # envía hum = null
    python fase-3/client/predict_client.py --train              # entrena una versión nueva
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "example_prediction.csv"


def call(method: str, url: str, payload: dict | None = None, timeout: float = 600) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def read_record(path: Path, row: int) -> dict:
    """Convierte una fila del CSV de ejemplo en el JSON que espera /predict."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not 1 <= row <= len(rows):
        raise SystemExit(f"El CSV tiene {len(rows)} filas; --row debe estar entre 1 y {len(rows)}.")
    return {key: float(value) if "." in value else int(value) for key, value in rows[row - 1].items()}


def show(title: str, status: int, body: dict, payload: dict | None = None) -> None:
    print(f"\n== {title} -> HTTP {status}")
    if payload is not None:
        print("Solicitud:", json.dumps(payload, ensure_ascii=False))
    print("Respuesta:", json.dumps(body, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API de demanda de bicicletas.")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV con las 11 variables predictoras.")
    parser.add_argument("--row", type=int, default=1, help="Fila del CSV a enviar (desde 1).")
    parser.add_argument("--drop", action="append", default=[], help="Campo a eliminar del JSON.")
    parser.add_argument("--null", action="append", default=[], help="Campo a enviar como null.")
    parser.add_argument("--train", action="store_true", help="Llama a POST /train en lugar de /predict.")
    parser.add_argument("--skip-health", action="store_true", help="No consulta /health antes.")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    if not args.skip_health:
        show("GET /health", *call("GET", f"{base}/health"))

    if args.train:
        show("POST /train", *call("POST", f"{base}/train"))
        return

    payload = read_record(args.csv, args.row)
    for field in args.drop:
        payload.pop(field, None)
    for field in args.null:
        payload[field] = None

    status, body = call("POST", f"{base}/predict", payload)
    show("POST /predict", status, body, payload)


if __name__ == "__main__":
    main()
