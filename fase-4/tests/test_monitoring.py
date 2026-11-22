"""Pruebas del registro de /predict y del script de monitoreo."""

import csv
import subprocess
import sys

from fastapi.testclient import TestClient

from api.main import create_app
from conftest import FASE4, VALID_RECORD
from monitoring.logger import FIELDS, CsvPredictionLogger
from scripts.monitor import format_rate, summarize


def _rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_two_valid_and_one_invalid_give_three_rows_and_33_percent(small_settings, tmp_path):
    log_path = tmp_path / "logs" / "predictions_log.csv"
    client = TestClient(create_app(small_settings, CsvPredictionLogger(log_path)))

    missing_temp = {k: v for k, v in VALID_RECORD.items() if k != "temp"}
    assert client.post("/predict", json=VALID_RECORD).status_code == 200
    assert client.post("/predict", json={**VALID_RECORD, "hr": 17}).status_code == 200
    assert client.post("/predict", json=missing_temp).status_code == 422

    # /health y /train no se registran.
    client.get("/health")

    rows = _rows(log_path)
    assert list(rows[0]) == FIELDS
    assert [row["estado_http"] for row in rows] == ["200", "200", "422"]
    assert [row["version"] for row in rows] == ["v1", "v1", ""]
    assert rows[0]["prediccion"] and rows[2]["prediccion"] == ""

    summary = summarize(log_path)
    assert (summary.total, summary.failed) == (3, 1)
    assert format_rate(summary) == "33,3 %"


def test_log_is_kept_between_app_restarts(small_settings, tmp_path):
    log_path = tmp_path / "predictions_log.csv"

    for _ in range(2):
        client = TestClient(create_app(small_settings, CsvPredictionLogger(log_path)))
        client.post("/predict", json=VALID_RECORD)

    assert len(_rows(log_path)) == 2


def test_unavailable_model_is_logged_as_503_without_version(small_settings, tmp_path):
    from api.settings import Settings

    settings = Settings("v99", small_settings.models_dir, small_settings.train_config)
    log_path = tmp_path / "predictions_log.csv"
    client = TestClient(create_app(settings, CsvPredictionLogger(log_path)))

    assert client.post("/predict", json=VALID_RECORD).status_code == 503
    (row,) = _rows(log_path)
    assert (row["estado_http"], row["version"], row["prediccion"]) == ("503", "", "")


def test_empty_or_missing_log_reports_no_requests(tmp_path):
    missing = tmp_path / "no_existe.csv"
    empty = tmp_path / "vacio.csv"
    empty.write_text(",".join(FIELDS) + "\n", encoding="utf-8")

    for path in (missing, empty):
        summary = summarize(path)
        assert summary.total == 0
        assert format_rate(summary) == "sin solicitudes"


def test_monitor_cli_output(tmp_path):
    log_path = tmp_path / "log.csv"
    log_path.write_text(
        "fecha_hora,estado_http,version,prediccion\n"
        "2026-11-21T10:00:00,200,v2,592.6559\n"
        "2026-11-21T10:01:00,200,v2,301.1000\n"
        "2026-11-21T10:02:00,422,,\n",
        encoding="utf-8",
    )

    output = subprocess.run(
        [sys.executable, str(FASE4 / "scripts" / "monitor.py"), "--log", str(log_path)],
        capture_output=True, text=True, check=True,
    ).stdout

    assert "Total de solicitudes: 3" in output
    assert "Porcentaje de fallos: 33,3 %" in output
