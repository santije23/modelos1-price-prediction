"""Pruebas de la API con el cliente de FastAPI."""

from fastapi.testclient import TestClient

from api.main import create_app, next_version_id
from api.settings import Settings
from conftest import VALID_RECORD


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def test_health_returns_200(small_settings):
    response = _client(small_settings).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v1"
    assert body["model_loaded"] is True


def test_valid_record_returns_prediction_and_version(small_settings):
    response = _client(small_settings).post("/predict", json=VALID_RECORD)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediccion", "version"}
    assert body["version"] == "v1"
    assert isinstance(body["prediccion"], float)
    assert body["prediccion"] >= 0


def test_trailing_slash_is_accepted(small_settings):
    response = _client(small_settings).post("/predict/", json=VALID_RECORD)

    assert response.status_code == 200
    assert response.json()["version"] == "v1"


def test_wrong_method_explains_allowed_method(small_settings):
    response = _client(small_settings).get("/predict")

    assert response.status_code == 405
    assert response.json()["detail"] == (
        "El método GET no está permitido en /predict. Use POST."
    )


def test_missing_required_field_is_rejected(small_settings):
    record = {key: value for key, value in VALID_RECORD.items() if key != "temp"}

    response = _client(small_settings).post("/predict", json=record)

    assert response.status_code == 422
    assert "temp" in response.json()["detail"]


# --- Casos adicionales ---------------------------------------------------------

def test_null_value_is_imputed(small_settings):
    record = {**VALID_RECORD, "hum": None}

    response = _client(small_settings).post("/predict", json=record)

    assert response.status_code == 200
    assert response.json()["prediccion"] >= 0


def test_incompatible_type_is_rejected(small_settings):
    record = {**VALID_RECORD, "hr": "ocho"}

    response = _client(small_settings).post("/predict", json=record)

    assert response.status_code == 422
    assert "hr" in response.json()["detail"]


def test_out_of_range_value_is_rejected(small_settings):
    record = {**VALID_RECORD, "hr": 25}

    response = _client(small_settings).post("/predict", json=record)

    assert response.status_code == 422
    assert "hr" in response.json()["detail"]


def test_unavailable_model_returns_503(small_settings):
    settings = Settings(
        model_version="v99",
        models_dir=small_settings.models_dir,
        train_config=small_settings.train_config,
    )
    client = _client(settings)

    assert client.get("/health").json()["model_loaded"] is False

    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 503
    assert "v99" in response.json()["detail"]


def test_train_creates_new_version_and_keeps_previous(small_settings):
    previous = (small_settings.models_dir / "v1" / "modelo.joblib").read_bytes()

    response = _client(small_settings).post("/train")

    assert response.status_code == 201
    assert response.json()["version"] == "v2"
    new_dir = small_settings.models_dir / "v2"
    for name in ("modelo.joblib", "metrics.json", "metadata.json"):
        assert (new_dir / name).is_file()
    assert (small_settings.models_dir / "v1" / "modelo.joblib").read_bytes() == previous


def test_next_version_id(tmp_path):
    assert next_version_id(tmp_path) == "v1"
    for name in ("v1", "v2", "docker-v1", "v10"):
        (tmp_path / name).mkdir()
    assert next_version_id(tmp_path) == "v11"
