"""API REST de demanda de bicicletas (fase 3).

Rutas:
    GET  /health   estado de la API y de la versión configurada.
    POST /predict  predicción para un registro JSON.
    POST /train    entrena una versión nueva con la configuración de fase 2.

La instancia que arranca uvicorn está en ``api/server.py``; este módulo solo
expone ``create_app`` para que las pruebas y la fase 4 creen la aplicación con
su propia configuración.
"""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.settings import REPO_ROOT, Settings, load_settings

# Reutilizar el paquete de fase 2 sin duplicar código.
_FASE2_SRC = str(REPO_ROOT / "fase-2" / "src")
if _FASE2_SRC not in sys.path:
    sys.path.insert(0, _FASE2_SRC)

from bike_demand.config import load_config  # noqa: E402
from bike_demand.features.build_features import PREDICTOR_COLUMNS  # noqa: E402
from bike_demand.models.predict import load_version, predict_dataframe  # noqa: E402
from bike_demand.models.train import train_model  # noqa: E402

from api.schemas import (  # noqa: E402
    BikeRecord,
    HealthResponse,
    PredictResponse,
    TrainResponse,
)

PREDICT_PATH = "/predict"
_VERSION_PATTERN = re.compile(r"^v(\d+)$")


class PredictionLogger(Protocol):
    """Punto de extensión para registrar cada llamada a /predict (fase 4)."""

    def log(
        self,
        status_code: int,
        version: str | None,
        prediction: float | None,
    ) -> None: ...


class ModelStore:
    """Mantiene cargada la versión configurada y la recarga si falta."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = None
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self.settings.model_version

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> bool:
        with self._lock:
            if self.pipeline is not None:
                return True
            try:
                self.pipeline, self.metadata = load_version(self.settings.model_dir)
                self.error = None
            except Exception as exc:  # archivo ausente, joblib corrupto, etc.
                self.pipeline = None
                self.error = (
                    f"El modelo '{self.version}' no está disponible en "
                    f"{self.settings.model_dir}: {exc}"
                )
            return self.pipeline is not None


def next_version_id(models_dir: Path) -> str:
    """Primer identificador vN libre (v3 si existen v1 y v2)."""

    numbers = [
        int(match.group(1))
        for child in (models_dir.iterdir() if models_dir.is_dir() else [])
        if child.is_dir() and (match := _VERSION_PATTERN.match(child.name))
    ]
    return f"v{max(numbers, default=0) + 1}"


def _describe_errors(errors: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    """Convierte los errores de Pydantic en mensajes que nombran el campo."""

    details = []
    missing = []

    for error in errors:
        location = [str(part) for part in error.get("loc", ()) if part != "body"]
        field = ".".join(location) or "body"

        if error.get("type") == "missing":
            missing.append(field)
            message = f"Falta el campo obligatorio '{field}'."
        elif error.get("type") == "json_invalid":
            message = "El cuerpo no es un JSON válido."
        else:
            message = f"Valor inválido en '{field}': {error.get('msg')}."

        details.append({"campo": field, "mensaje": message})

    if missing:
        summary = "Faltan campos obligatorios: " + ", ".join(missing)
    else:
        summary = "; ".join(item["mensaje"] for item in details) or "Solicitud inválida."

    return summary, details


def create_app(
    settings: Settings | None = None,
    prediction_logger: PredictionLogger | None = None,
) -> FastAPI:
    """Crea la aplicación. ``prediction_logger`` lo usa la fase 4."""

    settings = settings or load_settings()
    store = ModelStore(settings)
    store.load()

    app = FastAPI(
        title="API de demanda de bicicletas",
        description="Predicción horaria de alquileres (Capital Bikeshare).",
        version="3.0.0",
    )
    app.state.settings = settings
    app.state.store = store
    app.state.prediction_logger = prediction_logger
    train_lock = threading.Lock()

    def _log(status_code: int, version: str | None, prediction: float | None) -> None:
        if prediction_logger is not None:
            prediction_logger.log(status_code, version, prediction)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        summary, details = _describe_errors(list(exc.errors()))

        if request.url.path == PREDICT_PATH:
            _log(status.HTTP_422_UNPROCESSABLE_CONTENT, None, None)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": summary, "errores": details},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=store.version,
            model_loaded=store.loaded or store.load(),
        )

    @app.post(PREDICT_PATH, response_model=PredictResponse)
    def predict(record: BikeRecord) -> PredictResponse:
        if not store.load():
            _log(status.HTTP_503_SERVICE_UNAVAILABLE, None, None)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=store.error,
            )

        row = record.model_dump()
        frame = pd.DataFrame([[row[name] for name in PREDICTOR_COLUMNS]],
                             columns=PREDICTOR_COLUMNS, dtype="float64")

        try:
            value = float(predict_dataframe(frame, store.pipeline).iloc[0])
        except Exception as exc:
            _log(status.HTTP_500_INTERNAL_SERVER_ERROR, store.version, None)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No se pudo generar la predicción: {exc}",
            ) from exc

        _log(status.HTTP_200_OK, store.version, value)
        return PredictResponse(prediccion=value, version=store.version)

    @app.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED)
    def train() -> TrainResponse:
        # Un entrenamiento a la vez para no competir por el mismo identificador.
        if not train_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Ya hay un entrenamiento en curso.",
            )
        try:
            version = next_version_id(settings.models_dir)
            config = load_config(settings.train_config)
            result = train_model(
                config,
                version=version,
                output_dir=settings.models_dir,
            )
        except FileExistsError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No se pudo entrenar: {exc}",
            ) from exc
        finally:
            train_lock.release()

        return TrainResponse(
            version=result["version"],
            mae=result["mae"],
            rmse=result["rmse"],
            r2=result["r2"],
            version_dir=Path(result["version_dir"]).relative_to(REPO_ROOT).as_posix()
            if Path(result["version_dir"]).is_relative_to(REPO_ROOT)
            else result["version_dir"],
        )

    return app

