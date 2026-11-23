"""Esquemas de entrada y salida de la API.

Los 11 campos coinciden con ``PREDICTOR_COLUMNS`` de fase 2. Todos son
obligatorios (la clave debe estar presente), pero admiten ``null``: el valor
faltante lo imputa el pipeline guardado, igual que en fase 2.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


def _number(low: float, high: float, description: str):
    """Número dentro de [low, high] o null."""
    return Annotated[
        Annotated[float, Field(ge=low, le=high)] | None,
        Field(description=description),
    ]


class BikeRecord(BaseModel):
    """Un registro horario con las variables predictoras del proyecto."""

    # strict=True: "8" (texto) se rechaza; 8 y 8.0 se aceptan.
    # Campos adicionales (por ejemplo cnt) se ignoran.
    model_config = ConfigDict(
        strict=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "season": 1,
                "yr": 1,
                "mnth": 6,
                "hr": 8,
                "holiday": 0,
                "weekday": 1,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.55,
                "hum": 0.65,
                "windspeed": 0.20,
            }
        },
    )

    season: _number(1, 4, "Estación del año (1-4).")
    yr: _number(0, 1, "Año codificado (0 = 2011, 1 = 2012).")
    mnth: _number(1, 12, "Mes (1-12).")
    hr: _number(0, 23, "Hora del día (0-23).")
    holiday: _number(0, 1, "Día festivo (0/1).")
    weekday: _number(0, 6, "Día de la semana (0-6).")
    workingday: _number(0, 1, "Día laboral (0/1).")
    weathersit: _number(1, 4, "Condición climática (1-4).")
    temp: _number(0, 1, "Temperatura normalizada (0-1).")
    hum: _number(0, 1, "Humedad normalizada (0-1).")
    windspeed: _number(0, 1, "Velocidad del viento normalizada (0-1).")


class PredictResponse(BaseModel):
    prediccion: float = Field(description="Alquileres estimados para la hora.")
    version: str = Field(description="Versión del modelo que generó la predicción.")


class TrainResponse(BaseModel):
    version: str
    mae: float
    rmse: float
    r2: float
    version_dir: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
