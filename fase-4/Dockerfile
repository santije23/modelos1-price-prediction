# API de la fase 3 con registro de solicitudes (fase 4).
# Construir desde la raíz del repositorio:
#   docker build -f fase-4/Dockerfile -t bike-demand-monitor .
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY fase-3/requirements.txt ./fase-3/requirements.txt
RUN pip install --no-cache-dir -r fase-3/requirements.txt

COPY . .

# .git no entra a la imagen: el commit se pasa al construir para metadata.json.
#   --build-arg GIT_COMMIT=$(git rev-parse HEAD)
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

ENV PYTHONPATH="/app/fase-2/src:/app/fase-3:/app/fase-4" \
    PREDICTION_LOG_PATH="/app/fase-4/logs/predictions_log.csv"

WORKDIR /app/fase-4

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
