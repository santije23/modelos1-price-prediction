# Imagen de la API REST (fase 3). Construir desde la raíz del repositorio:
#   docker build -f fase-3/Dockerfile -t bike-demand-api .
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

ENV PYTHONPATH="/app/fase-2/src:/app/fase-3"

WORKDIR /app/fase-3

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
