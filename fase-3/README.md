# Fase 3 · API REST

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II**

**Integrantes:** Santiago Jiménez Escobar & Santiago Durán Carmona

La fase 3 expone el modelo de demanda horaria de bicicletas mediante una API REST con **FastAPI**. La API reutiliza el paquete de la fase 2 (`fase-2/src/bike_demand`): las mismas funciones de entrenamiento (`train_model`), carga de versiones (`load_version`), preprocesamiento (`prepare_dataframe`) y predicción (`predict_dataframe`). No se duplica lógica.

## Índice

1. [Operaciones](#1-operaciones)
2. [Estructura](#2-estructura)
3. [Esquema de entrada y validación](#3-esquema-de-entrada-y-validación)
4. [Selección de la versión del modelo](#4-selección-de-la-versión-del-modelo)
5. [Ejecución con Docker](#5-ejecución-con-docker)
6. [Ejecución local](#6-ejecución-local)
7. [Cliente de prueba](#7-cliente-de-prueba)
8. [Entrenamiento por HTTP](#8-entrenamiento-por-http)
9. [Pruebas automatizadas](#9-pruebas-automatizadas)
10. [Limitaciones](#10-limitaciones)

## 1. Operaciones

| Método y ruta | Descripción | Respuesta |
|---|---|---|
| `GET /health` | Indica que la API está en ejecución y si la versión configurada está cargada. | `200` `{"status": "ok", "model_version": "v2", "model_loaded": true}` |
| `POST /predict` | Recibe un registro JSON con las 11 variables predictoras y devuelve la demanda estimada. | `200` `{"prediccion": 592.66, "version": "v2"}` |
| `POST /train` | Entrena con los datos y la configuración de la fase 2 (`fase-2/configs/config.yaml`) y guarda una versión nueva. Acepta un cuerpo vacío. | `201` `{"version": "v3", "mae": ..., "rmse": ..., "r2": ..., "version_dir": "fase-2/models/v3"}` |

Códigos de error:

| Código | Cuándo |
|---|---|
| `422` | Falta una variable obligatoria, llega un tipo incompatible (por ejemplo texto) o un valor fuera de rango. El mensaje nombra el campo. |
| `503` | La versión configurada no existe o no se puede cargar. |
| `409` | Ya hay un entrenamiento en curso. |

La documentación interactiva queda disponible en `http://localhost:8000/docs`.

## 2. Estructura

```text
fase-3/
├── api/
│   ├── main.py          # create_app(): rutas, validación y carga del modelo
│   ├── schemas.py       # BikeRecord (11 predictores) y respuestas
│   ├── settings.py      # versión activa y rutas (api.yaml + variables de entorno)
│   └── server.py        # instancia `app` que arranca uvicorn
├── configs/api.yaml     # versión activa, carpeta de modelos y configuración de /train
├── client/predict_client.py   # cliente de prueba (solo biblioteca estándar)
├── tests/               # pytest con el cliente de FastAPI
├── requirements.txt     # dependencias del proyecto + fastapi, uvicorn, httpx
└── Dockerfile
```

Las versiones del modelo se leen y se escriben en `fase-2/models/`, la misma carpeta de la fase 2.

## 3. Esquema de entrada y validación

El cuerpo de `POST /predict` es un objeto JSON con las mismas 11 columnas del CSV de la fase 2 (`data/example_prediction.csv`). La variable objetivo `cnt` **no** se envía; si llega, se ignora.

| Campo | Rango | Descripción |
|---|---|---|
| `season` | 1–4 | Estación del año |
| `yr` | 0–1 | Año (0 = 2011, 1 = 2012) |
| `mnth` | 1–12 | Mes |
| `hr` | 0–23 | Hora |
| `holiday` | 0–1 | Día festivo |
| `weekday` | 0–6 | Día de la semana |
| `workingday` | 0–1 | Día laboral |
| `weathersit` | 1–4 | Condición climática |
| `temp` | 0–1 | Temperatura normalizada |
| `hum` | 0–1 | Humedad normalizada |
| `windspeed` | 0–1 | Velocidad del viento normalizada |

Reglas:

- **Todas las claves son obligatorias.** Si falta una, la respuesta es `422` con su nombre: `{"detail": "Faltan campos obligatorios: temp", ...}`.
- **Los valores son números.** Un texto como `"ocho"` o `"8"` se rechaza con `422`.
- **`null` está permitido** para representar un dato faltante. El pipeline guardado lo imputa con la regla aprendida en la fase 2 (mediana en variables numéricas y valor más frecuente en las categóricas).
- Un valor fuera de rango (por ejemplo `hr: 25`) se rechaza con `422`.

Ejemplo real de solicitud y respuesta (fila 1 de `data/example_prediction.csv`, versión `v2`):

```json
POST /predict
{"season": 1, "yr": 1, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 1, "workingday": 1,
 "weathersit": 1, "temp": 0.55, "hum": 0.65, "windspeed": 0.2}

HTTP 200
{"prediccion": 592.6558988095238, "version": "v2"}
```

## 4. Selección de la versión del modelo

La versión que atiende `/predict` se define en `fase-3/configs/api.yaml`:

```yaml
model_version: v2
models_dir: fase-2/models
train_config: fase-2/configs/config.yaml
```

Versiones disponibles en el repositorio:

| Versión | Carpeta | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| `v1` | `fase-2/models/v1/` | 26,95 | 45,20 | 0,9345 |
| `v2` (activa) | `fase-2/models/v2/` | 26,88 | 45,02 | 0,9350 |

Para usar otra versión hay dos opciones:

1. Editar `model_version` en `api.yaml` y reiniciar la API.
2. Sin editar archivos, definir la variable de entorno `MODEL_VERSION` al arrancar:
   - Docker: `docker run ... -e MODEL_VERSION=v1 bike-demand-api`
   - PowerShell: `$env:MODEL_VERSION = "v1"`
   - Linux: `MODEL_VERSION=v1 uvicorn ...`

`GET /health` informa qué versión está configurada y si se cargó. Si la carpeta no existe, `/health` devuelve `model_loaded: false` y `/predict` responde `503` con un mensaje que indica la versión y la ruta buscada. Una versión nueva creada con `/train` no se activa sola: hay que seleccionarla con cualquiera de las dos opciones anteriores.

## 5. Ejecución con Docker

Todos los comandos se ejecutan desde la **raíz del repositorio**.

### Construir la imagen

```bash
docker build -f fase-3/Dockerfile -t bike-demand-api --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
```

`GIT_COMMIT` es opcional: como `.git` no se copia a la imagen, este argumento permite que las versiones creadas con `/train` registren el commit en `metadata.json` (sin él queda `"unknown"`).

### Iniciar el contenedor

**Windows (PowerShell):**

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}/fase-2/models:/app/fase-2/models" `
  -v "${PWD}/data:/app/data:ro" `
  --name bike-api bike-demand-api
```

**Linux:**

```bash
docker run --rm -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  -v "$PWD/data:/app/data:ro" \
  --name bike-api bike-demand-api
```

| Elemento | Valor |
|---|---|
| Puerto | `8000` (el servidor escucha en `0.0.0.0:8000` y se publica con `-p 8000:8000`) |
| Datos | `data/` del equipo montado en `/app/data` (solo lectura); `/train` lee `data/processed/bike_share_with_nans.csv` |
| Modelos | `fase-2/models/` del equipo montado en `/app/fase-2/models`. Las versiones creadas con `/train` quedan en el equipo al cerrar el contenedor |

## 6. Ejecución local

```powershell
python -m pip install -r fase-3/requirements.txt
python -m uvicorn api.server:app --app-dir fase-3 --port 8000
```

`api/main.py` añade `fase-2/src` a la ruta de importación, por lo que no es necesario definir `PYTHONPATH`.

## 7. Cliente de prueba

`fase-3/client/predict_client.py` consulta `/health` y envía una fila de `data/example_prediction.csv` a `/predict`. Usa solo la biblioteca estándar de Python.

```bash
python fase-3/client/predict_client.py                 # /health + fila 1
python fase-3/client/predict_client.py --row 3         # otra fila
python fase-3/client/predict_client.py --drop temp     # sin temp -> 422
python fase-3/client/predict_client.py --null hum      # hum = null -> imputado
python fase-3/client/predict_client.py --train         # POST /train
```

Equivalente con `curl` (Linux):

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"season":1,"yr":1,"mnth":6,"hr":8,"holiday":0,"weekday":1,"workingday":1,"weathersit":1,"temp":0.55,"hum":0.65,"windspeed":0.2}'
curl -X POST http://localhost:8000/train
```

### Resultado obtenido en Docker

Con el contenedor iniciado según la sección 5 (imagen `python:3.12-slim`, versión activa `v2`):

```text
> python fase-3/client/predict_client.py

== GET /health -> HTTP 200
Respuesta: {"status": "ok", "model_version": "v2", "model_loaded": true}

== POST /predict -> HTTP 200
Solicitud: {"season": 1, "yr": 1, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 1, "workingday": 1, "weathersit": 1, "temp": 0.55, "hum": 0.65, "windspeed": 0.2}
Respuesta: {"prediccion": 592.6558988095239, "version": "v2"}

> python fase-3/client/predict_client.py --skip-health --drop temp

== POST /predict -> HTTP 422
Respuesta: {"detail": "Faltan campos obligatorios: temp", "errores": [{"campo": "temp", "mensaje": "Falta el campo obligatorio 'temp'."}]}
```

## 8. Entrenamiento por HTTP

`POST /train` no necesita cuerpo. Usa `fase-2/configs/config.yaml` (datos `data/processed/bike_share_with_nans.csv`, objetivo `cnt`, semilla 42), elige el siguiente identificador libre `vN` y responde cuando el modelo, las métricas y los metadatos ya están guardados. Las versiones anteriores no se modifican y la versión activa no cambia.

Comprobación manual en Docker (duró unos 22 s):

```text
> python fase-3/client/predict_client.py --skip-health --train

== POST /train -> HTTP 201
Respuesta: {"version": "v3", "mae": 26.951121098005075, "rmse": 45.19643807695587, "r2": 0.9344626819106555, "version_dir": "fase-2/models/v3"}
```

La nueva versión quedó en el equipo, gracias al volumen montado:

```text
fase-2/models/
├── v1/   modelo.joblib  metrics.json  metadata.json
├── v2/   modelo.joblib  metrics.json  metadata.json
└── v3/   modelo.joblib  metrics.json  metadata.json   <- creada por /train
```

`v3` reproduce exactamente las métricas de `v1` porque usa la misma configuración y la misma semilla. Esto confirma que el entrenamiento es determinista. Para activarla: `MODEL_VERSION=v3`.

## 9. Pruebas automatizadas

```bash
cd fase-3
python -m pytest -v
```

Dentro del contenedor:

```bash
docker run --rm bike-demand-api python -m pytest -v tests/
```

Las pruebas entrenan un modelo pequeño con datos sintéticos en una carpeta temporal, por lo que no dependen de los modelos publicados.

| Prueba | Comprueba |
|---|---|
| `test_health_returns_200` | `/health` responde 200 con `status: ok` |
| `test_valid_record_returns_prediction_and_version` | Una entrada válida devuelve 200 con `prediccion` y `version` |
| `test_missing_required_field_is_rejected` | Sin `temp`, la respuesta es 422 y el mensaje nombra `temp` |
| `test_null_value_is_imputed` | `hum: null` se imputa y devuelve 200 |
| `test_incompatible_type_is_rejected` | `hr: "ocho"` devuelve 422 |
| `test_out_of_range_value_is_rejected` | `hr: 25` devuelve 422 |
| `test_unavailable_model_returns_503` | Una versión inexistente devuelve 503 |
| `test_train_creates_new_version_and_keeps_previous` | `/train` crea `v2` con sus tres artefactos y no modifica `v1` |
| `test_next_version_id` | El identificador nuevo es el siguiente `vN` libre |

Resultados obtenidos:

```text
Local (Windows, Python 3.14.0):            9 passed
Docker (python:3.12-slim, Python 3.12):    9 passed
```

## 10. Limitaciones

- `/train` es síncrono: la respuesta llega cuando los artefactos están guardados (unos segundos con el dataset completo). Solo se permite un entrenamiento a la vez.
- La API no tiene autenticación; está pensada para la demostración local.
- Cambiar la versión activa requiere reiniciar la API.
- Para registrar cada solicitud y calcular métricas de operación, use la API de la fase 4 (`fase-4/`), que es esta misma API con el registro habilitado.
