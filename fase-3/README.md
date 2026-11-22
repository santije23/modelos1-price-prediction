# Fase 3 · API REST

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II**

**Integrantes:** Santiago Jiménez Escobar & Santiago Durán Carmona

La fase 3 expone el modelo de demanda horaria de bicicletas mediante una API REST construida con **FastAPI**. La API reutiliza el paquete de la fase 2 (`fase-2/src/bike_demand`) sin duplicar código:

- `train_model` para entrenar;
- `load_version` para cargar una versión;
- `prepare_dataframe` para el preprocesamiento;
- `predict_dataframe` para predecir.

También usa las versiones guardadas en `fase-2/models/`.

## Índice

1. [Endpoints](#1-endpoints)
2. [Estructura](#2-estructura)
3. [Esquema de entrada y validación](#3-esquema-de-entrada-y-validación)
4. [Selección de la versión del modelo](#4-selección-de-la-versión-del-modelo)
5. [Guía de prueba paso a paso con Docker](#5-guía-de-prueba-paso-a-paso-con-docker)
6. [Probar los endpoints](#6-probar-los-endpoints)
   - [Desde el navegador](#61-desde-el-navegador)
   - [Con Postman](#62-con-postman)
   - [Con el cliente de Python](#63-con-el-cliente-de-python)
   - [Con la terminal](#64-con-la-terminal)
7. [Entrenamiento por HTTP (/train)](#7-entrenamiento-por-http-train)
8. [Pruebas automatizadas (pytest)](#8-pruebas-automatizadas-pytest)
9. [Ejecución local sin Docker](#9-ejecución-local-sin-docker)
10. [Detener el contenedor y relación con la fase 4](#10-detener-el-contenedor-y-relación-con-la-fase-4)
11. [Solución de problemas](#11-solución-de-problemas)
12. [Comportamiento de la API](#12-comportamiento-de-la-api)

---

## 1. Endpoints

URL base: `http://localhost:8000`

| Método | Ruta | Cuerpo | Respuesta correcta | Descripción |
|---|---|---|---|---|
| `GET` | `/health` | — | `200` | Indica que la API está en ejecución, qué versión está configurada y si se cargó. |
| `POST` | `/predict` | JSON con las 11 variables predictoras | `200` | Devuelve la demanda estimada y la versión que la produjo. |
| `POST` | `/train` | vacío | `201` | Entrena con los datos y la configuración de la fase 2 y guarda una versión nueva. |
| `GET` | `/docs` | — | `200` | Documentación interactiva (Swagger UI) para probar los endpoints desde el navegador. |

### `GET /health`

Respuesta `200`:

```json
{"status": "ok", "model_version": "v2", "model_loaded": true}
```

| Campo | Significado |
|---|---|
| `status` | Siempre `"ok"` mientras la API está en ejecución. |
| `model_version` | Versión configurada para `/predict`. |
| `model_loaded` | `true` si la carpeta de esa versión existe y el modelo se cargó; `false` en caso contrario (entonces `/predict` responde `503`). |

### `POST /predict`

Solicitud (cabecera `Content-Type: application/json`):

```json
{
  "season": 1, "yr": 1, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 1,
  "workingday": 1, "weathersit": 1, "temp": 0.55, "hum": 0.65, "windspeed": 0.20
}
```

Respuesta `200`:

```json
{"prediccion": 592.6558988095239, "version": "v2"}
```

| Campo | Significado |
|---|---|
| `prediccion` | Número estimado de bicicletas alquiladas en esa hora (regresión; nunca es negativo). |
| `version` | Versión del modelo que generó la predicción. |

Respuestas de error:

| Código | Causa | Ejemplo de respuesta |
|---|---|---|
| `422` | Falta una variable obligatoria | `{"detail": "Faltan campos obligatorios: temp", "errores": [{"campo": "temp", "mensaje": "Falta el campo obligatorio 'temp'."}]}` |
| `422` | Tipo incompatible (texto) | `{"detail": "Valor inválido en 'hr': Input should be a valid number.", ...}` |
| `422` | Valor fuera de rango | `{"detail": "Valor inválido en 'hr': Input should be less than or equal to 23.", ...}` |
| `422` | El cuerpo no es JSON | `{"detail": "El cuerpo no es un JSON válido.", ...}` |
| `503` | La versión configurada no existe o no se puede cargar | `{"detail": "El modelo 'v9' no está disponible en /app/fase-2/models/v9: ..."}` |

### `POST /train`

No requiere cuerpo. Respuesta `201` cuando el modelo, las métricas y los metadatos ya están guardados:

```json
{"version": "v3", "mae": 26.951121098005075, "rmse": 45.19643807695587, "r2": 0.9344626819106555, "version_dir": "fase-2/models/v3"}
```

Si se envía otra solicitud de entrenamiento mientras uno está en curso, la respuesta es `409`.

---

## 2. Estructura

```text
fase-3/
├── api/
│   ├── main.py          # create_app(): rutas, validación y carga del modelo
│   ├── schemas.py       # BikeRecord (11 predictores) y modelos de respuesta
│   ├── settings.py      # versión activa y rutas (api.yaml + variables de entorno)
│   └── server.py        # instancia `app` que arranca uvicorn
├── configs/api.yaml     # versión activa, carpeta de modelos y configuración de /train
├── client/
│   ├── predict_client.py                     # cliente de prueba en Python (biblioteca estándar)
│   └── bike_demand_api.postman_collection.json   # colección de Postman
├── tests/               # pytest con el cliente de pruebas de FastAPI
├── requirements.txt     # dependencias del proyecto + fastapi, uvicorn, httpx
├── Dockerfile
└── README.md
```

---

## 3. Esquema de entrada y validación

El cuerpo de `POST /predict` tiene las mismas 11 columnas del CSV de la fase 2 (`data/example_prediction.csv`), con los mismos nombres y tipos usados al entrenar. Todas las variables del proyecto son numéricas: las categóricas (`season`, `weathersit`, etc.) se entrenaron como códigos numéricos, así que se envían como números.

| Campo | Tipo | Rango | Descripción |
|---|---|---|---|
| `season` | número | 1–4 | Estación del año (codificada como en el dataset original) |
| `yr` | número | 0–1 | Año (0 = 2011, 1 = 2012) |
| `mnth` | número | 1–12 | Mes |
| `hr` | número | 0–23 | Hora del día |
| `holiday` | número | 0–1 | Día festivo |
| `weekday` | número | 0–6 | Día de la semana (0 = domingo) |
| `workingday` | número | 0–1 | Día laboral |
| `weathersit` | número | 1–4 | Clima (1 despejado … 4 lluvia o nieve fuerte) |
| `temp` | número | 0–1 | Temperatura normalizada |
| `hum` | número | 0–1 | Humedad normalizada |
| `windspeed` | número | 0–1 | Velocidad del viento normalizada |

Reglas de validación:

1. **Todas las claves son obligatorias.** Si falta una, se responde `422` con su nombre.
2. **Los valores deben ser números.** Un texto como `"ocho"` o `"8"` se rechaza con `422`.
3. **`null` está permitido** para representar un dato faltante. El pipeline guardado en la fase 2 lo imputa con la regla aprendida al entrenar: la mediana en las variables numéricas y el valor más frecuente en las categóricas.
4. **Los valores fuera de rango** (por ejemplo `hr: 25`) se rechazan con `422`.
5. **La variable objetivo `cnt` no se envía.** Si llega, se ignora.

---

## 4. Selección de la versión del modelo

La versión que atiende `/predict` se define en `fase-3/configs/api.yaml`:

```yaml
model_version: v2                        # versión activa
models_dir: fase-2/models                # carpeta de versiones (compartida con fase 2)
train_config: fase-2/configs/config.yaml # configuración que usa /train
```

Versiones publicadas en el repositorio:

| Versión | Carpeta | Configuración | MAE | RMSE | R² |
|---|---|---|---:|---:|---:|
| `v1` | `fase-2/models/v1/` | `fase-2/configs/config.yaml` | 26,95 | 45,20 | 0,9345 |
| `v2` (activa) | `fase-2/models/v2/` | `fase-2/configs/versions/v2.yaml` | 26,88 | 45,02 | 0,9350 |

Cada carpeta contiene `modelo.joblib`, `metrics.json` y `metadata.json`.

Para usar otra versión disponible, elija una de estas dos formas:

- **Editar la configuración:** cambiar `model_version: v1` en `fase-3/configs/api.yaml` y reiniciar la API. En Docker hay que reconstruir la imagen, porque el YAML se copia dentro de ella.
- **Variable de entorno, sin editar archivos:** iniciar la API con `MODEL_VERSION`. Esta variable tiene prioridad sobre el YAML.
  - Docker: añadir `-e MODEL_VERSION=v1` al comando `docker run`.
  - PowerShell: `$env:MODEL_VERSION = "v1"`.
  - Linux: `MODEL_VERSION=v1 python -m uvicorn ...`.

`GET /health` confirma qué versión quedó configurada. Una versión nueva creada con `/train` no se activa sola; se selecciona de la misma forma, por ejemplo con `-e MODEL_VERSION=v3`.

---

## 5. Guía de prueba paso a paso con Docker

Esta guía parte de una copia nueva del repositorio. Todos los comandos se ejecutan desde la **raíz del repositorio** (la carpeta que contiene `fase-1/`, `fase-2/`, `fase-3/`, `fase-4/` y `data/`).

### Paso 1. Requisitos

- **Docker Desktop** (Windows) o **Docker Engine** (Linux) instalado y **en ejecución**. Compruébelo con:

  ```bash
  docker version
  ```

  Si aparece un error como `failed to connect to the docker API`, abra Docker Desktop y espere a que indique *Engine running*.

- **Git**, para clonar el repositorio.
- **Python 3.12 o superior**, solo para el cliente de Python y para pytest local. El navegador y Postman no lo necesitan.

### Paso 2. Obtener el proyecto

```bash
git clone https://github.com/santije23/modelos1-price-prediction
cd modelos1-price-prediction
```

### Paso 3. Construir la imagen

```bash
docker build -f fase-3/Dockerfile -t bike-demand-api .
```

- El punto final (`.`) es el contexto de construcción: la raíz del repositorio.
- La primera vez tarda varios minutos porque instala las dependencias.
- Al terminar, `docker images bike-demand-api` muestra la imagen.

Para que las versiones creadas con `/train` registren el commit en su `metadata.json`, se añade `--build-arg GIT_COMMIT=$(git rev-parse HEAD)` al comando anterior. Sin ese argumento, el campo queda como `"unknown"`, porque la carpeta `.git` no se copia a la imagen.

### Paso 4. Iniciar el contenedor

**Windows (PowerShell):**

```powershell
docker run -d --rm -p 8000:8000 `
  -v "${PWD}/fase-2/models:/app/fase-2/models" `
  -v "${PWD}/data:/app/data:ro" `
  --name bike-api bike-demand-api
```

**Linux:**

```bash
docker run -d --rm -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  -v "$PWD/data:/app/data:ro" \
  --name bike-api bike-demand-api
```

Qué significa cada opción:

| Opción | Efecto |
|---|---|
| `-d` | Ejecuta el contenedor en segundo plano y devuelve la terminal. |
| `--rm` | Elimina el contenedor al detenerlo. Los archivos de los volúmenes se conservan. |
| `-p 8000:8000` | Publica el puerto. El servidor escucha en `0.0.0.0:8000` dentro del contenedor y queda accesible en `http://localhost:8000`. |
| `-v .../fase-2/models:/app/fase-2/models` | **Montaje de modelos.** La API lee `v1` y `v2` de esa carpeta del equipo, y `/train` escribe ahí las versiones nuevas, que permanecen al cerrar el contenedor. |
| `-v .../data:/app/data:ro` | **Acceso a los datos** (solo lectura). `/train` lee `data/processed/bike_share_with_nans.csv`. |
| `--name bike-api` | Nombre del contenedor, para detenerlo o ver sus registros. |
| `--user` (Linux) | Hace que los archivos creados por `/train` pertenezcan a su usuario y no a root. |

### Paso 5. Comprobar que arrancó

```bash
docker ps --filter name=bike-api
docker logs bike-api
```

En los registros debe aparecer `Uvicorn running on http://0.0.0.0:8000`. La carga del modelo tarda unos segundos.

### Paso 6. Probar los endpoints

Siga la [sección 6](#6-probar-los-endpoints) con cualquiera de las opciones: navegador, Postman, cliente de Python o terminal.

### Paso 7. Detener el contenedor

```bash
docker stop bike-api
```

Vea la [sección 10](#10-detener-el-contenedor-y-relación-con-la-fase-4) antes de iniciar la API de la fase 4.

---

## 6. Probar los endpoints

Con el contenedor en ejecución (sección 5), se esperan los resultados de esta tabla. Son los obtenidos con la versión `v2`:

| # | Prueba | Código esperado | Resultado obtenido |
|---|---|---|---|
| 1 | `GET /health` | `200` | `{"status": "ok", "model_version": "v2", "model_loaded": true}` |
| 2 | `POST /predict` con la fila 1 de `data/example_prediction.csv` | `200` | `{"prediccion": 592.6558988095239, "version": "v2"}` |
| 3 | `POST /predict` con la fila 3 | `200` | `{"prediccion": 781.2367182409821, "version": "v2"}` |
| 4 | `POST /predict` sin `temp` | `422` | `{"detail": "Faltan campos obligatorios: temp", ...}` |
| 5 | `POST /predict` con `"hum": null` | `200` | Predicción con `hum` imputada |
| 6 | `POST /predict` con `"hr": "ocho"` | `422` | `{"detail": "Valor inválido en 'hr': ...", ...}` |
| 7 | `POST /train` | `201` | `{"version": "v3", ...}` (ver sección 7) |

### 6.1. Desde el navegador

**`GET /health`:** abra `http://localhost:8000/health`. El navegador muestra el JSON de estado.

**`POST /predict` y `POST /train` con Swagger UI:** un navegador no envía solicitudes POST desde la barra de direcciones. Para eso, FastAPI genera una página interactiva:

1. Abra `http://localhost:8000/docs`.
2. Despliegue **POST /predict** y pulse **Try it out**.
3. El cuadro *Request body* ya trae un ejemplo válido (fila 1 del CSV). Pulse **Execute**.
4. En *Server response* debe aparecer **Code 200** y el cuerpo `{"prediccion": ..., "version": "v2"}`.
5. Para probar la validación, borre la línea `"temp": 0.55,` del cuerpo (revise que el JSON siga siendo válido, sin coma sobrante al final) y pulse **Execute**. Debe aparecer **Code 422** con `Faltan campos obligatorios: temp`.
6. Para probar un dato faltante, cambie `"hum": 0.65` por `"hum": null` y ejecute. Debe responder **200**.
7. Para entrenar, despliegue **POST /train**, pulse **Try it out** y **Execute**. Tras unos 20–30 segundos aparece **Code 201** con la versión nueva.

`http://localhost:8000/redoc` muestra la misma documentación en formato de lectura.

### 6.2. Con Postman

El repositorio incluye la colección `fase-3/client/bike_demand_api.postman_collection.json`, con las 7 solicitudes de la tabla anterior y comprobaciones automáticas (pestaña *Test Results*).

**Importar la colección:**

1. Abra Postman y pulse **Import** (esquina superior izquierda).
2. Arrastre el archivo `bike_demand_api.postman_collection.json` o selecciónelo con **files**.
3. Aparece la colección **Demanda de bicicletas - API (fases 3 y 4)**. Su variable `baseUrl` vale `http://localhost:8000`: haga clic en la colección y abra la pestaña **Variables** para verla o cambiarla.

**Enviar una solicitud:**

1. Abra **2. POST /predict - válida (fila 1 del CSV)**.
2. En la pestaña **Body** verá el JSON en formato *raw → JSON*.
3. Pulse **Send**. Abajo debe aparecer `Status: 200 OK` y la respuesta con `prediccion` y `version`.
4. En **Test Results** deben aparecer las comprobaciones en verde (*PASS*).

**Ejecutar todas de una vez:** clic derecho sobre la colección → **Run collection** → **Run**. Se envían las 7 solicitudes en orden; la 7 entrena una versión nueva y tarda unos segundos más. Si no desea entrenar, desmarque esa solicitud antes de ejecutar.

**Crear la solicitud a mano** (sin importar la colección):

1. **New → HTTP Request**.
2. Método **POST** y URL `http://localhost:8000/predict`.
3. Pestaña **Body** → **raw** → tipo **JSON**, y pegue el JSON de la [sección 1](#post-predict).
4. Pulse **Send**.

Postman añade automáticamente la cabecera `Content-Type: application/json` al elegir *raw → JSON*.

### 6.3. Con el cliente de Python

`fase-3/client/predict_client.py` consulta `/health` y envía una fila de `data/example_prediction.csv` a `/predict`. Usa solo la biblioteca estándar, así que no necesita instalar dependencias.

```bash
python fase-3/client/predict_client.py                       # /health + fila 1
python fase-3/client/predict_client.py --row 3               # fila 3
python fase-3/client/predict_client.py --drop temp           # sin temp -> 422
python fase-3/client/predict_client.py --null hum            # hum = null -> 200
python fase-3/client/predict_client.py --train               # POST /train
python fase-3/client/predict_client.py --url http://localhost:8001   # otra URL
```

En Windows puede usarse `py` en lugar de `python`.

Salida obtenida con el contenedor de la sección 5:

```text
> python fase-3/client/predict_client.py

== GET /health -> HTTP 200
Respuesta: {"status": "ok", "model_version": "v2", "model_loaded": true}

== POST /predict -> HTTP 200
Solicitud: {"season": 1, "yr": 1, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 1, "workingday": 1, "weathersit": 1, "temp": 0.55, "hum": 0.65, "windspeed": 0.2}
Respuesta: {"prediccion": 592.6558988095239, "version": "v2"}

> python fase-3/client/predict_client.py --skip-health --drop temp

== POST /predict -> HTTP 422
Solicitud: {"season": 1, "yr": 1, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 1, "workingday": 1, "weathersit": 1, "hum": 0.65, "windspeed": 0.2}
Respuesta: {"detail": "Faltan campos obligatorios: temp", "errores": [{"campo": "temp", "mensaje": "Falta el campo obligatorio 'temp'."}]}
```

### 6.4. Con la terminal

**Windows (PowerShell):**

```powershell
Invoke-RestMethod http://localhost:8000/health

$body = '{"season":1,"yr":1,"mnth":6,"hr":8,"holiday":0,"weekday":1,"workingday":1,"weathersit":1,"temp":0.55,"hum":0.65,"windspeed":0.2}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body

Invoke-RestMethod -Method Post -Uri http://localhost:8000/train
```

En PowerShell, `Invoke-RestMethod` muestra los errores (`422`, `503`) como excepción, con el mensaje JSON de la API en el texto del error.

**Linux / macOS (curl):**

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"season":1,"yr":1,"mnth":6,"hr":8,"holiday":0,"weekday":1,"workingday":1,"weathersit":1,"temp":0.55,"hum":0.65,"windspeed":0.2}'

curl -X POST http://localhost:8000/train
```

---

## 7. Entrenamiento por HTTP (/train)

`POST /train` usa `fase-2/configs/config.yaml`, la misma configuración de la fase 2:

- datos: `data/processed/bike_share_with_nans.csv`;
- objetivo: `cnt`;
- semilla: 42.

Funcionamiento:

1. Elige el siguiente identificador libre `vN`: con `v1` y `v2` existentes, crea `v3`.
2. Responde cuando el modelo, las métricas y los metadatos ya están guardados.
3. No modifica las versiones anteriores ni cambia la versión activa.

**Comprobación manual realizada en Docker:**

```text
> python fase-3/client/predict_client.py --skip-health --train

== POST /train -> HTTP 201
Respuesta: {"version": "v3", "mae": 26.951121098005075, "rmse": 45.19643807695587, "r2": 0.9344626819106555, "version_dir": "fase-2/models/v3"}
```

La nueva versión quedó guardada en el equipo gracias al volumen montado:

```text
fase-2/models/
├── v1/   modelo.joblib  metrics.json  metadata.json
├── v2/   modelo.joblib  metrics.json  metadata.json
└── v3/   modelo.joblib  metrics.json  metadata.json   <- creada por /train
```

Para verificarlo después de detener el contenedor:

```powershell
Get-ChildItem fase-2/models -Recurse -File      # Windows
```

```bash
find fase-2/models -type f | sort               # Linux
```

`v3` reproduce las métricas de `v1` porque usa la misma configuración y semilla, lo que confirma que el entrenamiento es determinista. Para que `/predict` la use, inicie el contenedor con `-e MODEL_VERSION=v3`.

---

## 8. Pruebas automatizadas (pytest)

Las pruebas usan el cliente de pruebas de FastAPI (`TestClient`). Primero entrenan un modelo pequeño con datos sintéticos en una carpeta temporal, así que no dependen de los modelos publicados ni de un servidor en ejecución.

### Dentro de Docker (no requiere Python local)

```bash
docker run --rm bike-demand-api python -m pytest -v tests/
```

### En el equipo

1. Crear y activar un entorno virtual desde la raíz del repositorio:

   **Windows (PowerShell):**

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   Si PowerShell bloquea la activación, ejecute primero `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

   **Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Instalar las dependencias de la fase 3, que incluyen las del proyecto:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r fase-3/requirements.txt
   ```

3. Ejecutar las pruebas:

   ```bash
   cd fase-3
   python -m pytest -v
   cd ..
   ```

### Resultado obtenido

```text
tests/test_api.py::test_health_returns_200 PASSED
tests/test_api.py::test_valid_record_returns_prediction_and_version PASSED
tests/test_api.py::test_missing_required_field_is_rejected PASSED
tests/test_api.py::test_null_value_is_imputed PASSED
tests/test_api.py::test_incompatible_type_is_rejected PASSED
tests/test_api.py::test_out_of_range_value_is_rejected PASSED
tests/test_api.py::test_unavailable_model_returns_503 PASSED
tests/test_api.py::test_train_creates_new_version_and_keeps_previous PASSED
tests/test_api.py::test_next_version_id PASSED
9 passed
```

El mismo resultado se obtuvo en Windows (Python 3.14.0) y en Docker (`python:3.12-slim`).

| Prueba | Comprueba |
|---|---|
| `test_health_returns_200` | `/health` responde 200 con `status: ok` |
| `test_valid_record_returns_prediction_and_version` | Una entrada válida devuelve 200 con `prediccion` y `version` |
| `test_missing_required_field_is_rejected` | Sin `temp`, la respuesta es 422 y el mensaje nombra `temp` |
| `test_null_value_is_imputed` | `hum: null` se imputa y devuelve 200 |
| `test_incompatible_type_is_rejected` | `hr: "ocho"` devuelve 422 |
| `test_out_of_range_value_is_rejected` | `hr: 25` devuelve 422 |
| `test_unavailable_model_returns_503` | Una versión inexistente devuelve 503 y `/health` informa `model_loaded: false` |
| `test_train_creates_new_version_and_keeps_previous` | `/train` crea una versión nueva con sus tres artefactos y no modifica la anterior |
| `test_next_version_id` | El identificador nuevo es el siguiente `vN` libre |

---

## 9. Ejecución local sin Docker

Con el entorno virtual del paso anterior activo, desde la raíz del repositorio:

```bash
python -m uvicorn api.server:app --app-dir fase-3 --port 8000
```

- La API queda en `http://localhost:8000` y se prueba igual que en la sección 6.
- Se detiene con `Ctrl + C`.
- `api/main.py` añade `fase-2/src` a la ruta de importación, así que no hace falta definir `PYTHONPATH`.

Para usar otra versión:

```powershell
$env:MODEL_VERSION = "v1"; python -m uvicorn api.server:app --app-dir fase-3 --port 8000
```

---

## 10. Detener el contenedor y relación con la fase 4

La fase 4 **no es un servicio adicional**: es esta misma API con el registro de solicitudes habilitado, empaquetada en otra imagen (`bike-demand-monitor`). Ambas imágenes usan el puerto `8000`, por lo que **no pueden ejecutarse a la vez con el mismo puerto**. Si lo intenta, Docker responde `port is already allocated`.

Antes de iniciar la fase 4, detenga el contenedor de la fase 3:

```bash
docker stop bike-api
```

| Quiero... | Contenedor | Comando de inicio |
|---|---|---|
| Probar la API (fase 3) | `bike-api` (imagen `bike-demand-api`) | Sección 5 de este README |
| Probar la API registrando cada solicitud (fase 4) | `bike-monitor` (imagen `bike-demand-monitor`) | Sección 3 de [fase-4/README.md](../fase-4/README.md) |

Los dos contenedores comparten la misma carpeta de modelos (`fase-2/models`), así que una versión creada con `/train` en uno está disponible en el otro.

---

## 11. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `failed to connect to the docker API` | Docker Desktop no está en ejecución | Abrir Docker Desktop y esperar *Engine running* |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | Otro contenedor (por ejemplo el de la fase 4) ya usa el puerto 8000 | `docker ps` para identificarlo, `docker stop bike-monitor` y luego `docker rm -f bike-api` para eliminar el contenedor que quedó creado sin iniciar; después repetir el paso 4 |
| `Conflict. The container name "/bike-api" is already in use` | Quedó un contenedor anterior con ese nombre | `docker rm -f bike-api` y volver a iniciar |
| El navegador no carga `localhost:8000` justo al iniciar | El modelo aún se está cargando | Esperar unos segundos y revisar `docker logs bike-api` |
| `/health` muestra `model_loaded: false` y `/predict` responde 503 | La versión configurada no existe en `fase-2/models` o el volumen no se montó | Revisar el `-v` de modelos y el valor de `MODEL_VERSION` |
| `422` en una solicitud que parece correcta | Falta un campo, hay un texto en lugar de número o un valor fuera de rango | Leer `detail`: indica el campo exacto |
| `Invoke-RestMethod` muestra error rojo en PowerShell | Es la forma en que PowerShell presenta respuestas 4xx/5xx | El JSON de la API aparece dentro del mensaje |

---

## 12. Comportamiento de la API

- **`/train` es síncrono:** responde cuando los artefactos ya están guardados (unos 20–30 segundos con el conjunto completo). Se atiende un entrenamiento a la vez; una solicitud simultánea recibe `409`.
- **Versión fija por ejecución:** la versión activa se lee al iniciar la API. Para cambiarla se reinicia la API con la nueva configuración (sección 4).
- **Versiones inmutables:** ninguna operación modifica ni elimina versiones existentes.
- **Registro de solicitudes:** el registro de cada llamada a `/predict` y las métricas de operación se habilitan con la API de la fase 4 ([fase-4/README.md](../fase-4/README.md)).
