# Fase 4 · Monitoreo

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II**

**Integrantes:** Santiago Jiménez Escobar & Santiago Durán Carmona

La fase 4 registra cada llamada a `POST /predict` de la API de la fase 3 y calcula dos métricas de operación: el **total de solicitudes** y el **porcentaje de solicitudes fallidas**. También identifica qué versión del modelo predice, cómo volver a la anterior y cuándo reentrenar.

## Índice

1. [Relación con la fase 3: ¿qué contenedor se usa?](#1-relación-con-la-fase-3-qué-contenedor-se-usa)
2. [Estructura](#2-estructura)
3. [Guía de prueba paso a paso con Docker](#3-guía-de-prueba-paso-a-paso-con-docker)
4. [Registro de solicitudes](#4-registro-de-solicitudes)
5. [Script de métricas de operación](#5-script-de-métricas-de-operación)
6. [Pruebas automatizadas (pytest)](#6-pruebas-automatizadas-pytest)
7. [Ejecución local sin Docker](#7-ejecución-local-sin-docker)
8. [Versiones del modelo](#8-versiones-del-modelo)
9. [Política de reentrenamiento](#9-política-de-reentrenamiento)
10. [Reporte](#10-reporte)

---

## 1. Relación con la fase 3: ¿qué contenedor se usa?

La fase 4 **no agrega un segundo servicio**: es la **misma API de la fase 3** (mismos endpoints `/health`, `/predict`, `/train` y `/docs`) con el registro de solicitudes habilitado. `fase-4/app.py` crea la API con `create_app(prediction_logger=CsvPredictionLogger(...))`, de modo que el código de la fase 3 no se modifica.

Por eso hay **dos imágenes y se ejecuta una a la vez**:

| Imagen | Contenedor | Puerto | ¿Registra solicitudes? | Documentación |
|---|---|---|---|---|
| `bike-demand-api` | `bike-api` | 8000 | No | [fase-3/README.md](../fase-3/README.md) |
| `bike-demand-monitor` | `bike-monitor` | 8000 | Sí, en `fase-4/logs/predictions_log.csv` | Este README |

Ambas publican el puerto `8000`, así que **antes de iniciar la de la fase 4 hay que detener la de la fase 3**. Si no, Docker responde `Bind for 0.0.0.0:8000 failed: port is already allocated` y deja creado un contenedor `bike-monitor` sin iniciar, que se elimina con `docker rm -f bike-monitor`. Para evitarlo, detenga primero la API de la fase 3:

```bash
docker stop bike-api
```

Para comprobar que no queda ningún contenedor en el puerto 8000:

```bash
docker ps
```

Si se desea tener las dos en ejecución a la vez, la de la fase 4 puede publicarse en otro puerto del equipo con `-p 8001:8000`. En ese caso, todas las URL de este README cambian a `http://localhost:8001` (y el cliente se usa con `--url http://localhost:8001`).

---

## 2. Estructura

```text
fase-4/
├── app.py                     # API de la fase 3 con el registro habilitado
├── monitoring/
│   └── logger.py              # CsvPredictionLogger: una fila por llamada a /predict
├── scripts/
│   └── monitor.py             # total de solicitudes y % de fallos
├── logs/
│   ├── predictions_log.csv        # registro vivo; se crea con la primera solicitud (no se versiona)
│   └── sample_predictions_log.csv # muestra entregada: registro de la ejecución del reporte
├── tests/test_monitoring.py
├── Dockerfile
└── README.md
```

---

## 3. Guía de prueba paso a paso con Docker

Todos los comandos se ejecutan desde la **raíz del repositorio**. Los requisitos son los mismos de la fase 3 (Docker en ejecución; Python solo para el cliente y el script de métricas). Ver la [sección 5, paso 1, de la fase 3](../fase-3/README.md#paso-1-requisitos).

### Paso 1. Detener la API de la fase 3 (si está en ejecución)

```bash
docker stop bike-api
```

Si responde `No such container`, no había nada en ejecución y se puede continuar.

Si en un intento anterior apareció `port is already allocated` o `container name "/bike-monitor" is already in use`, elimine también el contenedor que quedó creado:

```bash
docker rm -f bike-monitor
```

### Paso 2. Construir la imagen de la fase 4

```bash
docker build -f fase-4/Dockerfile -t bike-demand-monitor .
```

Para que las versiones creadas con `/train` registren el commit en su `metadata.json`, se añade `--build-arg GIT_COMMIT=$(git rev-parse HEAD)` al comando anterior.

### Paso 3. Partir de un registro vacío (para reproducir el reporte)

El registro conserva el historial entre ejecuciones. Para repetir el ensayo del reporte desde cero, elimine el registro vivo; la muestra entregada no se toca:

```powershell
Remove-Item fase-4/logs/predictions_log.csv -ErrorAction SilentlyContinue   # Windows
```

```bash
rm -f fase-4/logs/predictions_log.csv                                         # Linux
```

Compruebe que el script informa un registro vacío:

```bash
python fase-4/scripts/monitor.py
```

```text
Total de solicitudes: 0
Solicitudes fallidas (4xx/5xx): 0
Porcentaje de fallos: sin solicitudes
```

### Paso 4. Iniciar la API con el registro habilitado

**Windows (PowerShell):**

```powershell
docker run -d --rm -p 8000:8000 `
  -v "${PWD}/fase-2/models:/app/fase-2/models" `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/fase-4/logs:/app/fase-4/logs" `
  --name bike-monitor bike-demand-monitor
```

**Linux:**

```bash
docker run -d --rm -p 8000:8000 --user "$(id -u):$(id -g)" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/fase-4/logs:/app/fase-4/logs" \
  --name bike-monitor bike-demand-monitor
```

| Opción | Efecto |
|---|---|
| `-v .../fase-2/models:...` | Modelos `v1`, `v2` y los creados con `/train` (los mismos de la fase 3) |
| `-v .../data:...:ro` | Datos para `/train` (solo lectura) |
| `-v .../fase-4/logs:/app/fase-4/logs` | **Registro persistente.** El CSV se escribe en `fase-4/logs/predictions_log.csv` del equipo y se conserva al detener el contenedor |

Compruebe que arrancó:

```bash
docker logs bike-monitor
```

Debe aparecer `Uvicorn running on http://0.0.0.0:8000`. Luego abra `http://localhost:8000/health` en el navegador. Debe mostrar `{"status":"ok","model_version":"v2","model_loaded":true}`.

### Paso 5. Enviar dos solicitudes válidas y una inválida

Elija **una** de las tres opciones.

**Opción A: cliente de Python:**

```bash
python fase-3/client/predict_client.py --skip-health --row 1
python fase-3/client/predict_client.py --skip-health --row 3
python fase-3/client/predict_client.py --skip-health --row 2 --drop temp
```

**Opción B: Postman:** importe `fase-3/client/bike_demand_api.postman_collection.json` (ver [fase 3, sección 6.2](../fase-3/README.md#62-con-postman)) y envíe en orden:

1. **2. POST /predict - válida (fila 1 del CSV)** → `200`
2. **3. POST /predict - válida (fila 3 del CSV)** → `200`
3. **4. POST /predict - falta temp (inválida)** → `422`

**Opción C: navegador:** abra `http://localhost:8000/docs`, despliegue **POST /predict**, pulse **Try it out** y luego:

1. Pulse **Execute** con el ejemplo precargado → `200`.
2. Cambie `"hr": 8` por `"hr": 17` y pulse **Execute** → `200`.
3. Borre la línea de `temp` (sin dejar una coma sobrante) y pulse **Execute** → `422`.

> `GET /health` y `POST /train` no se registran: el registro mide solo las solicitudes de predicción, como piden las especificaciones.

### Paso 6. Revisar el registro y calcular las métricas

```powershell
Get-Content fase-4/logs/predictions_log.csv     # Windows
```

```bash
cat fase-4/logs/predictions_log.csv             # Linux
```

```bash
python fase-4/scripts/monitor.py
```

Resultado esperado: 3 filas en el CSV, y el script muestra `Total de solicitudes: 3` y `Porcentaje de fallos: 33,3 %`. Compare cada fila con la respuesta recibida en el paso 5.

### Paso 7. Detener el contenedor

```bash
docker stop bike-monitor
```

El registro permanece en `fase-4/logs/predictions_log.csv`. Si vuelve a iniciar el contenedor, las solicitudes nuevas se agregan al final del mismo archivo.

---

## 4. Registro de solicitudes

Cada llamada a `/predict` agrega **una fila** al CSV, incluidas las rechazadas por validación:

| Columna | Contenido | En un error |
|---|---|---|
| `fecha_hora` | Fecha y hora ISO 8601 en que se respondió (UTC dentro de Docker) | Siempre presente |
| `estado_http` | Código HTTP devuelto (`200`, `422`, `503`, ...) | Siempre presente |
| `version` | Versión del modelo que produjo la respuesta | Vacía si no se usó modelo |
| `prediccion` | Valor estimado | Vacía |

¿Dónde se escribe cada evento? (`fase-3/api/main.py`):

- **`200` y `503`:** dentro de la ruta `/predict`, cuando ya se conoce el resultado. En `200` se toma la versión que generó la predicción.
- **`422`:** FastAPI rechaza la entrada antes de llamar a la ruta. Por eso se registra en el manejador de errores de validación (`RequestValidationError`), solo cuando la ruta es `/predict`.
- La prueba `test_two_valid_and_one_invalid_give_three_rows_and_33_percent` comprueba que cada intento genera exactamente una fila.

Persistencia (`fase-4/monitoring/logger.py`):

- El archivo se abre en modo *append*.
- La cabecera se escribe solo al crearlo.
- La escritura está protegida con un candado para solicitudes simultáneas.

La ruta por defecto es `fase-4/logs/predictions_log.csv`. Se cambia con la variable de entorno `PREDICTION_LOG_PATH`.

---

## 5. Script de métricas de operación

```bash
python fase-4/scripts/monitor.py                                               # registro vivo
python fase-4/scripts/monitor.py --log fase-4/logs/sample_predictions_log.csv  # muestra entregada
```

Cálculo:

1. **Total de solicitudes:** número de filas de datos del CSV.
2. **Solicitudes fallidas:** filas con `estado_http` entre 400 y 599.
3. **Porcentaje de fallos:** fallidas ÷ total × 100, con un decimal.
4. Si el archivo no existe o no tiene filas, muestra `Total de solicitudes: 0` y `Porcentaje de fallos: sin solicitudes`.

Este porcentaje describe el funcionamiento del servicio, no la calidad del modelo. La calidad predictiva se evalúa con los alquileres reales, según la [política de reentrenamiento](#9-política-de-reentrenamiento).

---

## 6. Pruebas automatizadas (pytest)

**Dentro de Docker:**

```bash
docker run --rm bike-demand-monitor python -m pytest -v tests/
```

**En el equipo** (con el entorno virtual de la [fase 3, sección 8](../fase-3/README.md#en-el-equipo) activo):

```bash
cd fase-4
python -m pytest -v
cd ..
```

Resultado obtenido (Windows con Python 3.14.0 y Docker con `python:3.12-slim`):

```text
tests/test_monitoring.py::test_two_valid_and_one_invalid_give_three_rows_and_33_percent PASSED
tests/test_monitoring.py::test_log_is_kept_between_app_restarts PASSED
tests/test_monitoring.py::test_unavailable_model_is_logged_as_503_without_version PASSED
tests/test_monitoring.py::test_empty_or_missing_log_reports_no_requests PASSED
tests/test_monitoring.py::test_monitor_cli_output PASSED
5 passed
```

| Prueba | Comprueba |
|---|---|
| `test_two_valid_and_one_invalid_give_three_rows_and_33_percent` | 2 válidas + 1 sin `temp` generan 3 filas (200, 200, 422), sin versión ni predicción en el error, y 33,3 % de fallos |
| `test_log_is_kept_between_app_restarts` | El registro se conserva al reiniciar la API |
| `test_unavailable_model_is_logged_as_503_without_version` | Un 503 se registra sin versión ni predicción |
| `test_empty_or_missing_log_reports_no_requests` | Registro vacío o inexistente → total 0 y «sin solicitudes» |
| `test_monitor_cli_output` | La salida del script por línea de comandos |

---

## 7. Ejecución local sin Docker

Con el entorno virtual activo y desde la raíz del repositorio:

```bash
python -m uvicorn app:app --app-dir fase-4 --port 8000
```

- El registro se escribe en `fase-4/logs/predictions_log.csv`.
- Las pruebas de los pasos 5 y 6 de la sección 3 se realizan igual.
- Se detiene con `Ctrl + C`.

---

## 8. Versiones del modelo

| Versión | Carpeta | Configuración | MAE | RMSE | R² | Estado |
|---|---|---|---:|---:|---:|---|
| `v1` | `fase-2/models/v1/` | `fase-2/configs/config.yaml` | 26,95 | 45,20 | 0,9345 | Anterior |
| `v2` | `fase-2/models/v2/` | `fase-2/configs/versions/v2.yaml` | 26,88 | 45,02 | 0,9350 | **Activa** |

- Las dos versiones están publicadas en el repositorio, cada una con `modelo.joblib`, `metrics.json` y `metadata.json`.
- `POST /train` crea versiones adicionales (`v3`, `v4`, ...) en la misma carpeta, sin modificar estas.

**Versión que usa la API:** `v2`, definida en `fase-3/configs/api.yaml` (`model_version: v2`). Se confirma de tres formas:

- en `GET /health` (`"model_version": "v2"`);
- en cada respuesta de `/predict`;
- en la columna `version` del registro.

**Volver a la versión anterior (`v1`):**

- **Opción 1, sin reconstruir la imagen:** detener el contenedor y volver a iniciarlo añadiendo `-e MODEL_VERSION=v1` al comando `docker run` del paso 4.

  ```powershell
  docker stop bike-monitor
  docker run -d --rm -p 8000:8000 -e MODEL_VERSION=v1 `
    -v "${PWD}/fase-2/models:/app/fase-2/models" `
    -v "${PWD}/data:/app/data:ro" `
    -v "${PWD}/fase-4/logs:/app/fase-4/logs" `
    --name bike-monitor bike-demand-monitor
  ```

- **Opción 2, cambio permanente:** editar `model_version: v1` en `fase-3/configs/api.yaml` y reconstruir la imagen (paso 2).

En ambos casos, `GET /health` mostrará `"model_version": "v1"` y las filas nuevas del registro tendrán `v1`.

---

## 9. Política de reentrenamiento

Cada mes, o cuando se acumulen 720 horas nuevas (30 días) con el número real de alquileres `cnt`, el equipo revisa el modelo. Un integrante agrega esos datos al CSV de entrenamiento y ejecuta `POST /train` para generar una versión candidata. La candidata se activa en `fase-3/configs/api.yaml` solo si, sobre los mismos días reservados (`GroupShuffleSplit` por `dteday`, semilla 42), reduce el MAE de la versión activa al menos en un 2 % sin aumentar el RMSE; en caso contrario se mantiene la versión activa.

---

## 10. Reporte

**Ejecución:** API de la fase 4 en Docker (contenedor `bike-monitor`, imagen `python:3.12-slim`, versión activa `v2`), partiendo de un registro vacío. Se siguieron los pasos 3 a 6 de la sección 3 con el cliente de Python.

**Solicitudes enviadas:**

| # | Comando | Contenido | Respuesta |
|---|---|---|---|
| 1 | `python fase-3/client/predict_client.py --skip-health --row 1` | Fila 1 de `data/example_prediction.csv` (junio, 8:00, día laboral, despejado) | `200` `{"prediccion": 592.6558988095239, "version": "v2"}` |
| 2 | `python fase-3/client/predict_client.py --skip-health --row 3` | Fila 3 (julio, 17:00, día laboral, nublado) | `200` `{"prediccion": 781.2367182409821, "version": "v2"}` |
| 3 | `python fase-3/client/predict_client.py --skip-health --row 2 --drop temp` | Fila 2 sin la variable `temp` | `422` `{"detail": "Faltan campos obligatorios: temp", ...}` |

**Registro obtenido** (copiado en `fase-4/logs/sample_predictions_log.csv`):

```csv
fecha_hora,estado_http,version,prediccion
2026-11-22T02:32:44,200,v2,592.6559
2026-11-22T02:32:44,200,v2,781.2367
2026-11-22T02:32:44,422,,
```

**Indicadores** (`python fase-4/scripts/monitor.py`):

```text
Total de solicitudes: 3
Solicitudes fallidas (4xx/5xx): 1
Porcentaje de fallos: 33,3 %
```

| Indicador | Valor |
|---|---|
| Total de solicitudes | 3 |
| Porcentaje de solicitudes fallidas | 33,3 % (1 de 3) |

Antes de enviar las solicitudes, el mismo script mostró `Total de solicitudes: 0` y `Porcentaje de fallos: sin solicitudes`.

**Interpretación:** la API procesó dos registros con la versión `v2` y rechazó el tercero porque faltaba la temperatura (`temp`), así que la tasa de fallos fue 1/3 (33,3 %). Ese rechazo era el resultado esperado de la prueba de validación. Las tres filas del registro coinciden con los conteos del script.

**Versiones:** `v2` es la versión activa y `v1` la anterior. Ambas están en `fase-2/models/`, y la sección 8 explica cómo volver a `v1`.
