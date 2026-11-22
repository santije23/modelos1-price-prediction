# Fase 4 · Monitoreo

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II**

**Integrantes:** Santiago Jiménez Escobar & Santiago Durán Carmona

La fase 4 registra cada llamada a `POST /predict` de la API de la fase 3 y calcula dos métricas de operación: el total de solicitudes y el porcentaje de solicitudes fallidas. La API es la misma de la fase 3. `fase-4/app.py` la crea con `create_app(prediction_logger=...)`, de modo que el código de la fase 3 no cambia.

## 1. Estructura

```text
fase-4/
├── app.py                     # API de fase 3 con el registro habilitado
├── monitoring/logger.py       # CsvPredictionLogger: una fila por llamada a /predict
├── scripts/monitor.py         # total de solicitudes y % de fallos
├── logs/
│   ├── predictions_log.csv        # registro vivo (se crea al arrancar; ignorado por Git)
│   └── sample_predictions_log.csv # muestra entregada (ejecución del reporte)
├── tests/test_monitoring.py
└── Dockerfile
```

## 2. Registro de solicitudes

Cada llamada a `/predict` agrega **una fila** al CSV, incluidas las rechazadas por validación:

| Columna | Contenido |
|---|---|
| `fecha_hora` | Fecha y hora ISO 8601 de la respuesta (UTC dentro de Docker) |
| `estado_http` | Código devuelto (`200`, `422`, `503`, ...) |
| `version` | Versión del modelo que respondió; vacía si no se usó modelo |
| `prediccion` | Valor estimado; vacía en un error |

¿Dónde se registra cada caso?

- **200 y 503:** dentro de la propia ruta `/predict`, cuando ya se conoce el resultado.
- **422:** la validación de FastAPI rechaza la entrada antes de llamar a la ruta. Por eso se registra en el manejador de `RequestValidationError`, solo si la ruta es `/predict`.

Las pruebas comprueban que cada intento produce exactamente una fila.

El archivo se abre en modo *append* y la cabecera se escribe solo al crearlo, así que el historial se conserva entre ejecuciones. La ruta por defecto es `fase-4/logs/predictions_log.csv` y se puede cambiar con la variable de entorno `PREDICTION_LOG_PATH`.

## 3. Ejecución

Desde la **raíz del repositorio**.

### Docker

```bash
docker build -f fase-4/Dockerfile -t bike-demand-monitor --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
```

**Windows (PowerShell):**

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}/fase-2/models:/app/fase-2/models" `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/fase-4/logs:/app/fase-4/logs" `
  --name bike-monitor bike-demand-monitor
```

**Linux:**

```bash
docker run --rm -p 8000:8000 --user "$(id -u):$(id -g)" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/fase-4/logs:/app/fase-4/logs" \
  --name bike-monitor bike-demand-monitor
```

El registro persistente queda en `fase-4/logs/predictions_log.csv` del equipo, porque esa carpeta está montada en el contenedor.

### Local

```powershell
python -m pip install -r fase-3/requirements.txt
python -m uvicorn app:app --app-dir fase-4 --port 8000
```

### Métricas de operación

```bash
python fase-4/scripts/monitor.py                                          # registro por defecto
python fase-4/scripts/monitor.py --log fase-4/logs/sample_predictions_log.csv
```

Se consideran fallidas las respuestas con código entre 400 y 599. Si el registro no existe o no tiene filas, el script muestra `Total de solicitudes: 0` y `Porcentaje de fallos: sin solicitudes`.

### Pruebas

```bash
cd fase-4
python -m pytest -v
```

Resultado: `5 passed` tanto en local (Python 3.14.0) como en Docker (`docker run --rm bike-demand-monitor python -m pytest -v tests/`).

## 4. Versiones del modelo

| Versión | Carpeta | Configuración | MAE | RMSE | R² | Estado |
|---|---|---|---:|---:|---:|---|
| `v1` | `fase-2/models/v1/` | `fase-2/configs/config.yaml` | 26,95 | 45,20 | 0,9345 | Anterior |
| `v2` | `fase-2/models/v2/` | `fase-2/configs/versions/v2.yaml` | 26,88 | 45,02 | 0,9350 | **Activa** |
| `v3` | `fase-2/models/v3/` | `fase-2/configs/config.yaml` vía `POST /train` | 26,95 | 45,20 | 0,9345 | Candidata de la prueba de `/train` (se genera al ejecutar `/train`; no está versionada en Git) |

- **Versión que usa la API:** `model_version: v2` en `fase-3/configs/api.yaml`. Se puede confirmar con `GET /health` y en la columna `version` del registro.
- **Volver a la anterior:** cambiar `model_version: v1` en `fase-3/configs/api.yaml` y reiniciar, o arrancar con `-e MODEL_VERSION=v1` (Docker) o `$env:MODEL_VERSION = "v1"` (PowerShell). No hace falta modificar código ni mover archivos, porque las versiones nunca se sobrescriben.

## 5. Política de reentrenamiento

1. **Cuándo revisar el modelo:** cada mes, o antes si se acumulan 720 horas nuevas (unos 30 días) con el conteo real `cnt` registrado. También se revisa la operación si la tasa de fallos del registro supera el 5 % en una semana, aunque esos fallos normalmente indican datos de entrada mal formados y se corrigen en quien envía la solicitud, no en el modelo.
2. **Quién ejecuta `/train`:** uno de los integrantes del equipo, una vez incorporados los datos nuevos al CSV de entrenamiento. La versión resultante queda como candidata.
3. **Cómo se decide si se activa:** se comparan la candidata y la versión activa con el **MAE** de la fase 2 sobre los mismos días reservados (`GroupShuffleSplit` por `dteday`, semilla 42). La candidata se activa (`model_version` en `api.yaml`) solo si reduce el MAE en al menos un 2 % sin aumentar el RMSE. Si no, se conserva la versión activa.

## 6. Reporte

**Ejecución:** API de la fase 4 en Docker (`bike-demand-monitor`, versión activa `v2`), registro vacío al iniciar. Se enviaron tres solicitudes con el cliente de la fase 3:

| # | Comando | Contenido | Respuesta |
|---|---|---|---|
| 1 | `python fase-3/client/predict_client.py --skip-health --row 1` | Fila 1 de `data/example_prediction.csv` (junio, 8:00, día laboral) | `200` `{"prediccion": 592.6558988095239, "version": "v2"}` |
| 2 | `python fase-3/client/predict_client.py --skip-health --row 3` | Fila 3 (julio, 17:00, nublado) | `200` `{"prediccion": 781.2367182409821, "version": "v2"}` |
| 3 | `python fase-3/client/predict_client.py --skip-health --row 2 --drop temp` | Fila 2 sin `temp` | `422` `{"detail": "Faltan campos obligatorios: temp", ...}` |

**Registro obtenido** (`fase-4/logs/sample_predictions_log.csv`):

```csv
fecha_hora,estado_http,version,prediccion
2026-11-22T02:32:44,200,v2,592.6559
2026-11-22T02:32:44,200,v2,781.2367
2026-11-22T02:32:44,422,,
```

**Salida del script de monitoreo:**

```text
> python fase-4/scripts/monitor.py
Total de solicitudes: 3
Solicitudes fallidas (4xx/5xx): 1
Porcentaje de fallos: 33,3 %
```

Antes de enviar las solicitudes, el mismo script mostraba `Total de solicitudes: 0` y `Porcentaje de fallos: sin solicitudes`.

**Interpretación:** la API procesó dos registros con la versión `v2` y rechazó el tercero porque faltaba la temperatura (`temp`). La tasa de fallos fue 1/3 (33,3 %). El rechazo era el resultado esperado de la prueba de validación y no indica un problema del modelo. Las tres filas del registro coinciden con los conteos del script. El porcentaje describe el funcionamiento del servicio; medir la calidad predictiva requiere conocer los alquileres reales de esas horas, como establece la política de la sección 5.
