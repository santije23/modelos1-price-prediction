# Predicción de la demanda horaria de bicicletas compartidas - Fase 2

**Proyecto Integrador - Modelos y Simulación de Sistemas I - 2026 II**

**Integrantes:** Santiago Jiménez Escobar & Santiago Durán Carmona

---

## Tabla de contenido

1. [Problema y relación con la Fase 1](#1-problema-y-relaci%C3%B3n-con-la-fase-1)
2. [Cambios respecto a la Fase 1](#2-cambios-respecto-a-la-fase-1)
3. [Organización del código](#3-organizaci%C3%B3n-del-c%C3%B3digo)
4. [Preprocesamiento y reglas de datos](#4-preprocesamiento-y-reglas-de-datos)
5. [Datos](#5-datos)
6. [Ejecución local](#6-ejecuci%C3%B3n-local)
7. [Versiones del modelo](#7-versiones-del-modelo)
8. [Resultados](#8-resultados)
9. [Docker](#9-docker)
10. [Pruebas](#10-pruebas)
11. [Identificación de la entrega y fuentes](#11-identificaci%C3%B3n-de-la-entrega-y-fuentes)

---

## 1. Problema y relación con la Fase 1

Se predice **`cnt`**, el número de bicicletas alquiladas en una hora, a partir de variables de calendario y meteorológicas del sistema Capital Bikeshare (Washington D. C., 17.379 registros horarios de 2011 y 2012). Es un problema de **regresión**.

La Fase 1 (carpeta `fase-1/`, conservada sin cambios) desarrolló el análisis y el modelo en un notebook. La Fase 2 convierte ese trabajo en una aplicación ejecutable por línea de comandos:

- `scripts/train.py` entrena el modelo y guarda una **versión** identificable.
- `scripts/evaluate.py` evalúa una versión guardada sobre el conjunto de prueba.
- `src/bike_demand/models/predict.py` expone la función de inferencia, reutilizable sobre datos nuevos.

Se conservan el problema, el conjunto de datos y el enfoque de modelado: **Random Forest** como modelo.

## 2. Cambios respecto a la Fase 1

|Aspecto|Fase 1|Fase 2|
|---|---|---|
|Organización|Un notebook|Paquete `src/bike_demand/` + scripts con `argparse`|
|Configuración|Valores fijos dentro del notebook|`configs/config.yaml` centraliza rutas, semilla, `test_size`, features y parámetros del modelo|
|Validación de datos|No había|`data/validation.py` valida columnas y variable objetivo con mensajes claros|
|Fuga de información|Riesgo de incluir `casual` y `registered`|Se eliminan explícitamente (`leakage_columns` en config)|
|Variables cíclicas|Manual o ausente|`hr` y `mnth` con seno/coseno; flag `is_rush_hour`|
|Almacenamiento|`model_v1.joblib` + `metricas_v1.json` sueltos en `fase-1/`|`fase-2/models/model_v2.joblib` + `fase-2/reports/metrics_v2.json`, con un `Pipeline` completo (preprocesamiento + modelo)|
|Rutas|Detectadas dentro del notebook|Resueltas por `config.py` a rutas absolutas desde la raíz del repo|
|Tests|No había|10 tests con `pytest` (loader, validación, features, train/predict)|
|Ejecución|Manual en Jupyter|CLI reproducible + imagen Docker|

**Partición train/test.** Se mantiene la partición aleatoria de la Fase 1 (`train_test_split` con `random_state` fijo) para conservar comparabilidad con las métricas ya reportadas. El preprocesamiento se ajusta **solo con `X_train`**, por lo que no hay fuga de información entre train y test.

## 3. Organización del código

```
fase-2/
├── configs/
│   └── config.yaml                # parámetros: rutas, semilla, test_size, features, modelo
├── src/
│   └── bike_demand/
│       ├── __init__.py
│       ├── config.py              # carga y resuelve rutas del YAML
│       ├── data/
│       │   ├── loader.py          # load_data()
│       │   └── validation.py      # validate_schema(), validate_target()
│       ├── features/
│       │   └── build_features.py  # prepare_dataframe(), build_pipeline()
│       ├── models/
│       │   ├── train.py           # train_model()
│       │   └── predict.py         # load_model(), predict()
│       └── utils/
│           └── metrics.py         # compute_metrics()
├── scripts/
│   ├── train.py                   # CLI de entrenamiento
│   └── evaluate.py                # CLI de evaluación
├── tests/
│   ├── conftest.py                # expone src/ a pytest
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_model.py
├── Dockerfile
├── .dockerignore
└── README.md
```

### Principios de diseño

- `scripts/train.py` y `scripts/evaluate.py` son **orquestadores delgados**: solo parsean `--config` y llaman a funciones del paquete.
- Nada se ejecuta al importar y que todo está dentro de `main()` y `if __name__ == "__main__":`.
- No hay rutas absolutas hardcodeadas porque las rutas salen de `configs/config.yaml` y `config.py` las resuelve a absolutas desde la raíz del repo.
- Las transformaciones no se duplican, ya que `prepare_dataframe()` se usa igual en entrenamiento y en evaluación, y el `Pipeline` (preprocesamiento + modelo) se guarda como un único artefacto.

### Módulos

|Módulo|Responsabilidad|Funciones principales|
|---|---|---|
|`config.py`|Cargar el YAML y resolver rutas|`load_config`|
|`data/loader.py`|Leer el CSV crudo|`load_data`|
|`data/validation.py`|Validar esquema y target|`validate_schema`, `validate_target`|
|`features/build_features.py`|Encoding cíclico, flag rush hour, Pipeline|`add_cyclic_features`, `add_rush_hour_flag`, `prepare_dataframe`, `build_pipeline`|
|`models/train.py`|Entrenamiento y persistencia|`train_model`|
|`models/predict.py`|Carga e inferencia|`load_model`, `predict`|
|`utils/metrics.py`|Métricas|`compute_metrics`|

## 4. Preprocesamiento y reglas de datos

El preprocesamiento es un `ColumnTransformer` con `StandardScaler` sobre las columnas numéricas resultantes de `prepare_dataframe()`, dentro de un `Pipeline` junto con el `RandomForestRegressor`. Se ajusta **únicamente con `X_train`**.

### Reglas aplicadas antes del Pipeline (`prepare_dataframe`)

- Se eliminan las columnas de fuga: `casual`, `registered`, `cnt`.
- Se eliminan columnas sin valor predictivo: `instant`, `dteday`.
- `hr` y `mnth` se codifican como seno/coseno (`hr_sin`, `hr_cos`, `mnth_sin`, `mnth_cos`).
- Se añade `is_rush_hour` (1 si `hr ∈ {7, 8, 9, 17, 18, 19}` y `workingday == 1`).

### Reglas al predecir

- **Valores nulos:** el Pipeline actual **no** incluye imputador, porque el dataset por defecto no tiene nulos. Si llegan nulos, la transformación falla (ver limitación abajo).
- **Columna requerida ausente:** el `ColumnTransformer` falla al transformar y `predict()` propaga el error.
- **Columnas adicionales** (incluida `cnt`): se eliminan en `prepare_dataframe()`.
- **Orden de columnas:** no importa; `prepare_dataframe()` reordena y reconstruye.

> **Limitación conocida:** el Pipeline no incluye `SimpleImputer` porque el dataset con nulos inyectados de la Fase 1 (`bike_share_with_nans.csv`) no se usa en el entrenamiento por defecto. Si se entrena con ese CSV, hay que añadir el imputador (mediana en numéricas) a `build_pipeline()`.

## 5. Datos

- **Fuente original:** [Bike Sharing Dataset (Kaggle)](https://www.kaggle.com/datasets/amil09/bike-sharing-dataset/data?select=Bike+Share+hourly.csv).
- **Archivo usado por defecto:** `data/raw/bike_share_hourly.csv` (dataset original, sin nulos). Ruta definida en `configs/config.yaml`.
- **Archivo procesado de la Fase 1:** `data/processed/bike_share_with_nans.csv` (nulos MCAR inyectados con semilla 42). Disponible en el repo; **no** se usa por defecto.
- **No es necesario descargar nada de Kaggle** para entrenar: el CSV está versionado en el repositorio.

### Esquema del CSV de entrada

El dataset original tiene 17 columnas: `instant`, `dteday`, `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `casual`, `registered`, `cnt`. El entrenamiento usa todas excepto las eliminadas en `prepare_dataframe()`.

### Salida de `evaluate.py` y `metrics_v2.json`

JSON con tres claves: `mae`, `rmse`, `r2`.

## 6. Ejecución local

Terminal utilizada: **bash** en Linux / **PowerShell** en Windows. Todos los comandos se ejecutan desde la **raíz del repositorio** (`modelos1-price-prediction/`), porque las rutas de `config.yaml` son relativas a la raíz.

### 6.1 Entorno

**Linux (bash)**

```bash
python3 --version          # probado con Python 3.11 y 3.14
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si PowerShell bloquea la activación del entorno, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a intentarlo.

### 6.2 Entrenamiento

**Linux (bash)**

```bash
python fase-2/scripts/train.py --config fase-2/configs/config.yaml
```

**Windows (PowerShell)**

```powershell
python fase-2\scripts\train.py --config fase-2\configs\config.yaml
```

Salida esperada (ver [§8](#8-resultados)):

```
Entrenamiento completado. Métricas en test:
  mae: 24.5983
  rmse: 40.2846
  r2: 0.9488
```

Genera:

- `fase-2/models/model_v2.joblib` (Pipeline completo)
- `fase-2/reports/metrics_v2.json`

### 6.3 Evaluación

**Linux (bash)**

```bash
python fase-2/scripts/evaluate.py --config fase-2/configs/config.yaml
```

**Windows (PowerShell)**

```powershell
python fase-2\scripts\evaluate.py --config fase-2\configs\config.yaml
```

Salida esperada: el mismo JSON con `mae`, `rmse` y `r2` que `metrics_v2.json`. Como `evaluate.py` reusa el mismo `random_state` y el mismo split, los valores coinciden al decimal con los del entrenamiento.

### 6.4 Inferencia sobre datos nuevos

`predict()` está expuesto como función del paquete. Ejemplo mínimo:

**Linux (bash)**

```bash
python -c "
import sys; sys.path.insert(0, 'fase-2/src')
from bike_demand.config import load_config
from bike_demand.data.loader import load_data
from bike_demand.models.predict import predict
cfg = load_config('fase-2/configs/config.yaml')
df = load_data(cfg['data']['raw_path'])
print(predict(df.head(3), cfg))
"
```

**Windows (PowerShell)**

```powershell
python -c @"
import sys; sys.path.insert(0, 'fase-2/src')
from bike_demand.config import load_config
from bike_demand.data.loader import load_data
from bike_demand.models.predict import predict
cfg = load_config('fase-2/configs/config.yaml')
df = load_data(cfg['data']['raw_path'])
print(predict(df.head(3), cfg))
"@
```

Salida esperada (valores de la ejecución de referencia):

```
0    25.697872
1    34.503190
2    27.089444
```

### 6.5 Tests

**Linux (bash)**

```bash
pytest fase-2/tests/ -v
```

**Windows (PowerShell)**

```powershell
pytest fase-2\tests\ -v
```

Deben pasar los 10 tests. Ver [§10.1](#101-pruebas-automatizadas-con-pytest).

## 7. Versiones del modelo

Cada ejecución de `train.py` escribe en las rutas definidas en `config.yaml` (`output.model_path` y `output.metrics_path`). En la rama actual **no hay versionado por carpeta**: `model_v2.joblib` y `metrics_v2.json` se sobrescriben en cada ejecución. Si se necesita histórico, se cambia `output.model_path` en el YAML y se vuelve a ejecutar.

### Versión entregada

- `fase-2/models/model_v2.joblib` — Pipeline `ColumnTransformer(StandardScaler) + RandomForestRegressor`
- `fase-2/reports/metrics_v2.json` — métricas de la ejecución de referencia

### Artefactos generados

```
fase-2/
├── models/
│   └── model_v2.joblib     # ~67 MB; ignorado por .gitignore
└── reports/
    └── metrics_v2.json     # ignorado por .gitignore
```

## 8. Resultados

Métricas en el conjunto de prueba, semilla 42 (`train_test_split` con `test_size=0.2`).

|Modelo|MAE|RMSE|R²|
|---|---|---|---|
|Random Forest (Fase 2)|24.5983|40.2846|0.9488|

**Referencia de la Fase 1** (partición aleatoria, notebook): MAE ≈ 24.6, RMSE ≈ 40.3, R² ≈ 0.949. Las métricas de la Fase 2 son consistentes con las de la Fase 1 porque la partición y las features se mantienen.

### Limitaciones conocidas

- No se valida el rango de los valores de entrada (p. ej. `hr=30` se trata como valor numérico y produce predicciones sin sentido).
- El `Pipeline` no incluye `SimpleImputer`; si se entrena con `bike_share_with_nans.csv`, fallará al encontrar nulos.
- El Random Forest es no determinista por naturaleza; se fija `random_state` con la semilla de `config.yaml` para obtener resultados consistentes en el mismo entorno.
- El `.joblib` pesa ~67 MB y no se versiona. Se regenera con `train.py`.

## 9. Docker

La imagen empaqueta el código de la Fase 2 y ejecuta el entrenamiento. Los datos y artefactos se montan como volúmenes; no se copian dentro de la imagen.

### 9.1 Instalación de Docker

**Linux (Ubuntu/Debian, Fedora, Arch)**

- **Ubuntu/Debian:** sigue la instalación oficial desde el repositorio de Docker. Los pasos resumidos son: añadir la clave GPG, agregar el repositorio y ejecutar `sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`.
- **Fedora:** `sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin` (tras añadir el repositorio oficial de Docker).
- **Arch Linux:** `sudo pacman -S docker docker-buildx` y luego `sudo systemctl enable --now docker`.

Tras instalar, verifica con `sudo docker run hello-world`. Si quieres usar `docker` sin `sudo`, añade tu usuario al grupo `docker` y reinicia la sesión:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

> El grupo `docker` da privilegios equivalentes a root; úsalo solo en tu máquina de desarrollo.

**Windows**

Instala **Docker Desktop para Windows** desde la página oficial. Durante la instalación, asegúrate de que la opción **"Use WSL 2 based engine"** esté marcada. Si WSL 2 no está configurado, abre PowerShell como administrador, ejecuta `wsl --install` y reinicia el equipo. Al terminar, inicia Docker Desktop desde el menú de inicio y espera a que el icono de la ballena esté estable. Verifica en PowerShell con `docker run hello-world`.

### 9.2 Construcción

Desde la **raíz del repositorio**:

**Linux (bash)**

```bash
docker build -f fase-2/Dockerfile -t bike-demand-fase2 .
```

**Windows (PowerShell)**

```powershell
docker build -f fase-2\Dockerfile -t bike-demand-fase2 .
```

### 9.3 Entrenamiento en contenedor

`config.py` resuelve las rutas relativas contra la raíz del repositorio dentro del contenedor (`/app`). Por eso los volúmenes se montan en `/app/data`, `/app/models` y `/app/reports`, **no** dentro de `fase-2/`.

**Linux (bash)**

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/fase-2/models:/app/models" \
  -v "$(pwd)/fase-2/reports:/app/reports" \
  bike-demand-fase2
```

**Windows (PowerShell)**

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/fase-2/models:/app/models" `
  -v "${PWD}/fase-2/reports:/app/reports" `
  bike-demand-fase2
```

Salida esperada: las mismas métricas de la [§8](#8-resultados). Los archivos `model_v2.joblib` y `metrics_v2.json` aparecen en `fase-2/models/` y `fase-2/reports/` de tu máquina.

### 9.4 Tests dentro del contenedor

**Linux (bash)**

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  bike-demand-fase2 \
  python -m pytest tests/ -v
```

**Windows (PowerShell)**

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  bike-demand-fase2 `
  python -m pytest tests/ -v
```

### 9.5 Notas

- `.dockerignore` excluye `data/`, `models/`, `reports/`, `__pycache__/`, `.pytest_cache/` y `.git/`.
- El `Dockerfile` copia `requirements.txt` primero para aprovechar la caché de capas, y define `PYTHONPATH=/app/fase-2/src` para que `scripts/train.py` importe `bike_demand` sin variables adicionales.
- Si ejecutas Docker con `sudo` en Linux, los archivos generados en los volúmenes pertenecerán a `root`. Puedes cambiar el propietario con `sudo chown -R $USER:$USER fase-2/models fase-2/reports`.
- En Windows, si usas Git Bash en lugar de PowerShell, las rutas con `$(pwd)` pueden fallar por la conversión automática de rutas de MSYS. Usa PowerShell o define `MSYS_NO_PATHCONV=1`.

## 10. Pruebas

### 10.1 Pruebas automatizadas con pytest

**Linux (bash)**

```bash
pytest fase-2/tests/ -v
```

**Windows (PowerShell)**

```powershell
pytest fase-2\tests\ -v
```

Resultado esperado:

```
collected 10 items
test_data_loader.py::test_validate_schema_ok PASSED
test_data_loader.py::test_validate_schema_missing_column PASSED
test_data_loader.py::test_validate_target_ok PASSED
test_data_loader.py::test_validate_target_with_nan PASSED
test_data_loader.py::test_load_data_missing_file PASSED
test_features.py::test_add_cyclic_features PASSED
test_features.py::test_add_rush_hour_flag PASSED
test_features.py::test_prepare_dataframe_drops_leakage PASSED
test_model.py::test_train_model_creates_artifacts PASSED
test_model.py::test_predict_returns_series PASSED
===== 10 passed =====
```

Cobertura de los tests:

|Archivo|Tests|Qué comprueba|
|---|---|---|
|`test_data_loader.py`|5|Esquema válido, esquema con columna faltante, target válido, target con nulos, archivo inexistente|
|`test_features.py`|3|Encoding cíclico, flag rush hour, eliminación de columnas con fuga|
|`test_model.py`|2|Entrenamiento con dataset sintético de 200 filas (`tmp_path`); la inferencia devuelve una `Series` de longitud correcta|

Los tests de `test_model.py` no dependen de `data/raw/` ni del modelo ya entrenado: crean un CSV sintético en `tmp_path` y ejecutan el pipeline completo.

### 10.2 Pruebas manuales de `train.py` y `evaluate.py`

|#|Prueba|Resultado esperado|
|---|---|---|
|1|`train.py --config fase-2/configs/config.yaml`|Métricas MAE/RMSE/R² y creación de `model_v2.joblib` y `metrics_v2.json`|
|2|`evaluate.py --config fase-2/configs/config.yaml`|Las mismas métricas que `train.py`, al decimal|
|3|`pytest fase-2/tests/ -v`|10 tests en verde|
|4|Inferencia con `python -c "..."` ([§6.4](#64-inferencia-sobre-datos-nuevos))|3 predicciones no negativas|
|5|`docker build ...` y `docker run ...` ([§9](#9-docker))|Métricas dentro del contenedor iguales a las locales|

Las capturas de pantalla de las pruebas manuales se mantienen en `fase-2/img/`.

## 11. Fuentes consultadas y código externo adaptado

- **Dataset original:** [Bike Sharing Dataset (Kaggle)](https://www.kaggle.com/datasets/amil09/bike-sharing-dataset/data?select=Bike+Share+hourly.csv). Licencia de uso según Kaggle.
- **scikit-learn:** `Pipeline`, `ColumnTransformer`, `StandardScaler`, `RandomForestRegressor`, `train_test_split`, `mean_absolute_error`, `mean_squared_error`, `r2_score`. Documentación: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/).
- **joblib:** serialización de artefactos. Documentación: [https://joblib.readthedocs.io/](https://joblib.readthedocs.io/).
- **pytest:** ejecución de tests y fixture `tmp_path`. Documentación: [https://docs.pytest.org/](https://docs.pytest.org/).
- **PyYAML:** lectura de `config.yaml`. Documentación: [https://pyyaml.org/wiki/PyYAMLDocumentation](https://pyyaml.org/wiki/PyYAMLDocumentation).
- **Docker:** imagen `python:3.12-slim` y volúmenes bind-mount. Documentación: [https://docs.docker.com/](https://docs.docker.com/).
