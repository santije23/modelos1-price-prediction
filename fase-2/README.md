# Fase 2 · Entrenamiento y predicción versionados

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II**

**Integrantes:** Santiago Jiménez Escobar & Santiago Durán Carmona

## Índice

1. [Objetivo](#1-objetivo)
2. [Datos y variables](#2-datos-y-variables)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Requisitos](#4-requisitos)
5. [Instalación local](#5-instalación-local)
   - [Windows](#instalación-en-windows)
   - [Linux](#instalación-en-linux)
6. [Configuración del entrenamiento](#6-configuración-del-entrenamiento)
7. [Entrenamiento de modelos](#7-entrenamiento-de-modelos)
8. [Versionamiento y almacenamiento](#8-versionamiento-y-almacenamiento)
9. [Predicción mediante archivos CSV](#9-predicción-mediante-archivos-csv)
10. [Pruebas automatizadas](#10-pruebas-automatizadas)
11. [Ejecución con Docker](#11-ejecución-con-docker)
    - [Construcción de la imagen](#construcción-de-la-imagen)
    - [Pruebas dentro del contenedor](#pruebas-dentro-del-contenedor)
    - [Entrenamiento dentro de Docker](#entrenamiento-dentro-de-docker)
    - [Predicción dentro de Docker](#predicción-dentro-de-docker)
    - [Persistencia de archivos](#persistencia-de-archivos)
12. [Resultados de las pruebas](#12-resultados-de-las-pruebas)
13. [Limitaciones](#13-limitaciones)
14. [Trabajo posterior](#14-trabajo-posterior)

---

## 1. Objetivo

La segunda fase del proyecto desarrolla un flujo reproducible de entrenamiento, evaluación, almacenamiento y predicción para el modelo de demanda de bicicletas compartidas construido durante la primera fase.

El objetivo es separar las responsabilidades del proceso de Machine Learning en componentes independientes que puedan ejecutarse mediante scripts, reutilizarse desde otros módulos y ejecutarse tanto en un entorno local como dentro de un contenedor Docker.

Se implementan las siguientes funcionalidades:

- Carga y validación de los datos.
- Preparación de variables predictoras.
- Ingeniería de características.
- Tratamiento de valores faltantes.
- Entrenamiento de modelos de regresión.
- Evaluación mediante métricas de desempeño.
- Almacenamiento de modelos y metadatos.
- Identificación de modelos mediante versiones independientes.
- Generación de predicciones a partir de archivos CSV.
- Pruebas automatizadas con `pytest`.
- Ejecución reproducible mediante Docker.

## 2. Datos y variables

Se utiliza el conjunto de datos de alquileres horarios de bicicletas compartidas de Capital Bikeshare, correspondiente a Washington D. C.

El conjunto de datos contiene **17.379 registros horarios**, correspondientes al período 2011–2012.

El archivo procesado utilizado para la fase 2 es:

`data/processed/bike_share_with_nans.csv`

La variable objetivo es `cnt`, que representa el número total de bicicletas alquiladas durante una hora.

### Variables predictoras

El modelo utiliza 11 variables de entrada:

| Variable | Tipo | Descripción |
|---|---|---|
| `season` | Numérico | Estación del año (1–4). |
| `yr` | Numérico | Año codificado (0/1). |
| `mnth` | Numérico | Mes del año (1–12). |
| `hr` | Numérico | Hora del día (0–23). |
| `holiday` | Numérico | Indicador de día festivo. |
| `weekday` | Numérico | Día de la semana (0–6). |
| `workingday` | Numérico | Indicador de día laboral. |
| `weathersit` | Numérico | Condición climática. |
| `temp` | Numérico | Temperatura normalizada. |
| `hum` | Numérico | Humedad normalizada. |
| `windspeed` | Numérico | Velocidad del viento normalizada. |

### Variables excluidas

| Variable | Motivo |
|---|---|
| `instant` | Identificador secuencial del registro. |
| `dteday` | Fecha original; se utiliza para agrupar la partición cuando corresponde, no como predictor. |
| `casual` | Componente de la variable objetivo. |
| `registered` | Componente de la variable objetivo. |
| `atemp` | Variable altamente correlacionada con `temp`. |
| `cnt` | Variable que se desea predecir. |

Las variables `casual` y `registered` se excluyen para evitar fuga de información (*data leakage*), ya que su suma corresponde directamente a `cnt`.

### Valores faltantes

El conjunto procesado incorpora valores faltantes simulados en variables meteorológicas. Esto permite comprobar que el sistema puede preparar los datos y generar predicciones incluso cuando algunos valores de entrada no están disponibles.

La imputación se integra en el pipeline de Scikit-Learn para que los parámetros correspondientes se aprendan durante el entrenamiento y se reutilicen al predecir.

### Partición de los datos

La evaluación utiliza una partición de entrenamiento y prueba del 80 % y 20 %, respectivamente.

Cuando se utiliza `GroupShuffleSplit`, la agrupación por `dteday` evita repartir registros de un mismo día entre ambos conjuntos. Esto permite evaluar el modelo con días que no participaron en el entrenamiento.

## 3. Estructura del proyecto

La estructura relevante de la fase 2 es:

```text
fase-2/
├── configs/
│   ├── config.yaml
│   └── versions/
│       └── v2.yaml
│
├── models/
│   ├── v1/
│   │   ├── modelo.joblib
│   │   ├── metrics.json
│   │   └── metadata.json
│   └── v2/
│       ├── modelo.joblib
│       ├── metrics.json
│       └── metadata.json
│
├── outputs/
│   └── predictions.csv
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── src/
│   └── bike_demand/
│       ├── config.py
│       ├── data/
│       ├── features/
│       ├── models/
│       └── utils/
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_model.py
│
└── Dockerfile
```

Los directorios de modelos y resultados se generan durante la ejecución cuando todavía no existen.

Los archivos de configuración definen los parámetros utilizados por cada entrenamiento. Los scripts ejecutables se encuentran en `scripts/`, mientras que la lógica reutilizable se organiza dentro de `src/bike_demand/`.

## 4. Requisitos

Para ejecutar la fase 2 localmente se requiere:

- Python 3.12 o superior.
- pip.
- Git (opcional, para clonar el repositorio).
- Docker Engine o Docker Desktop, para ejecutar las pruebas en contenedores.

Las dependencias de Python se encuentran declaradas en el archivo `requirements.txt` de la raíz del repositorio.

La imagen Docker utiliza Python 3.12, lo que permite comprobar el funcionamiento del proyecto en un entorno independiente de la versión de Python instalada localmente.

## 5. Instalación local

Los siguientes procedimientos se realizan desde la raíz del repositorio.

### Instalación en Windows

Se recomienda utilizar PowerShell.

#### 1. Crear el entorno virtual

```powershell
py -m venv .venv
```

#### 2. Activar el entorno

```powershell
.venv\Scripts\Activate.ps1
```

Si PowerShell impide ejecutar el script de activación, puede habilitarse la ejecución de scripts locales para el usuario actual:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Después se vuelve a ejecutar el comando de activación.

#### 3. Instalar las dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 4. Configurar la ruta de importación

Antes de ejecutar los scripts:

```powershell
$env:PYTHONPATH = "fase-2/src"
```

Esta variable permite importar el paquete `bike_demand` desde la carpeta `src`.

### Instalación en Linux

#### 1. Crear el entorno virtual

```bash
python3 -m venv .venv
```

#### 2. Activar el entorno

```bash
source .venv/bin/activate
```

#### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 4. Configurar la ruta de importación

La variable puede establecerse directamente en cada comando:

```bash
PYTHONPATH=fase-2/src
```

No es necesario exportarla permanentemente.

## 6. Configuración del entrenamiento

Los parámetros de entrenamiento se definen mediante archivos YAML.

Configuración principal:

`fase-2/configs/config.yaml`

Configuración alternativa:

`fase-2/configs/versions/v2.yaml`

Estos archivos permiten definir, entre otros aspectos:

- Nombre del proyecto y semilla aleatoria.
- Rutas de los datos.
- Variable objetivo.
- Columnas excluidas.
- Características utilizadas.
- Parámetros del modelo.
- Directorios de almacenamiento.

El uso de configuraciones independientes permite entrenar y comparar versiones sin modificar directamente el código de los scripts.

Los archivos de configuración deben ejecutarse desde la raíz del repositorio para que las rutas relativas a los datos y los artefactos se resuelvan correctamente.

## 7. Entrenamiento de modelos

El entrenamiento se ejecuta mediante:

`scripts/train.py`

El script recibe un archivo de configuración y un identificador de versión.

### Entrenar la versión v1

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "fase-2/src"

python fase-2/scripts/train.py `
  --config fase-2/configs/config.yaml `
  --version v1
```

**Linux:**

```bash
PYTHONPATH=fase-2/src python fase-2/scripts/train.py \
  --config fase-2/configs/config.yaml \
  --version v1
```

### Entrenar la versión v2

**Windows (PowerShell):**

```powershell
python fase-2/scripts/train.py `
  --config fase-2/configs/versions/v2.yaml `
  --version v2
```

**Linux:**

```bash
PYTHONPATH=fase-2/src python fase-2/scripts/train.py \
  --config fase-2/configs/versions/v2.yaml \
  --version v2
```

Al finalizar, el programa informa:

- Identificador de la versión entrenada.
- Directorio donde se almacenaron los artefactos.
- MAE.
- RMSE.
- R².

Los valores de las métricas dependen de la configuración utilizada y del conjunto de datos.

## 8. Versionamiento y almacenamiento

Cada entrenamiento genera un directorio independiente para evitar sobrescribir modelos anteriores.

Por ejemplo:

```text
fase-2/models/
├── v1/
│   ├── modelo.joblib
│   ├── metrics.json
│   └── metadata.json
│
└── v2/
    ├── modelo.joblib
    ├── metrics.json
    └── metadata.json
```

Cada directorio contiene tres artefactos:

| Archivo | Contenido |
|---|---|
| `modelo.joblib` | Pipeline entrenado y serializado. |
| `metrics.json` | Métricas de evaluación del modelo. |
| `metadata.json` | Información de la versión y del entrenamiento. |

### Protección contra sobrescritura

El sistema rechaza un entrenamiento cuando el identificador de versión ya existe.

Por ejemplo:

```bash
PYTHONPATH=fase-2/src python fase-2/scripts/train.py \
  --config fase-2/configs/config.yaml \
  --version v1
```

Si `v1` ya existe, la ejecución se rechaza y se conservan los artefactos anteriores.

Para consultar las versiones almacenadas:

```bash
find fase-2/models -maxdepth 2 -type f | sort
```

En PowerShell puede utilizarse:

```powershell
Get-ChildItem fase-2/models -Recurse -File
```

La política de versionamiento permite mantener varios modelos disponibles para su evaluación y utilización posterior.

## 9. Predicción mediante archivos CSV

La inferencia se ejecuta mediante:

`scripts/predict.py`

El programa recibe:

- `--model`: carpeta de la versión o archivo del modelo.
- `--input`: archivo CSV con las variables predictoras.
- `--output`: ruta del CSV donde se guardarán las predicciones.

### Esquema de entrada

El CSV debe contener las 11 columnas predictoras:

```csv
season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,hum,windspeed
1,1,6,8,0,1,1,1,0.55,0.65,0.20
1,1,6,9,0,1,1,1,0.60,0.60,0.18
2,1,7,17,0,2,1,2,0.70,0.50,0.15
```

La variable `cnt` no debe incluirse, porque constituye el valor que se desea predecir.

Tampoco es necesario incluir las características derivadas que se calculan durante el procesamiento de los datos.

Los valores nulos admitidos son tratados por el pipeline guardado durante el entrenamiento.

### Crear un archivo de ejemplo

Desde la raíz del repositorio:

**Linux:**

```bash
mkdir -p data

cat > data/example_prediction.csv <<'EOF'
season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,hum,windspeed
1,1,6,8,0,1,1,1,0.55,0.65,0.20
1,1,6,9,0,1,1,1,0.60,0.60,0.18
2,1,7,17,0,2,1,2,0.70,0.50,0.15
EOF
```

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force data | Out-Null

@'
season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,hum,windspeed
1,1,6,8,0,1,1,1,0.55,0.65,0.20
1,1,6,9,0,1,1,1,0.60,0.60,0.18
2,1,7,17,0,2,1,2,0.70,0.50,0.15
'@ | Set-Content -Encoding utf8 data/example_prediction.csv
```

### Ejecutar una predicción

**Windows (PowerShell):**

```powershell
python fase-2/scripts/predict.py `
  --model fase-2/models/v1 `
  --input data/example_prediction.csv `
  --output fase-2/outputs/predictions_v1.csv
```

**Linux:**

```bash
PYTHONPATH=fase-2/src python fase-2/scripts/predict.py \
  --model fase-2/models/v1 \
  --input data/example_prediction.csv \
  --output fase-2/outputs/predictions_v1.csv
```

El resultado esperado tiene la siguiente estructura:

```csv
fila,prediccion,version
1,600.914465070877,v1
2,297.4793891117053,v1
3,771.7686613075398,v1
```

Los valores son ilustrativos de una ejecución comprobada; las predicciones dependen del modelo utilizado.

El archivo de salida contiene:

- `fila`: posición original del registro.
- `prediccion`: cantidad estimada de alquileres.
- `version`: identificador del modelo utilizado.

El programa conserva el orden de las filas procesadas.

## 10. Pruebas automatizadas

Las pruebas se implementan mediante `pytest`.

Se ejecutan desde el directorio `fase-2/`.

### Linux

```bash
cd fase-2
PYTHONPATH=src python -m pytest -v
```

### Windows (PowerShell)

```powershell
cd fase-2
$env:PYTHONPATH = "src"
python -m pytest -v
```

### Cobertura funcional

Las pruebas comprueban:

| Módulo | Validación |
|---|---|
| `test_data_loader.py` | Validación del esquema de datos. |
| `test_data_loader.py` | Detección de columnas obligatorias ausentes. |
| `test_data_loader.py` | Validación de la variable objetivo. |
| `test_data_loader.py` | Detección de valores faltantes en el objetivo. |
| `test_data_loader.py` | Manejo de archivos inexistentes. |
| `test_features.py` | Generación de características cíclicas. |
| `test_features.py` | Generación de indicador de hora pico. |
| `test_features.py` | Separación entre predictores y variable objetivo. |
| `test_model.py` | Entrenamiento y almacenamiento de artefactos. |
| `test_model.py` | Imputación de valores nulos y predicción. |
| `test_model.py` | Equivalencia entre el pipeline en memoria y el serializado. |
| `test_model.py` | Rechazo de versiones duplicadas. |
| `test_model.py` | Coexistencia de versiones independientes. |

La ejecución satisfactoria de todas las pruebas permite comprobar que los componentes principales funcionan de manera integrada.

## 11. Ejecución con Docker

Docker permite ejecutar la fase 2 en un entorno aislado, sin depender directamente de las versiones de Python ni de las dependencias instaladas en el equipo anfitrión.

La imagen se construye utilizando el `Dockerfile` ubicado en `fase-2/`.

### Construcción de la imagen

Desde la raíz del repositorio:

**Linux:**

```bash
sudo docker build -f fase-2/Dockerfile -t bike-demand-fase2 .
```

**Windows (PowerShell):**

```powershell
docker build -f fase-2/Dockerfile -t bike-demand-fase2 .
```

Comprobar la imagen:

```bash
docker images bike-demand-fase2
```

En Linux, si el usuario no tiene permisos para acceder al socket de Docker, puede ser necesario anteponer `sudo`.

### Pruebas dentro del contenedor

Las mismas pruebas automatizadas pueden ejecutarse dentro de la imagen:

**Linux:**

```bash
sudo docker run --rm \
  --entrypoint python \
  bike-demand-fase2 \
  -m pytest -v tests/
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  --entrypoint python `
  bike-demand-fase2 `
  -m pytest -v tests/
```

Esta prueba comprueba que las dependencias instaladas en la imagen permiten ejecutar la suite automatizada sin utilizar el entorno virtual local.

### Entrenamiento dentro de Docker

Para entrenar modelos y conservar sus archivos en el equipo anfitrión, se monta la raíz del repositorio como volumen.

**Linux:**

```bash
sudo docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/app" \
  -w /app \
  bike-demand-fase2 \
  python fase-2/scripts/train.py \
    --config fase-2/configs/config.yaml \
    --version docker-v1
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -w /app `
  bike-demand-fase2 `
  python fase-2/scripts/train.py `
    --config fase-2/configs/config.yaml `
    --version docker-v1
```

En Linux se utiliza `--user` para que los archivos creados dentro del contenedor pertenezcan al usuario que ejecuta el comando y no queden como archivos propiedad de root.

El entrenamiento genera los artefactos en:

```text
fase-2/models/docker-v1/
├── modelo.joblib
├── metrics.json
└── metadata.json
```

### Predicción dentro de Docker

Una vez entrenado el modelo, se puede utilizar dentro de otro contenedor:

**Linux:**

```bash
sudo docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/app" \
  -w /app \
  bike-demand-fase2 \
  python fase-2/scripts/predict.py \
    --model fase-2/models/docker-v1 \
    --input data/example_prediction.csv \
    --output fase-2/outputs/docker_predictions.csv
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -w /app `
  bike-demand-fase2 `
  python fase-2/scripts/predict.py `
    --model fase-2/models/docker-v1 `
    --input data/example_prediction.csv `
    --output fase-2/outputs/docker_predictions.csv
```

El resultado se guarda en:

`fase-2/outputs/docker_predictions.csv`

### Persistencia de archivos

La opción `-v` monta un directorio del equipo anfitrión dentro del contenedor.

En este proyecto se utiliza:

```text
Directorio del equipo             Directorio del contenedor

Repositorio completo  ──────────> /app
```

De esta manera, los modelos entrenados, los metadatos y los resultados CSV se escriben directamente en el repositorio.

La opción `--rm` elimina el contenedor al terminar la ejecución, pero no elimina los archivos que se guardaron en el volumen montado.

Por tanto, los artefactos pueden consultarse desde Windows o Linux después de cerrar el contenedor.

## 12. Resultados de las pruebas

Durante la validación de la fase 2 se realizaron pruebas de entrenamiento, predicción, versionamiento, ejecución local y ejecución mediante Docker.

### Entrenamiento

Se obtuvieron los siguientes resultados en las ejecuciones realizadas:

| Ejecución | Configuración | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| `v4` | `config.yaml` | 26,9511 | 45,1964 | 0,9345 |
| `v5` | `versions/v2.yaml` | 26,8826 | 45,0202 | 0,9350 |

Estos resultados corresponden a las ejecuciones de prueba identificadas como `v4` y `v5`. No deben confundirse con los identificadores oficiales `v1` y `v2` hasta que se ejecuten los entrenamientos con esos nombres.

### Pruebas automatizadas en Linux

Entorno utilizado:

- Arch Linux.
- Python 3.14.8.
- pytest 9.2.0.

Resultado:

```text
13 passed in 1.24s
```

### Pruebas automatizadas en Docker

Entorno utilizado:

- Imagen basada en `python:3.12-slim`.
- Python 3.12.16.
- pytest 9.2.0.

Resultado:

```text
13 passed in 1.44s
```

### Prueba de entrenamiento en Docker

Se ejecutó el entrenamiento con el identificador `docker-v1`.

Resultado:

```text
Entrenamiento completado: docker-v1

MAE: 26.9511
RMSE: 45.1964
R2: 0.9345
```

Se comprobó la generación de:

- `modelo.joblib`
- `metrics.json`
- `metadata.json`

### Prueba de predicción

Se ejecutó la inferencia con tres registros de entrada, tanto localmente como dentro de Docker.

Las predicciones obtenidas fueron equivalentes:

| Fila | Predicción local | Predicción Docker |
|---|---:|---:|
| 1 | 600,9145 | 600,9145 |
| 2 | 297,4794 | 297,4794 |
| 3 | 771,7687 | 771,7687 |

Ambas ejecuciones identificaron correctamente la versión del modelo utilizada.

Esta comprobación demuestra que el pipeline serializado puede cargarse y utilizarse en ambos entornos para procesar las mismas entradas.

## 13. Limitaciones

- Las métricas representan el desempeño obtenido sobre el conjunto de prueba utilizado y no garantizan el mismo rendimiento ante cualquier conjunto de datos futuro.
- Los valores faltantes fueron simulados para evaluar el mecanismo de imputación y no representan necesariamente fallos reales de sensores.
- El conjunto de datos corresponde a un sistema específico y a un período histórico limitado.
- La calidad de las predicciones depende de que los datos de entrada tengan un esquema compatible con el utilizado durante el entrenamiento.
- Las pruebas realizadas comprueban las funcionalidades implementadas, pero no sustituyen una evaluación de generalización con datos externos.
- La fase 2 no proporciona todavía un servicio REST para consumir las predicciones desde otras aplicaciones.

## 14. Trabajo posterior

La fase 2 establece la base para las siguientes etapas del proyecto.

La fase 3 incorporará una API REST para exponer las funcionalidades de entrenamiento y predicción mediante endpoints HTTP.

La fase 4 incorporará mecanismos de monitoreo y registro de solicitudes, cálculo de métricas operacionales, identificación de versiones activas y procedimientos de recuperación ante cambios de modelo.

Estas etapas permitirán evolucionar el flujo de Machine Learning hacia un servicio reproducible y monitoreable.

---

**Proyecto Integrador · Modelos y Simulación de Sistemas I**

**Universidad de Antioquia · 2026-II**
