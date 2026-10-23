# Predicción de la demanda horaria de bicicletas compartidas — Fase 2

**Proyecto Integrador · Modelos y Simulación de Sistemas I · 2026 II**

**Integrantes:** Santiago Jiménez Escobar · Santiago Durán Carmona

> 🚧 **Borrador en construcción.** Las secciones marcadas con 🚧 se completan en los siguientes pasos de la Fase 2 (pruebas, Docker, versiones definitivas). Este documento ya cubre la **organización del código** y el uso de `train.py` y `predict.py`.

---

## 1. Problema y relación con la Fase 1

Se predice **`cnt`**, el número de bicicletas alquiladas en una hora, a partir de variables de calendario y meteorológicas del sistema Capital Bikeshare (Washington D. C., 17.379 registros horarios de 2011 y 2012). Es un problema de **regresión**.

La Fase 1 (carpeta `fase-1/`, conservada sin cambios) desarrolló el análisis y el modelo en un notebook. La Fase 2 convierte ese trabajo en una aplicación ejecutable por línea de comandos:

- `train.py` entrena el modelo y el modelo base, y guarda una **versión** identificable.
- `predict.py` carga una versión guardada y genera predicciones sobre datos nuevos.

Se conservan el problema, el conjunto de datos y el enfoque de modelado: **Random Forest** como modelo y **Ridge** como modelo base (baseline).

## 2. Cambios respecto a la Fase 1

| Aspecto | Fase 1 | Fase 2 |
|---|---|---|
| Partición train/test | Aleatoria (`train_test_split`), con horas del mismo día en ambos conjuntos | **Agrupada por día** (`GroupShuffleSplit` sobre `dteday`): cada día completo va a un solo conjunto |
| Organización | Un notebook | Módulos reutilizables en `src/` + dos scripts con `argparse` |
| Validación de datos | No había | Se valida archivo, columnas, tipos y variable objetivo, con mensajes de error claros |
| Almacenamiento | `model_v1.joblib` + `metricas_v1.json` sueltos | Una carpeta por versión con modelo, métricas y metadatos; nunca se sobrescribe |
| Rutas | Detectadas desde el notebook (`BASE_DIR`) | Siempre por argumentos de línea de comandos |

**Por qué cambia la partición.** En el dataset cada día aparece hasta 24 veces (una fila por hora) y esas filas se parecen mucho. Con una partición aleatoria, el modelo podía ver en entrenamiento horas vecinas de un día que luego se evaluaba en prueba, lo que hacía las métricas algo optimistas. Con la partición agrupada, el conjunto de prueba contiene **días completamente nuevos**. Las métricas son por ello algo menores que las de la Fase 1, pero más realistas. El baseline se evalúa con la misma partición.

La estrategia de nulos, la selección de variables (exclusión de `casual` y `registered` por fuga de información, de `atemp` por colinealidad, y de `instant` y `dteday`) y los modelos se mantienen como en la Fase 1.

## 3. Organización del código

```
fase-2/
├── train.py              # punto de entrada: entrena y guarda una versión
├── predict.py            # punto de entrada: predice con una versión guardada
├── prepare_data.py       # 🚧 punto de entrada: genera el CSV con nulos inyectados (MCAR)
├── requirements.txt      # 🚧
├── Dockerfile            # 🚧
├── .dockerignore         # 🚧
├── src/
│   ├── config.py         # constantes: variable objetivo y columnas
│   ├── data.py           # lectura y validación de datos
│   ├── preprocessing.py  # construcción del preprocesamiento
│   ├── training.py       # partición, construcción y ajuste de modelos
│   ├── evaluation.py     # métricas del modelo y del baseline
│   ├── storage.py        # guardado y carga de versiones
│   └── prediction.py     # validación y predicción sobre datos nuevos
├── tests/                # 🚧 pruebas con pytest
├── data/
│   ├── raw/              # datos originales (no se modifican)
│   ├── processed/        # datos derivados (CSV con nulos inyectados)
│   └── ejemplo_prediccion.csv
├── models/               # versiones del modelo: v1/, v2/, ...
└── outputs/              # predicciones generadas
```

**Principios de diseño**

- `train.py` y `predict.py` son **orquestadores delgados**: solo reciben argumentos (`argparse`) y llaman, en orden, a funciones de `src/`. Toda la lógica vive en los módulos.
- **Nada se ejecuta al importar** un módulo: la ejecución está dentro de `main()` y de `if __name__ == "__main__":`.
- **No hay rutas absolutas, entradas interactivas ni dependencia de un notebook.** Las rutas llegan por argumentos.
- **Las transformaciones no se duplican.** El preprocesamiento se define una sola vez (`build_preprocessor()`), forma parte del `Pipeline` y se guarda junto con el modelo. `predict.py` solo carga y predice: no reajusta nada.

| Módulo | Responsabilidad | Funciones principales |
|---|---|---|
| `config.py` | Esquema de columnas único para todo el proyecto | `TARGET`, `GROUP_COL`, `NUM_COLS`, `CAT_COLS`, `FEATURE_COLS` |
| `data.py` | Leer y validar datos; no transforma nada | `load_data`, `validate_columns`, `validate_numeric_types`, `validate_target`, `select_features` |
| `preprocessing.py` | Construir el `ColumnTransformer` (sin ajustarlo) | `build_preprocessor` |
| `training.py` | Partición agrupada y modelos | `split_data`, `build_model`, `fit_model` |
| `evaluation.py` | MAE, RMSE y R²; comparación con el baseline | `compute_metrics`, `evaluate_models` |
| `storage.py` | Versiones: guardar sin sobrescribir, cargar, metadatos | `save_version`, `load_version`, `build_metadata` |
| `prediction.py` | Validar entrada, predecir y escribir resultados | `predict_dataframe`, `build_output`, `write_predictions` |

## 4. Preprocesamiento y reglas de datos

El preprocesamiento es un `ColumnTransformer` dentro de un `Pipeline`, y se ajusta **únicamente con el conjunto de entrenamiento**:

| Tipo | Variables | Tratamiento |
|---|---|---|
| Numéricas | `temp`, `hum`, `windspeed` | Imputación por **mediana** + estandarización |
| Categóricas | `season`, `weathersit`, `mnth`, `hr`, `weekday`, `yr`, `holiday`, `workingday` | Imputación por **moda** + one-hot |

**Reglas al predecir** (se aplican con lo aprendido en el entrenamiento):

- **Valores nulos en una celda:** se imputan (mediana en numéricas, moda en categóricas). Un nulo **no** es un error.
- **Columna requerida ausente:** error con mensaje claro y código de salida distinto de cero.
- **Tipo de dato incompatible** (por ejemplo, texto en una columna numérica): error claro y código de salida distinto de cero.
- **Columnas adicionales** (incluida `cnt` si viene en el CSV): se **ignoran**.
- **Orden de las columnas del CSV:** no importa; se reordenan según el esquema guardado con la versión.
- **Categorías no vistas en el entrenamiento** (por ejemplo, `weathersit=9`): no generan error; el one-hot las deja en ceros.
- **Filas:** se devuelve exactamente una predicción por fila, en el mismo orden. No se eliminan observaciones.

**Limitación conocida:** no se validan los rangos de los valores (por ejemplo, `hr=30`); un valor fuera de rango se trata como categoría no vista.

## 5. Datos

- **Fuente:** [Bike Sharing Dataset (Kaggle)](https://www.kaggle.com/datasets/amil09/bike-sharing-dataset/data?select=Bike+Share+hourly.csv).
- **Ubicación esperada:** descargar el archivo, renombrarlo a `bike_share_hourly.csv` y colocarlo en `fase-2/data/raw/`.
- **Datos con nulos inyectados:** la Fase 1 inyectó valores faltantes MCAR (1,2 % en `hum` y 1,7 % en `windspeed`, semilla 42). 🚧 El script `prepare_data.py` regenera `data/processed/bike_share_with_nans.csv` a partir del archivo original.

### Esquema del CSV de predicción

La variable objetivo **no** es necesaria. Se incluye un ejemplo en `data/ejemplo_prediccion.csv` (10 filas, dos de ellas con nulos en `hum` y `windspeed`).

| Columna | Descripción | Valores |
|---|---|---|
| `temp` | Temperatura normalizada | 0 a 1 |
| `hum` | Humedad normalizada (admite nulos) | 0 a 1 |
| `windspeed` | Velocidad del viento normalizada (admite nulos) | 0 a 1 |
| `season` | Estación | 1 a 4 |
| `weathersit` | Situación climática | 1 a 4 |
| `mnth` | Mes | 1 a 12 |
| `hr` | Hora del día | 0 a 23 |
| `weekday` | Día de la semana | 0 a 6 |
| `yr` | Año (0 = 2011, 1 = 2012) | 0 o 1 |
| `holiday` | Día festivo | 0 o 1 |
| `workingday` | Día laboral | 0 o 1 |

### Salida de `predict.py`

CSV con dos columnas (más una opcional con `--id-col`):

- `fila`: posición de la fila en el CSV de entrada (desde 1).
- `prediccion`: **número estimado de bicicletas alquiladas en esa hora** (regresión; no hay etiquetas de clase).

## 6. Ejecución local

Terminal utilizada: **PowerShell** (Windows). Todos los comandos se ejecutan desde la carpeta `fase-2`.

🚧 *Versión de Python, creación del entorno e instalación de dependencias: se completa al fijar `requirements.txt`.*

### Entrenamiento

```powershell
python train.py --data data/processed/bike_share_with_nans.csv --target cnt --output-dir models --version v1 --seed 42
```

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--data` | CSV con datos etiquetados | obligatorio |
| `--target` | Variable objetivo | `cnt` |
| `--output-dir` | Directorio donde se guardan las versiones | `models` |
| `--version` | Identificador de la versión (no se sobrescribe si ya existe) | obligatorio |
| `--seed` | Semilla aleatoria | `42` |
| `--test-size` | Proporción de días para prueba | `0.2` |
| `--n-estimators` | Número de árboles del Random Forest | `200` |
| `--min-samples-leaf` | Mínimo de muestras por hoja del Random Forest | `1` |

### Predicción

```powershell
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

`--model` acepta la ruta a `modelo.joblib` o la carpeta de la versión (`models/v1`). Con `--id-col` se copia una columna del CSV de entrada como identificador en la salida.

### Ayuda

```powershell
python train.py --help
python predict.py --help
```

## 7. Versiones del modelo

Cada ejecución de `train.py` crea una carpeta nueva dentro de `models/`. Una versión existente **nunca se sobrescribe**: si el identificador ya existe, el script termina con un error antes de entrenar.

```
models/
└── v1/
    ├── modelo.joblib     # pipeline completo: preprocesamiento + modelo
    ├── metrics.json      # métricas del modelo y del baseline (train y test)
    └── metadata.json     # versión, fecha, datos, partición, variables, entorno y commit
```

🚧 *Se documentarán aquí las versiones `v1` y `v2` entregadas y sus diferencias.*

## 8. Resultados

Métricas en el conjunto de prueba (días no vistos), partición agrupada por día, semilla 42. Valores **provisionales** de una ejecución de verificación; 🚧 se actualizarán con la versión entregada.

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (Ridge) | 72,68 | 97,84 | 0,693 |
| Random Forest | 31,93 | 51,64 | 0,914 |

El Random Forest reduce el MAE en ≈ 56 % y el RMSE en ≈ 47 % frente al baseline.

**Limitaciones conocidas**

- **Sobreajuste moderado:** R² de 0,990 en entrenamiento frente a 0,914 en prueba.
- El modelo se equivoca más en horas pico, con mal clima y en fines de semana (según el análisis de la Fase 1).
- No se validan rangos de los valores de entrada.
- El Random Forest es no determinista por naturaleza; se fija `random_state` con la semilla indicada para obtener resultados consistentes en el mismo entorno.

## 9. 🚧 Docker

*Pendiente: comandos de construcción y ejecución con montajes de carpetas.*

## 10. 🚧 Pruebas

*Pendiente: comando `python -m pytest` y descripción de las pruebas.*

## 11. 🚧 Identificación de la entrega y fuentes

- **Commit o etiqueta de la Entrega 2:** *pendiente (se sugiere la etiqueta `entrega-2`).*
- **Fuentes consultadas y código externo adaptado:** *pendiente.*
