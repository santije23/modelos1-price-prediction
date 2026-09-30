# Predicción de la demanda horaria de bicicletas compartidas utilizando variables temporales y meteorológicas

Proyecto Integrador - Modelos y Simulación de Sistemas I (Universidad de Antioquia) **Fase 1: Modelo predictivo**

## Integrantes del equipo

- **Santiago Jiménez Escobar** [santiago.jimeneze@udea.edu.co](mailto:santiago.jimeneze@udea.edu.co)
    
- **Santiago Durán Carmona** [santiago.duran1@udea.edu.co](mailto:santiago.duran1@udea.edu.co)
    

## Descripción del problema

El proyecto busca desarrollar un modelo de **Machine Learning** capaz de predecir la demanda horaria de un sistema de bicicletas compartidas, entendida como el número total de bicicletas alquiladas durante una hora determinada.

La predicción se realiza a partir de variables **temporales y de calendario**, como el año, el mes, el día de la semana, la hora, la condición de día festivo o laboral y la estación del año. También se utilizan **variables meteorológicas**: temperatura, humedad, velocidad del viento y condición general del clima. La sensación térmica (`atemp`) se excluye del conjunto de predictores por su alta colinealidad con la temperatura (`temp`), con una correlación cercana a 0,99.

El conjunto de datos corresponde al histórico real del sistema **Capital Bikeshare**, ubicado en Washington D. C., y contiene **17.379 registros horarios** correspondientes al período comprendido entre el 1 de enero de 2011 y el 31 de diciembre de 2012.

La variable objetivo del modelo es `cnt`, que representa el **número total de alquileres registrados durante cada hora**. Esta variable corresponde a la suma de los alquileres realizados por usuarios casuales (`casual`) y usuarios registrados (`registered`).

Debido a que `cnt` se obtiene directamente de la suma de `casual` y `registered`, estas dos variables se excluyen del conjunto de variables predictoras. Utilizarlas durante el entrenamiento implicaría proporcionar al modelo información directamente relacionada con el resultado que se pretende predecir, lo que generaría **fuga de información (data leakage)** y produciría métricas de desempeño que no reflejarían la capacidad real de generalización del modelo.

La predicción de la demanda de bicicletas puede contribuir a mejorar la **gestión y planificación de los sistemas de movilidad urbana**. Conocer con anticipación la cantidad de bicicletas que podrían ser utilizadas permite a los operadores tomar mejores decisiones sobre la redistribución de bicicletas, la planificación del mantenimiento y la disponibilidad del servicio, especialmente durante las horas de mayor demanda.

Por estas razones, el problema se relaciona directamente con el área de **movilidad y transporte** y permite aplicar técnicas de Machine Learning a un caso real de predicción de demanda.

## Fuente del conjunto de datos

El conjunto de datos utilizado se encuentra disponible en Kaggle:

**Bike Sharing Dataset:** https://www.kaggle.com/datasets/amil09/bike-sharing-dataset/data?select=Bike+Share+hourly.csv

## Objetivo del modelo

Desarrollar un modelo de **Machine Learning** capaz de predecir la cantidad de bicicletas que serán alquiladas durante una determinada franja horaria, utilizando como variables de entrada las condiciones temporales, de calendario y meteorológicas correspondientes a dicha hora.

El modelo debe aprender la relación existente entre estas variables y la demanda histórica de bicicletas, con el propósito de generar predicciones que permitan anticipar los períodos de mayor y menor demanda.

## Preparación de los datos

**Selección de variables.** De las 17 columnas originales se utilizan 11 variables predictoras: `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `hum` y `windspeed`. Las columnas restantes se excluyen por los siguientes motivos:

|Columna|Motivo de exclusión|
|:--|:--|
|`instant`|Índice del registro, sin poder predictivo.|
|`dteday`|Fecha cruda; sus componentes ya están disponibles en `yr`, `mnth`, `weekday` y `hr`.|
|`casual` y `registered`|Componentes de `cnt`; su uso produciría fuga de información.|
|`atemp`|Colineal con `temp` (r ≈ 0,99); se conserva únicamente `temp`.|
|`cnt`|Variable objetivo.|

**Valores faltantes.** El conjunto de datos original no contiene valores nulos. Para poder aplicar y justificar una estrategia de imputación, se inyectaron valores faltantes de forma controlada, con mecanismo **MCAR** (_Missing Completely At Random_) y semilla fija: 1,2 % en `hum` y 1,7 % en `windspeed`, variables que en la práctica dependen de sensores ambientales. Además, se realizó un análisis de sensibilidad para comprobar que esta inyección no degrada de forma relevante el desempeño del modelo.

**Partición de los datos.** Se utilizó una partición aleatoria (_hold-out_) de 80 % para entrenamiento y 20 % para prueba, con `random_state=42`. El problema no se trata como una serie de tiempo, sino como una predicción puntual por hora a partir de variables explícitas de calendario y clima.

## Algoritmo utilizado

- **Modelo principal:** `Random Forest Regressor` (200 árboles, `random_state=42`)
    - **Razón de selección:** es un algoritmo de ensamble (_bagging_) capaz de capturar relaciones no lineales complejas entre las características (clima, hora, día de la semana, etc.) y la demanda de alquileres, con un alto rendimiento y una menor tendencia al sobreajuste (_overfitting_) que un árbol de decisión individual.
- **Modelo baseline (base):** `Ridge Regression` (`alpha=1.0`)
    - **Razón de selección:** se utilizó como punto de comparación inicial para medir la efectividad de un modelo lineal regularizado frente a un modelo basado en árboles.
- **Preprocesamiento:** ambos modelos se integraron en un `Pipeline` de Scikit-Learn con un `ColumnTransformer` que incluye:
    - **Variables numéricas** (`temp`, `hum`, `windspeed`): imputación de valores faltantes con la mediana y escalamiento con `StandardScaler`.
    - **Variables categóricas y discretas** (`season`, `weathersit`, `mnth`, `hr`, `weekday`, `yr`, `holiday`, `workingday`): imputación con el valor más frecuente y codificación `One-Hot Encoding`, para evitar que el modelo asuma un orden que no existe entre las categorías.

Al estar todas las transformaciones dentro del `Pipeline`, sus parámetros (mediana, media, categorías) se ajustan únicamente con los datos de entrenamiento, lo que evita la fuga de información hacia el conjunto de prueba.

## Métricas empleadas

Se seleccionó un conjunto de tres métricas complementarias para evaluar los modelos de regresión:

1. **MAE (Mean Absolute Error):** mide el error absoluto promedio en las mismas unidades de la variable objetivo. Permite interpretar de manera directa cuántos alquileres por hora se desvía el modelo en promedio.
2. **RMSE (Root Mean Squared Error):** penaliza con mayor severidad las grandes desviaciones o errores atípicos, lo que permite evaluar la precisión del modelo ante picos inusuales de demanda.
3. **R² (coeficiente de determinación):** cuantifica el porcentaje de la variabilidad de la demanda horaria que el modelo logra explicar.

## Principales resultados obtenidos

Al evaluar los modelos en el conjunto de prueba (20 % de los datos), el modelo **Random Forest Regressor** superó de forma significativa al modelo base (**Ridge Regression**):

|Modelo|MAE|RMSE|R²|
|:--|:-:|:-:|:-:|
|**Ridge Regression (baseline)**|74,06|100,49|0,6811 (68,11 %)|
|**Random Forest Regressor**|**29,34**|**48,02**|**0,9272 (92,72 %)**|

**Conclusiones clave de los resultados:**

- **Reducción del error:** Random Forest redujo el error promedio (`MAE`) a menos de la mitad, pasando de unos 74 a unos 29 alquileres de diferencia respecto al valor real.
- **Menor magnitud de los errores grandes:** el `RMSE` disminuyó de 100,49 a 48,02, lo que evidencia una reducción importante de los errores de mayor tamaño.
- **Alto poder explicativo:** el modelo alcanzó un `R²` de **0,9272**, explicando el **92,72 %** de la variabilidad de la demanda de bicicletas, frente a solo el 68,11 % del baseline.

## Validaciones adicionales

- **Sensibilidad a los valores faltantes:** con el conjunto de datos original (sin nulos inyectados), el RMSE fue de 47,90 y el R² de 0,9276, frente a 48,02 y 0,9272 con los nulos inyectados. La diferencia relativa en RMSE es de 0,26 %, por lo que la imputación dentro del `Pipeline` no degrada el desempeño de forma relevante.
- **Error por segmentos:** el mayor error absoluto se presenta en las horas pico (MAE de aproximadamente 60,5 a las 17:00, 57,5 a las 18:00 y 50,8 a las 8:00), con clima adverso (MAE de aproximadamente 49,6 con lluvia o nieve ligera) y en fines de semana y festivos (MAE de aproximadamente 34,9, frente a 26,9 en días laborales).
- **Análisis de residuos:** el sesgo medio global es de -2,02 alquileres, por lo que no se observa un sesgo global importante. Sin embargo, cuando la demanda real supera los 500 alquileres, el modelo tiende a subestimar en promedio unos 36 alquileres.
- **Prueba de sensibilidad a la temperatura:** al variar únicamente `temp` sobre un caso típico, la predicción aumenta con la temperatura hasta cierto punto y disminuye con el calor extremo, comportamiento coherente con el análisis exploratorio.
- **Persistencia del modelo:** el modelo guardado en formato `.joblib` produce exactamente las mismas predicciones que el modelo en memoria, imputa por sí solo los valores faltantes de un registro nuevo y no genera predicciones negativas.

## Limitaciones y mejoras futuras

**Limitaciones:**

- Cada día aparece hasta 24 veces (una fila por hora), por lo que la partición aleatoria puede dejar horas de un mismo día en ambos conjuntos. Esto puede hacer que las métricas de prueba sean algo optimistas frente a días completamente nuevos.
- Existe un sobreajuste moderado: el R² en entrenamiento es de 0,9899, frente a 0,9272 en prueba.
- Los valores faltantes fueron inyectados de forma simulada y no corresponden a fallos reales de sensores.
- Los datos provienen de un único sistema y de un período específico (2011 y 2012), por lo que los patrones podrían no generalizarse a otros contextos ni incorporar factores externos como eventos especiales.

**Mejoras futuras:**

- Optimización de hiperparámetros (`GridSearchCV` o `RandomizedSearchCV`) y comparación con otros algoritmos, como `GradientBoostingRegressor`.
- Validación agrupada por día (`GroupShuffleSplit` o `GroupKFold`) para obtener una estimación menos optimista del desempeño.
- Reducción del error en horas pico, clima adverso y fines de semana, así como de la subestimación de los valores altos de demanda.
- Reducción del sobreajuste limitando la profundidad de los árboles o ajustando `min_samples_leaf`.
- Codificación cíclica (seno y coseno) de `hr`, `mnth` y `weekday`, y creación de variables derivadas, como la hora pico o interacciones entre hora y tipo de día.

## Estructura del repositorio

```text
.
├── data/
│   ├── raw/
│   │   └── bike_share_hourly.csv                # Dataset original (incluido en el repositorio)
│   └── processed/
│       └── bike_share_with_nans.csv          # Generado por el notebook
├── fase-1/
│   ├── notebooks/
│   │   └── 01_modelo_predictivo.ipynb   # Notebook principal
│   ├── models/
│   │   └── model_v1.joblib                            # Generado por el notebook
│   └── reports/
│       ├── metricas_v1.json                            # Generado por el notebook
│       └── figures/                                             # Gráficas generadas por el notebook
├── fase-2/
│   └── .gitkeep                                                 # Pendiente
├── fase-3/
│   └── .gitkeep                                                 # Pendiente
├── fase-4/
│   └── .gitkeep                                                 # Pendiente
├── requirements.txt
└── README.md
```

El proyecto esta en fase de desarrollo y se organiza por fases. Las carpetas `fase-2`, `fase-3` y `fase-4` contienen únicamente un archivo `.gitkeep`, que permite conservar la estructura de carpetas en Git mientras se completan las etapas correspondientes.

|Fase|Contenido|Estado|
|:--|:--|:--|
|Fase 1|Modelo predictivo|Disponible|
|Fase 2|Scripts y Docker|Pendiente|
|Fase 3|API REST|Pendiente|
|Fase 4|Monitoreo|Pendiente|

El notebook localiza automáticamente la carpeta raíz del proyecto: busca, desde el directorio de trabajo hacia arriba, la primera carpeta que contenga `data/raw`. Por ello, debe ejecutarse desde dentro del repositorio.

## Instrucciones para ejecutar el notebook

### Requisitos previos

- **Python 3.11 o superior** (las versiones de `pandas` y `numpy` definidas en `requirements.txt` lo requieren).
- **pip**, incluido con las instalaciones estándar de Python.
- **Git** (opcional, solo si se clona el repositorio).

Comprobar la versión de Python instalada:

```bash
# Windows
py --version

# Linux
python3 --version
```

### 1. Clonar el proyecto

Clone el repositorio (o descargue el código como archivo ZIP y descomprímalo) y ubíquese en la carpeta raíz:

```bash
git clone https://github.com/santije23/modelos1-price-prediction
cd modelos1-price-prediction
```

### 2. Crear y activar un entorno virtual

**Windows**

- **PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución del script de activación, habilite la ejecución de scripts locales para el usuario actual y vuelva a activar el entorno:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

- **CMD**

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

**Linux (Bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias

Con el entorno virtual activo, ejecute (el comando es el mismo para Windows y Linux):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Abrir el notebook

Con el entorno virtual activo y ubicado en la raíz del proyecto, inicie Jupyter con cualquiera de las dos interfaces disponibles. El comando es el mismo en Windows y en Linux, y ambas vienen incluidas con las dependencias del proyecto:

```bash
jupyter notebook    # Jupyter Notebook
jupyter lab                 # JupyterLab
```

|                      | Jupyter Notebook                                                      | JupyterLab                                                                                       |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Enfoque**          | Interfaz sencilla, centrada en un notebook por pestaña del navegador. | Entorno de trabajo más completo, similar a un editor de código.                                  |
| **Funciones**        | Explorador de archivos y edición de notebooks.                        | Explorador lateral, varias pestañas y paneles en paralelo, terminal integrada y editor de texto. |
| **Recomendado para** | Abrir y ejecutar este notebook de forma directa.                      | Trabajar con varios archivos a la vez (notebook, terminal, `README.md`, etc.                     |

Para este proyecto, ambas opciones son equivalentes. Se abrirá el navegador con la interfaz elegida: navegue hasta `fase-1/notebooks/` y abra `01_modelo_predictivo.ipynb`.

Como alternativa, el notebook también puede abrirse en Visual Studio Code con la extensión de Jupyter, seleccionando como intérprete el de la carpeta `.venv`.

### 5. Ejecutar el notebook

En el menú de Jupyter, seleccione **Kernel > Restart Kernel and Run All Cells** (en VS Code, **Run All**) para ejecutar todas las celdas en orden. El entrenamiento del Random Forest puede tardar desde unos segundos hasta algunos minutos, según el equipo.

Las celdas deben ejecutarse en orden, de arriba hacia abajo, porque cada sección depende de variables definidas en las anteriores.

### 6. Verificar los resultados

Al finalizar, el notebook genera los siguientes archivos:

|Ruta|Contenido|
|:--|:--|
|`data/processed/bike_share_with_nans.csv`|Conjunto de datos con los valores faltantes inyectados.|
|`fase-1/models/model_v1.joblib`|Pipeline completo entrenado (preprocesamiento y Random Forest).|
|`fase-1/reports/metricas_v1.json`|Métricas del baseline y del modelo, hiperparámetros y variables utilizadas.|
|`fase-1/reports/figures/`|Gráficas del análisis exploratorio y de la evaluación del modelo.|

Para cargar el modelo guardado y generar predicciones desde Python:

```python
import joblib

modelo = joblib.load("fase-1/models/model_v1.joblib")
predicciones = modelo.predict(X_nuevos)  # X_nuevos: DataFrame con las 11 variables predictoras
```

### Reproducibilidad

Todos los procesos aleatorios (partición de los datos, inyección de valores faltantes y entrenamiento del Random Forest) utilizan la semilla `random_state=42`. Para obtener resultados idénticos a los reportados, se recomienda usar las versiones exactas de las librerías definidas en `requirements.txt`.