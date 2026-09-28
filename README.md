# Predicción de la demanda horaria de bicicletas compartidas utilizando variables temporales y meteorológicas

## Integrantes del equipo

* **Santiago Jiménez Escobar**
  [santiago.jimeneze@udea.edu.co](mailto:santiago.jimeneze@udea.edu.co)

* **Santiago Durán Carmona**
  [santiago.duran1@udea.edu.co](mailto:santiago.duran1@udea.edu.co)

## Descripción del problema

El proyecto busca desarrollar un modelo de **Machine Learning** capaz de predecir la demanda horaria de un sistema de bicicletas compartidas, entendida como el número total de bicicletas alquiladas durante una hora determinada.

La predicción se realizará a partir de diferentes variables **temporales y de calendario**, como el año, mes, día de la semana, hora, condición de día festivo o laboral y estación del año. También se utilizarán **variables meteorológicas**, entre ellas la temperatura, sensación térmica, humedad, velocidad del viento y condición general del clima.

El conjunto de datos corresponde al histórico real del sistema **Capital Bikeshare**, ubicado en Washington D. C., y contiene **17.379 registros horarios** correspondientes al período comprendido entre el 1 de enero de 2011 y el 31 de diciembre de 2012.

La variable objetivo del modelo será `cnt`, que representa el **número total de alquileres registrados durante cada hora**. Esta variable corresponde a la suma de los alquileres realizados por usuarios casuales (`casual`) y usuarios registrados (`registered`).

Debido a que `cnt` se obtiene directamente de la suma de `casual` y `registered`, estas dos variables serán excluidas del conjunto de variables predictoras. Utilizarlas durante el entrenamiento implicaría proporcionar al modelo información directamente relacionada con el resultado que se pretende predecir, generando **fuga de información (data leakage)** y produciendo métricas de desempeño que no representarían correctamente la capacidad real de generalización del modelo.

La predicción de la demanda de bicicletas puede contribuir a mejorar la **gestión y planificación de los sistemas de movilidad urbana**. Conocer anticipadamente la cantidad de bicicletas que podrían ser utilizadas permite a los operadores tomar mejores decisiones relacionadas con la redistribución de bicicletas, la planificación del mantenimiento y la disponibilidad del servicio, especialmente durante las horas de mayor demanda.

Por estas razones, el problema se encuentra directamente relacionado con el área de **movilidad y transporte**, permitiendo aplicar técnicas de Machine Learning a un problema real de predicción de demanda.

## Fuente del conjunto de datos

El conjunto de datos utilizado se encuentra disponible en Kaggle:

**Bike Sharing Dataset:**
https://www.kaggle.com/datasets/amil09/bike-sharing-dataset/data?select=Bike+Share+hourly.csv

## Objetivo del modelo

Desarrollar un modelo de **Machine Learning** capaz de predecir la cantidad de bicicletas que serán alquiladas durante una determinada franja horaria, utilizando como variables de entrada las condiciones temporales, de calendario y meteorológicas correspondientes a dicha hora.

El modelo deberá aprender la relación existente entre estas variables y la demanda histórica de bicicletas, con el propósito de generar predicciones que puedan ser utilizadas para anticipar los períodos de mayor y menor demanda.


## Algoritmo utilizado

* **Modelo Principal:** `Random Forest Regressor`
  * **Razón de selección:** Es un algoritmo de ensamble (Bagging) capaz de capturar relaciones no lineales complejas entre las características (clima, hora, día de la semana, etc.) y la demanda de alquileres, ofreciendo un alto rendimiento y reduciendo el sobreajuste (*overfitting*).
* **Modelo Baseline (Base):** `Ridge Regression`
  * **Razón de selección:** Se utilizó como punto de comparación inicial para medir la efectividad de un modelo lineal regularizado frente a un modelo basado en árboles.
* **Preprocesamiento:** Ambos modelos se integraron en un `Pipeline` de Scikit-Learn que incluyó imputación de valores faltantes (mediana), escalamiento de variables numéricas y codificación `One-Hot Encoding` para variables categóricas.


## Métricas empleadas

Se seleccionó un conjunto de tres métricas complementarias para evaluar los modelos de regresión:

1. **MAE (Mean Absolute Error):** Mide el error absoluto promedio en las mismas unidades de la variable objetivo. Permite interpretar de manera directa cuántos alquileres por hora se desvía el modelo en promedio.
2. **RMSE (Root Mean Squared Error):** Penaliza con mayor severidad las grandes desviaciones o errores atípicos, permitiendo evaluar la precisión del modelo ante picos inusuales en la demanda.
3. **R² (Coeficiente de determinación):** Cuantifica el porcentaje de variabilidad de la demanda horaria que el modelo logra explicar.


## Principales resultados obtenidos

Al evaluar los modelos en el conjunto de prueba (20% de los datos), el modelo **Random Forest Regressor** superó significativamente al modelo base (**Ridge Regression**):

| Modelo | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | 74.06 | 100.49 | 0.6811 (68.11%) |
| **Random Forest Regressor** | **29.34** | **48.02** | **0.9272 (92.72%)** |

**Conclusiones clave de los resultados:**
* **Reducción del error:** Random Forest redujo el error promedio (`MAE`) a menos de la mitad, pasando de ~74 a ~29 alquileres de diferencia respecto al valor real.
* **Mayor estabilidad:** El `RMSE` disminuyó de 100.49 a 48.02, lo que demuestra una importante reducción en la magnitud de los errores grandes.
* **Alto poder explicativo:** El modelo logró un `R²` de **0.9272**, explicando el **92.72%** de la variabilidad de la demanda de bicicletas, frente a solo el 68.11% del baseline.

## Instrucciones para ejecutar el notebook


