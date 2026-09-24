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



## Métrica empleada



## Principales resultados obtenidos



## Instrucciones para ejecutar el notebook


