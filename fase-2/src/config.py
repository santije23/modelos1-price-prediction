"""Constantes del proyecto: esquema de columnas."""

TARGET = "cnt"
GROUP_COL = "dteday"  # solo para agrupar la partición; NO es predictora

NUM_COLS = ["temp", "hum", "windspeed"]
CAT_COLS = ["season", "weathersit", "mnth", "hr", "weekday", "yr", "holiday", "workingday"]
FEATURE_COLS = NUM_COLS + CAT_COLS

DATA_SOURCE = "https://www.kaggle.com/datasets/amil09/bike-sharing-dataset/data?select=Bike+Share+hourly.csv"
REQUIREMENTS_FILE = "requirements.txt"