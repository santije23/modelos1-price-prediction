"""Instancia de la API que arranca uvicorn.

Desde la raíz del repositorio:
    uvicorn api.server:app --app-dir fase-3 --host 0.0.0.0 --port 8000
"""

from api.main import create_app

app = create_app()
