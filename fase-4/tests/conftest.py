"""Reutiliza el modelo pequeño de las pruebas de fase 3."""

import importlib.util
import sys
from pathlib import Path

FASE4 = Path(__file__).resolve().parents[1]
REPO_ROOT = FASE4.parent

for path in (FASE4, REPO_ROOT / "fase-3", REPO_ROOT / "fase-2" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_spec = importlib.util.spec_from_file_location(
    "fase3_fixtures", REPO_ROOT / "fase-3" / "tests" / "conftest.py"
)
fase3_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fase3_fixtures)

small_settings = fase3_fixtures.small_settings
VALID_RECORD = fase3_fixtures.VALID_RECORD
