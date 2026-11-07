"""Configuración de pytest para exponer fase-2/src como paquete importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
