"""Cargadores genéricos y cacheados de modelos (.joblib) y reportes (.json).

Como la clave de caché es el nombre del archivo (argumento), cada modelo/reporte
tiene su propia entrada y se elimina la duplicación de funciones ``load_*`` en
las páginas.
"""

import json
from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(filename: str):
    """Carga (y cachea) un pipeline ``.joblib`` desde ``models/``."""
    return joblib.load(MODELS_DIR / filename)


@st.cache_data(ttl="1h")
def load_report(filename: str) -> dict:
    """Carga (y cachea) un reporte JSON desde ``models/``."""
    return json.loads((MODELS_DIR / filename).read_text(encoding="utf-8"))
