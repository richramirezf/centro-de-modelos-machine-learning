import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.docs import apply_theme, render_standard
from src.housing_train import FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "housing_model.joblib"
REPORT_PATH = PROJECT_ROOT / "models" / "housing_report.json"


@st.cache_resource(show_spinner="Cargando modelo de valuación...")
def load_housing_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h")
def load_housing_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def render_variables_table() -> None:
    features = [
        ("metros_cuadrados", "Numerica (entera)", "40 - 300", "Superficie de la propiedad"),
        ("habitaciones", "Numerica (entera)", "1 - 6", "Numero de habitaciones"),
        ("antiguedad_anios", "Numerica (entera)", "0 - 50", "Antiguedad de la propiedad en anios"),
        ("tiene_garaje", "Binaria (0/1)", "0, 1", "Indica si la propiedad tiene garaje"),
        ("precio_usd (objetivo)", "Continua", "USD", "Precio estimado de la propiedad"),
    ]
    st.dataframe(pd.DataFrame(features, columns=["Variable", "Tipo", "Valores / Rango", "Descripcion"]), hide_index=True)
    st.caption("Dataset simulado de propiedades determinista (15000 registros). Preprocesamiento: StandardScaler sobre las numericas.")


apply_theme()

st.title("Valuación Inmobiliaria (Regresión)")
st.caption("Modelo XGBRegressor para estimar el precio (USD) de una propiedad segun superficie, habitaciones, antiguedad y garaje.")

tab_pred, tab_docs = st.tabs(["Valuación de Propiedad", "Documentación del Ejercicio"], default="Valuación de Propiedad")

with tab_pred:
    with st.container(border=True):
        st.subheader("Características de la propiedad")
        col1, col2 = st.columns(2)
        metros = col1.slider("Metros cuadrados", min_value=40, max_value=300, value=120, step=5)
        habitaciones = col2.slider("Habitaciones", min_value=1, max_value=6, value=3)
        antiguedad = st.slider("Antigüedad (años)", min_value=0, max_value=50, value=10)
        garaje = st.selectbox("Garaje", ["No", "Sí"], index=1)

    with st.container(border=True):
        st.subheader("Precio estimado")
        row = pd.DataFrame(
            [{"metros_cuadrados": metros, "habitaciones": habitaciones, "antiguedad_anios": antiguedad, "tiene_garaje": int(garaje == "Sí")}]
        )[FEATURES]
        pred = float(load_housing_model().predict(row)[0])

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Precio estimado de la propiedad", f"${pred:,.0f} USD")
        col_m2.metric("Precio por m²", f"${pred / metros:,.0f} USD/m²")
        st.caption(f"{metros} m² · {habitaciones} hab. · {antiguedad} años · {'Con garaje' if garaje == 'Sí' else 'Sin garaje'}.")

with tab_docs:
    st.subheader("Variables predictoras")
    render_variables_table()

    st.divider()
    st.subheader("Cómo funciona el modelo")
    report = load_housing_report()

    with st.container(border=True):
        st.markdown("**¿Qué hace en general?**")
        st.write(
            "Modelo de **regresión continua** que estima el valor de mercado de una propiedad en dólares a partir de sus "
            "características físicas. La salida es un precio (número real), no una clase."
        )
        st.markdown("**¿Cómo funciona técnicamente?**")
        st.write(
            "Pipeline de scikit-learn: ColumnTransformer que aplica StandardScaler a metros_cuadrados, habitaciones y "
            "antiguedad_anios (tiene_garaje pasa tal cual) seguido de XGBRegressor (400 árboles, learning_rate=0.05, max_depth=5). "
            "La predicción se ejecuta con predict()."
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("R² (test)", f"{report['test']['r2']:.4f}")
        col_b.metric("RMSE (test)", f"${report['test']['rmse']:,.0f}")
        col_c.metric("MAE (test)", f"${report['test']['mae']:,.0f}")
        st.markdown("**Casos de uso adicionales**")
        st.write(
            "Valoración automatizada (AVM) para portales inmobiliarios, soporte a tasaciones, análisis de inversión, "
            "pricing de carteras hipotecarias y detección de propiedades infravaloradas."
        )
        st.markdown("**Consideraciones**")
        st.write(
            "Dataset simulado: en producción se usan datos reales de transacciones con ubicación y más atributos. "
            "La valuación se evalúa con R²/RMSE/MAE; el error absoluto (~20k USD) debe contrastarse con el rango de precios del mercado."
        )

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard(
        "En este ejercicio se aplica una versión de los pasos 1-9 (objetivo de valuación, dataset simulado, split 80/20, "
        "escalado + XGBRegressor y métricas R²/RMSE/MAE) y el paso 11 con el despliegue en este dashboard."
    )
