from pathlib import Path

import pandas as pd
import streamlit as st
from src.academy_ui import render_concept

from src.docs import render_features_table as render_features_table_shared
from src.docs import render_model_doc, render_standard
from src.housing_train import FEATURES
from src.labs import render_pipeline_overview
from src.loaders import load_model, load_report
from src.theme import apply_theme

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_housing_model():
    return load_model("housing_model.joblib")


def load_housing_report() -> dict:
    return load_report("housing_report.json")


def render_variables_table() -> None:
    rows = [
        ("metros_cuadrados", "Numerica (entera)", "40 - 300", "Superficie de la propiedad"),
        ("habitaciones", "Numerica (entera)", "1 - 6", "Numero de habitaciones"),
        ("antiguedad_anios", "Numerica (entera)", "0 - 50", "Antiguedad de la propiedad en anios"),
        ("tiene_garaje", "Binaria (0/1)", "0, 1", "Indica si la propiedad tiene garaje"),
        ("precio_usd (objetivo)", "Continua", "USD", "Precio estimado de la propiedad"),
    ]
    render_features_table_shared(
        rows,
        "Dataset simulado de propiedades determinista (15000 registros). Preprocesamiento: StandardScaler sobre las numericas.",
    )


apply_theme()

st.title("Valuación Inmobiliaria (Regresión)")
st.caption("Modelo XGBRegressor para estimar el precio (USD) de una propiedad segun superficie, habitaciones, antiguedad y garaje.")

tab_pred, tab_docs, tab_lab = st.tabs(["Valuación de Propiedad", "Documentación del Ejercicio", "Laboratorio (A vs B)"], default="Valuación de Propiedad")

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

    render_model_doc(
        general=(
            "Modelo de **regresión continua** que estima el valor de mercado de una propiedad en dólares a partir de sus "
            "características físicas. La salida es un precio (número real), no una clase."
        ),
        technical=(
            "Pipeline de scikit-learn: ColumnTransformer que aplica StandardScaler a metros_cuadrados, habitaciones y "
            "antiguedad_anios (tiene_garaje pasa tal cual) seguido de XGBRegressor (400 árboles, learning_rate=0.05, max_depth=5). "
            "La predicción se ejecuta con predict()."
        ),
        metrics={
            "R² (test)": f"{report['test']['r2']:.4f}",
            "RMSE (test)": f"${report['test']['rmse']:,.0f}",
            "MAE (test)": f"${report['test']['mae']:,.0f}",
        },
        use_cases=(
            "Valoración automatizada (AVM) para portales inmobiliarios, soporte a tasaciones, análisis de inversión, "
            "pricing de carteras hipotecarias y detección de propiedades infravaloradas."
        ),
        considerations=(
            "Dataset simulado: en producción se usan datos reales de transacciones con ubicación y más atributos. "
            "La valuación se evalúa con R²/RMSE/MAE; el error absoluto (~20k USD) debe contrastarse con el rango de precios del mercado."
        ),
    )

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard(
        "En este ejercicio se aplica una versión de los pasos 1-9 (objetivo de valuación, dataset simulado, split 80/20, "
        "escalado + XGBRegressor y métricas R²/RMSE/MAE) y el paso 11 con el despliegue en este dashboard."
    )

with tab_lab:
    st.subheader("Comparador de propiedades (A vs B)")
    st.caption("Configura dos propiedades y compara su precio estimado. Cambia una sola característica para aislar su efecto (garaje, m², antigüedad…).")

    model = load_housing_model()

    def price_of(metros, hab, ant, gar):
        row = pd.DataFrame(
            [{"metros_cuadrados": metros, "habitaciones": hab, "antiguedad_anios": ant, "tiene_garaje": int(gar)}]
        )[FEATURES]
        return float(model.predict(row)[0])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Propiedad A")
        m_a = st.slider("m² A", 40, 300, 100, 5, key="lab_a_m2")
        h_a = st.slider("Habitaciones A", 1, 6, 3, key="lab_a_hab")
        a_a = st.slider("Antigüedad A", 0, 50, 20, key="lab_a_ant")
        g_a = st.selectbox("Garaje A", ["No", "Sí"], key="lab_a_gar")
    with col_b:
        st.markdown("#### Propiedad B")
        m_b = st.slider("m² B", 40, 300, 120, 5, key="lab_b_m2")
        h_b = st.slider("Habitaciones B", 1, 6, 3, key="lab_b_hab")
        a_b = st.slider("Antigüedad B", 0, 50, 10, key="lab_b_ant")
        g_b = st.selectbox("Garaje B", ["No", "Sí"], key="lab_b_gar", index=1)

    price_a = price_of(m_a, h_a, a_a, g_a == "Sí")
    price_b = price_of(m_b, h_b, a_b, g_b == "Sí")
    diff = price_b - price_a

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio A", f"${price_a:,.0f}")
    c2.metric("Precio B", f"${price_b:,.0f}")
    c3.metric("Diferencia (B − A)", f"${diff:+,.0f}", delta=f"{(diff / price_a) * 100:+.1f}%" if price_a else None)

    st.markdown(
        f"La propiedad B vale **${price_b:,.0f}** frente a **${price_a:,.0f}** de la A "
        f"({'+' if diff > 0 else ''}${diff:,.0f}). El modelo aprendió que el garaje y los m² suman; la antigüedad resta."
    )

    with st.expander("Ver cómo se generó el dataset simulado"):
        st.write(
            "El dataset se simuló con: `precio = 25,000 + 950·m² + 12,000·habitaciones − 1,300·antigüedad + 22,000·garaje + ruido`. "
            "Por eso el efecto del garaje (~+22k USD) o de cada m² (~950 USD) es casi lineal: XGBoost lo recupera de los datos."
        )
        render_concept("pipeline")

    st.divider()
    render_pipeline_overview(model, title="Pipeline en vivo de valuación")
