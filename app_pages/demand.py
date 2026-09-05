import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from src.academy_ui import render_concept

from src.docs import apply_theme, render_standard
from src.labs import render_pipeline_overview
from src.demand_train import DAYS, PRODUCTS, FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"
REPORT_PATH = PROJECT_ROOT / "models" / "demand_report.json"


@st.cache_resource(show_spinner="Cargando modelo de demanda...")
def load_demand_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h")
def load_demand_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def render_variables_table() -> None:
    features = [
        ("dia_semana", "Categorica", "Lunes a Domingo", "Dia de la semana de la venta"),
        ("tipo_producto", "Categorica", "Electronica, Ropa, Alimentos, Hogar, Juguetes", "Categoria del producto"),
        ("precio", "Numerica (decimal)", "5 - 120", "Precio unitario del producto"),
        ("promocion_activa", "Binaria (0/1)", "0, 1", "Indica si hay una promocion activa"),
        ("volumen_ventas (objetivo)", "Continua (entera)", "> 0", "Unidades proyectadas a vender"),
    ]
    st.dataframe(pd.DataFrame(features, columns=["Variable", "Tipo", "Valores / Rango", "Descripcion"]), hide_index=True)
    st.caption("Dataset simulado de ventas determinista (20000 registros). Preprocesamiento: OneHotEncoder sobre las categoricas.")


apply_theme()

st.title("Pronóstico de Demanda (Regresión)")
st.caption("Modelo XGBRegressor para proyectar el volumen de ventas en unidades segun dia, producto, precio y promocion.")

tab_pred, tab_docs, tab_lab = st.tabs(["Pronóstico de Demanda", "Documentación del Ejercicio", "Laboratorio (A vs B)"], default="Pronóstico de Demanda")

with tab_pred:
    with st.container(border=True):
        st.subheader("Configuración de la venta")
        col1, col2 = st.columns(2)
        dia = col1.selectbox("Día de la semana", DAYS)
        producto = col2.selectbox("Tipo de producto", PRODUCTS)
        precio = st.number_input("Precio (moneda local)", min_value=5.0, max_value=120.0, value=50.0, step=1.0)
        promocion = st.checkbox("Promoción activa", value=False)

    with st.container(border=True):
        st.subheader("Unidades proyectadas")
        row = pd.DataFrame(
            [{"dia_semana": dia, "tipo_producto": producto, "precio": precio, "promocion_activa": int(promocion)}]
        )[FEATURES]
        pred = float(load_demand_model().predict(row)[0])
        st.metric("Volumen de ventas estimado (unidades)", f"{pred:,.0f}")
        st.caption(f"Día {dia} · {producto} · Precio {precio:,.2f} · {'Con promoción' if promocion else 'Sin promoción'}.")

with tab_docs:
    st.subheader("Variables predictoras")
    render_variables_table()

    st.divider()
    st.subheader("Cómo funciona el modelo")
    report = load_demand_report()

    with st.container(border=True):
        st.markdown("**¿Qué hace en general?**")
        st.write(
            "Modelo de **regresión continua**: dado el día, la categoría de producto, el precio y si hay promoción activa, "
            "estima cuántas unidades se venderán. A diferencia de una clasificación, la salida es un número real (volumen)."
        )
        st.markdown("**¿Cómo funciona técnicamente?**")
        st.write(
            "Pipeline de scikit-learn: ColumnTransformer con OneHotEncoder sobre dia_semana y tipo_producto (precio y promoción pasan "
            "tal cual) seguido de XGBRegressor (300 árboles, learning_rate=0.05, max_depth=5). La predicción se ejecuta con predict()."
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("R² (test)", f"{report['test']['r2']:.4f}")
        col_b.metric("RMSE (test)", f"{report['test']['rmse']:.2f}")
        col_c.metric("MAE (test)", f"{report['test']['mae']:.2f}")
        st.markdown("**Casos de uso adicionales**")
        st.write(
            "Planeación de inventario y reabastecimiento, asignación de personal por turno, promociones por categoría, "
            "presupuestos de venta por canal y detección de impacto de precios."
        )
        st.markdown("**Consideraciones**")
        st.write(
            "Datos simulados: el rendimiento real depende de la calidad histórica. La regresión se evalúa con R²/RMSE/MAE (no con exactitud). "
            "Conviene incluir estacionalidad, series temporales y variables exógenas (feriados, clima) para producción."
        )

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard(
        "En este ejercicio se aplica una versión de los pasos 1-9 (definición del objetivo de ventas, dataset simulado, "
        "split 80/20, XGBRegressor y métricas R²/RMSE/MAE) y el paso 11 con el despliegue en este dashboard."
    )

with tab_lab:
    st.subheader("Comparador de escenarios (A vs B)")
    st.caption("Compara dos configuraciones de venta y observa la diferencia de unidades proyectadas. Prueba el impacto del día, el precio y la promoción.")

    model = load_demand_model()

    def row_from(day, product, price, promo):
        return pd.DataFrame(
            [{"dia_semana": day, "tipo_producto": product, "precio": price, "promocion_activa": int(promo)}]
        )[FEATURES]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Escenario A")
        d_a = st.selectbox("Día A", DAYS, key="lab_a_day")
        p_a = st.selectbox("Producto A", PRODUCTS, key="lab_a_prod")
        price_a = st.number_input("Precio A", 5.0, 120.0, 50.0, step=1.0, key="lab_a_price")
        promo_a = st.checkbox("Promoción activa A", key="lab_a_promo")
    with col_b:
        st.markdown("#### Escenario B")
        d_b = st.selectbox("Día B", DAYS, index=4, key="lab_b_day")
        p_b = st.selectbox("Producto B", PRODUCTS, index=2, key="lab_b_prod")
        price_b = st.number_input("Precio B", 5.0, 120.0, 45.0, step=1.0, key="lab_b_price")
        promo_b = st.checkbox("Promoción activa B", value=True, key="lab_b_promo")

    pred_a = float(model.predict(row_from(d_a, p_a, price_a, promo_a))[0])
    pred_b = float(model.predict(row_from(d_b, p_b, price_b, promo_b))[0])
    diff = pred_b - pred_a

    c1, c2, c3 = st.columns(3)
    c1.metric("Unidades A", f"{pred_a:,.0f}", help=f"{d_a} · {p_a} · ${price_a:,.2f} · {'promo' if promo_a else 'sin promo'}")
    c2.metric("Unidades B", f"{pred_b:,.0f}", help=f"{d_b} · {p_b} · ${price_b:,.2f} · {'promo' if promo_b else 'sin promo'}")
    c3.metric("Diferencia (B − A)", f"{diff:+,.0f}", delta=f"{(diff / pred_a) * 100:+.1f}%" if pred_a else None)

    st.markdown(
        f"El escenario B proyecta **{pred_b:,.0f}** unidades frente a **{pred_a:,.0f}** del A "
        f"({'+' if diff > 0 else ''}{diff:,.0f} unidades). Cambia un solo factor a la vez para aislar su efecto."
    )

    with st.expander("Ver cómo se generó el dataset simulado"):
        st.write(
            "El dataset de entrenamiento se simuló con: `volumen = exp(ln(base) − 0.012·precio + 0.45·promocion_activa + ruido)`, "
            "donde `base = peso_día · peso_producto · 40`. El modelo XGBoost aprende ese patrón (más día pico, más producto demandado, "
            "precio bajo y promoción activa ⇒ más unidades)."
        )
        render_concept("pipeline")

    st.divider()
    render_pipeline_overview(model, title="Pipeline en vivo de demanda")
