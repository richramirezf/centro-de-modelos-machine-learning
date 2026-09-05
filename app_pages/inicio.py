import streamlit as st

st.title("Centro de Modelos de Machine Learning")
st.caption("Plataforma unificada con la misma metodología por ejercicio: el caso de negocio, el modelo en acción y su documentación técnica completa.")

st.markdown(
    """
    Cada ejercicio sigue la **misma estructura**: predicción interactiva, análisis exploratorio (cuando aplica) y una pestaña de
    **documentación** con variables predictoras, funcionamiento técnico, métricas y el paso a paso estándar (CRISP-DM + MLOps).

    Selecciona un ejercicio desde la **barra lateral** o desde las tarjetas siguientes.
    """
)


def exercise_card(icon: str, title: str, description: str, button_label: str, page: str, key: str) -> None:
    st.markdown(f"## {icon} {title}")
    st.markdown(description)
    if st.button(button_label, key=key, icon=":material/open_in_new:", type="primary"):
        st.switch_page(page)


st.markdown("## Clasificación binaria")
col_a, col_b, col_c = st.columns(3, gap="large")
with col_a:
    exercise_card(
        ":material/credit_score:", "Scoring de Crédito",
        "Riesgo de **default crediticio** (German Credit, 1,000). LogReg + XGBoost, umbral dinámico y API FastAPI.",
        "Abrir Scoring de Crédito", "app_pages/credit_scoring.py", "go_credit",
    )
with col_b:
    exercise_card(
        ":material/support_agent:", "Churn Telco",
        "Abandono de clientes (IBM Telco, 7,043). XGBoost con **SHAP**, EDA y predicción por lotes.",
        "Abrir Churn Telco", "app_pages/telco_churn.py", "go_telco",
    )
with col_c:
    exercise_card(
        ":material/event_busy:", "Ausentismo Médico",
        "Inasistencias a citas (110,527). XGBoost **calibrado** con alertas por bandas de riesgo.",
        "Abrir Ausentismo Médico", "app_pages/noshow.py", "go_noshow",
    )

st.markdown("## Regresión y NLP")
col_d, col_e, col_f = st.columns(3, gap="large")
with col_d:
    exercise_card(
        ":material/storefront:", "Pronóstico de Demanda",
        "**Regresión**: volumen de ventas por día, producto, precio y promoción. XGBRegressor.",
        "Abrir Pronóstico de Demanda", "app_pages/demand.py", "go_demand",
    )
with col_e:
    exercise_card(
        ":material/home_work:", "Valuación Inmobiliaria",
        "**Regresión**: precio (USD) por superficie, habitaciones, antigüedad y garaje. XGBRegressor.",
        "Abrir Valuación Inmobiliaria", "app_pages/housing.py", "go_housing",
    )
with col_f:
    exercise_card(
        ":material/forum:", "Clasificador de Textos (NLP)",
        "**NLP**: intención de mensajes de chat (Soporte/Ventas/Reclamos/Horarios). TF-IDF + LogReg.",
        "Abrir Clasificador de Textos", "app_pages/intent.py", "go_intent",
    )

st.divider()

st.markdown(
    """
    **Metodología común aplicada en los seis ejercicios:**

    | Ejercicio | Tipo | Modelo | Preprocesamiento |
    |---|---|---|---|
    | Scoring de Crédito | Clasificación binaria | LogReg + XGBoost | OneHot + StandardScaler |
    | Churn Telco | Clasificación binaria | XGBoost + SHAP | OneHot + StandardScaler |
    | Ausentismo Médico | Clasificación binaria | XGBoost calibrado | StandardScaler |
    | Pronóstico de Demanda | Regresión continua | XGBRegressor | OneHotEncoder |
    | Valuación Inmobiliaria | Regresión continua | XGBRegressor | StandardScaler |
    | Clasificador de Textos | NLP multiclase | TF-IDF + LogReg | TfidfVectorizer |
    """
)
