import streamlit as st

st.title("Centro de Modelos de Machine Learning")
st.caption("Plataforma unificada con la misma metodología por ejercicio: el caso de negocio, el modelo en acción y su documentación técnica completa.")

st.markdown(
    """
    Cada ejercicio sigue la **misma estructura**: predicción interactiva, análisis exploratorio y una pestaña de
    **documentación de modelos** con variables predictoras, funcionamiento técnico, matriz de confusión de
    entrenamiento y el paso a paso estándar (CRISP-DM + MLOps).

    Selecciona un ejercicio desde la **barra lateral** o desde las tarjetas siguientes.
    """
)

col_cr, col_tc, col_ns = st.columns(3, gap="large")

with col_cr:
    st.markdown("## :material/credit_score: Scoring de Crédito")
    st.markdown(
        """
        **Clasificación de riesgo crediticio** sobre el German Credit Dataset (1,000 solicitudes).

        - Motores: Regresión Logística (interpretable) y XGBoost.
        - Umbral de decisión dinámico según apetito de riesgo.
        - Microservicio FastAPI (`src/api.py`).
        """
    )
    if st.button("Abrir Scoring de Crédito", key="go_credit", icon=":material/open_in_new:", type="primary"):
        st.switch_page("app_pages/credit_scoring.py")

with col_tc:
    st.markdown("## :material/support_agent: Churn Telco")
    st.markdown(
        """
        **Predicción de abandono de clientes** sobre Telco Customer Churn de IBM (7,043 clientes).

        - Motor: XGBoost con explicabilidad SHAP (waterfall).
        - EDA interactivo, predicción por cliente y por lote (CSV).
        - Retención proactiva antes de que el cliente se vaya.
        """
    )
    if st.button("Abrir Churn Telco", key="go_telco", icon=":material/open_in_new:", type="primary"):
        st.switch_page("app_pages/telco_churn.py")

with col_ns:
    st.markdown("## :material/event_busy: Ausentismo Médico")
    st.markdown(
        """
        **Predicción de inasistencias (No-Show)** sobre citas médicas (110,527 registros).

        - Motor: XGBoost balanceado hacia la clase No-Show.
        - Edad, tiempo de espera, condiciones previas y SMS.
        - Alertas visuales de riesgo para priorizar recordatorios.
        """
    )
    if st.button("Abrir Ausentismo Médico", key="go_noshow", icon=":material/open_in_new:", type="primary"):
        st.switch_page("app_pages/noshow.py")

st.divider()

st.markdown(
    """
    **Metodología común aplicada en los tres ejercicios:**

    | Fase | Crédito (Scoring) | Telco (Churn) | Médico (No-Show) |
    |---|---|---|---|
    | Preprocesamiento | OneHot + StandardScaler | OneHot + StandardScaler | StandardScaler |
    | Partición | 80/20 estratificada | 80/20 estratificada | 80/20 estratificada |
    | Modelo(s) | LogReg + XGBoost | XGBoost + SHAP | XGBoost balanceado |
    | Decisión | Umbral "Apetito de Riesgo" | Probabilidad de churn | Bandas de riesgo |
    | Documentación | Variables, matriz, teoría, estándar | Variables, matriz, teoría, estándar | Variables, matriz, teoría, estándar |
    """
)
