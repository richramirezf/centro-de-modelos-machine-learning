import streamlit as st

st.set_page_config(
    page_title="ML Hub · Scoring & Churn",
    page_icon=":material/model_training:",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("app_pages/inicio.py", title="Inicio", icon=":material/home:", default=True)
credito = st.Page(
    "app_pages/credit_scoring.py",
    title="Scoring de Crédito",
    icon=":material/credit_score:",
    url_path="credito",
)
churn = st.Page(
    "app_pages/telco_churn.py",
    title="Churn de Telecomunicaciones",
    icon=":material/support_agent:",
    url_path="churn",
)

pg = st.navigation(
    {
        "Portal": [home],
        "Ejercicios de Machine Learning": [credito, churn],
    },
    expanded=True,
)
pg.run()
