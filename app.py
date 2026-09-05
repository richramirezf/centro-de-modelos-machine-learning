import streamlit as st

st.set_page_config(
    page_title="Centro de Modelos de Machine Learning",
    page_icon=":material/model_training:",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("app_pages/inicio.py", title="Inicio", icon=":material/home:", default=True)
aprende = st.Page(
    "app_pages/aprende.py",
    title="Aprende ML",
    icon=":material/school:",
    url_path="aprende",
)

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
noshow = st.Page(
    "app_pages/noshow.py",
    title="Ausentismo Médico (No-Show)",
    icon=":material/event_busy:",
    url_path="noshow",
)

demand = st.Page(
    "app_pages/demand.py",
    title="Pronóstico de Demanda",
    icon=":material/storefront:",
    url_path="demanda",
)
housing = st.Page(
    "app_pages/housing.py",
    title="Valuación Inmobiliaria",
    icon=":material/home_work:",
    url_path="vivienda",
)
intent = st.Page(
    "app_pages/intent.py",
    title="Clasificador de Textos (NLP)",
    icon=":material/forum:",
    url_path="nlp",
)

pg = st.navigation(
    {
        "Portal": [home, aprende],
        "Clasificación Binaria": [credito, churn, noshow],
        "Regresión & NLP": [demand, housing, intent],
    },
    expanded=True,
)
pg.run()
