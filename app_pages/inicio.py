import streamlit as st

from src.manifest import EXERCISE_GROUPS, EXERCISES
from src.theme import apply_theme

apply_theme()

st.title("Centro de Modelos de Machine Learning")
st.caption("Plataforma unificada con la misma metodología por ejercicio: el caso de negocio, el modelo en acción y su documentación técnica completa.")

st.markdown(
    """
    Cada ejercicio sigue la **misma estructura**: predicción interactiva, análisis exploratorio (cuando aplica) y una pestaña de
    **documentación** con variables predictoras, funcionamiento técnico, métricas y el paso a paso estándar (CRISP-DM + MLOps).

    Selecciona un ejercicio desde la **barra lateral** o desde las tarjetas siguientes.
    """
)

with st.container(border=True):
    col_title, col_go = st.columns([5, 1])
    col_title.markdown("## :material/school: Aprende Machine Learning")
    col_title.markdown(
        "Ruta de aprendizaje, glosario con buscador, guía para elegir modelo, **quizzes con corrección en vivo** y FAQ técnico."
    )
    if col_go.button("Ir a Aprende ML", key="go_aprende", icon=":material/open_in_new:", type="primary"):
        st.switch_page("app_pages/aprende.py")


def exercise_card(exercise: dict) -> None:
    st.markdown(f"## {exercise['icon']} {exercise['title']}")
    st.markdown(exercise["description"])
    if st.button(
        f"Abrir {exercise['title']}",
        key=f"go_{exercise['id']}",
        icon=":material/open_in_new:",
        type="primary",
    ):
        st.switch_page(exercise["page"])


for group in EXERCISE_GROUPS:
    group_exercises = [ex for ex in EXERCISES if ex["group"] == group]
    st.markdown(f"## {group}")
    columns = st.columns(len(group_exercises), gap="large")
    for column, exercise in zip(columns, group_exercises):
        with column:
            exercise_card(exercise)

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
