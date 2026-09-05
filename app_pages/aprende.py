import streamlit as st

from src.academy_content import FAQ, QUIZZES, QUIZ_EXERCISE_TITLES, ROUTE
from src.academy_ui import (
    render_choose_model,
    render_faq,
    render_glossary,
    render_quiz,
    render_route,
)
from src.docs import apply_theme

apply_theme()

st.title("Aprende Machine Learning")
st.caption("Ruta sugerida, glosario, guía para elegir modelo y quizzes con corrección en vivo. Todo conectado con los seis ejercicios del portal.")

tab_route, tab_glossary, tab_guide, tab_quiz, tab_faq = st.tabs(
    [
        "Ruta de aprendizaje",
        "Glosario",
        "¿Qué modelo elegir?",
        "Quizzes",
        "FAQ técnico",
    ]
)

with tab_route:
    st.subheader("Ruta de aprendizaje")
    render_route(ROUTE)

with tab_glossary:
    st.subheader("Glosario de Machine Learning")
    render_glossary()

with tab_guide:
    st.subheader("¿Cómo elegir el modelo adecuado?")
    render_choose_model()

with tab_quiz:
    st.subheader("Pon a prueba lo aprendido")
    st.caption("Elige un quiz: cada ejercicio tiene preguntas específicas y hay un quiz general al final.")
    ex_keys = [key for key, _ in QUIZ_EXERCISE_TITLES]
    titles = {key: label for key, label in QUIZ_EXERCISE_TITLES}

    requested = st.query_params.get("quiz", None)
    default_key = requested if requested in ex_keys else "credito"

    selection = st.selectbox(
        "Quiz",
        options=[titles[k] for k in ex_keys],
        index=ex_keys.index(default_key),
        label_visibility="collapsed",
        key="quiz_select",
    )
    ex_key = next(k for k, label in QUIZ_EXERCISE_TITLES if label == selection)
    render_quiz(prefix=f"quiz_{ex_key}", bank=QUIZZES[ex_key], title=selection)

with tab_faq:
    st.subheader("Preguntas frecuentes y errores comunes")
    render_faq(FAQ)
