"""Renderizadores de contenido educativo reutilizables (glosario, ruta,
guía, FAQ y quizzes con feedback en vivo)."""

import streamlit as st

from src.academy_content import CONCEPTS, GLOSSARY, GLOSSARY_CATEGORIES


def render_concept(key: str | None, level: int = 4) -> None:
    """Render a reusable mini-lesson expander from CONCEPTS."""
    if not key or key not in CONCEPTS:
        return
    title, text = CONCEPTS[key]
    level_marks = "#" * level
    with st.expander(f"Concepto clave: {title}"):
        st.markdown(text)


def render_glossary() -> None:
    st.markdown(
        "Glosario de términos usados en el portal. Usa el buscador o filtra por categoría."
    )

    search = st.text_input(
        "Buscar término", placeholder="ej. calibración, umbral, TF-IDF...",
        label_visibility="collapsed",
    ).strip().lower()

    categories = ["Todas"] + GLOSSARY_CATEGORIES
    category = st.selectbox("Categoría", categories, key="glossary_cat")

    items = GLOSSARY
    if category != "Todas":
        items = [g for g in items if g["category"] == category]
    if search:
        items = [
            g for g in items
            if search in g["term"].lower() or search in g["definition"].lower()
        ]

    if not items:
        st.info("Sin resultados. Prueba con otro término o categoría.")
        return

    for g in items:
        st.markdown(f"- **{g['term']}** ({g['category']}): {g['definition']}")

    st.divider()
    st.caption(f"{len(items)} término(s) mostrados de {len(GLOSSARY)} en total.")


def render_route(route: list[dict]) -> None:
    st.markdown(
        """
        Ruta sugerida para aprender en orden. Primero se entra en **regresión y NLP** (fundamentos de pipelines),
        luego en **clasificación** con métricas, umbral y calibración. Nivel de dificultad: Básico → Intermedio → Avanzado.
        """
    )

    for item in route:
        with st.container(border=True):
            head, tail = st.columns([5, 1])
            level_colors = {"Básico": "green", "Intermedio": "orange", "Avanzado": "red"}
            tag = f":{level_colors.get(item['level'], 'gray')}[{item['level']}]"
            head.markdown(f"### {item['order']}. {item['title']} {tag}")
            head.markdown(f"**Tipo:** {item['type']} — **Objetivo:** {item['goal']}")
            for topic in item["learn"]:
                head.markdown(f"- {topic}")
            if tail.button("Abrir", key=f"route_{item['id']}", icon=":material/open_in_new:", type="secondary"):
                st.switch_page(item["page"])


def render_choose_model() -> None:
    import pandas as pd

    from src.academy_content import CHOOSE_MODEL_INTRO, CHOOSE_MODEL_TABLE

    st.markdown(CHOOSE_MODEL_INTRO)
    st.dataframe(pd.DataFrame(CHOOSE_MODEL_TABLE), hide_index=True)

    st.markdown(
        """
        **Regla práctica:** empieza con el modelo más simple que resuelva el problema (baseline → lineal),
        mide con la métrica correcta y solo escala a un ensamble si aporta una mejora real en datos de test.
        """
    )


def render_faq(faq: list[dict]) -> None:
    st.markdown("Preguntas frecuentes y errores conceptuales comunes.")
    for i, entry in enumerate(faq):
        with st.expander(entry["q"], icon=":material/help_outline:"):
            st.write(entry["a"])


def render_quiz(prefix: str, bank: list[dict], title: str) -> None:
    """Quiz de opción múltiple con feedback en vivo por pregunta."""
    st.markdown(f"### {title}")
    st.caption(
        "Selecciona una opción en cada pregunta: verás la corrección y la explicación al instante."
    )

    answered = 0
    correct = 0
    for i, q in enumerate(bank):
        options = q["opts"]
        key = f"{prefix}_q{i}"
        choice = st.radio(q["q"], options, index=None, key=key, label_visibility="visible")
        if choice is None:
            continue
        answered += 1
        is_correct = choice == options[q["answer"]]
        correct += int(is_correct)
        if is_correct:
            st.markdown(":green[**Correcto ✔**]")
        else:
            st.markdown(
                f":red[**Incorrecto ✘**] — Respuesta: **{options[q['answer']]}**"
            )
        with st.expander("¿Por qué?", icon=":material/lightbulb:"):
            st.write(q["why"])
            render_concept(q.get("concept"), level=5)
        st.divider()

    if answered == len(bank):
        st.metric("Resultado", f"{correct}/{len(bank)} aciertos", f"{correct / len(bank):.0%}")
        if correct == len(bank):
            st.success("¡Perfecto! Dominas este ejercicio.")
        elif correct >= len(bank) * 0.6:
            st.info("Buen resultado. Repasa las explicaciones de los fallos y vuelve a intentarlo.")
        else:
            st.warning("Revisa la documentación del ejercicio y vuelve a intentarlo.")
    else:
        st.progress(answered / len(bank), text=f"Respondidas {answered} de {len(bank)}")

    if answered or any(f"{prefix}_q{i}" in st.session_state for i in range(len(bank))):
        if st.button("Reiniciar quiz", key=f"{prefix}_reset", icon=":material/restart_alt:"):
            for i in range(len(bank)):
                st.session_state.pop(f"{prefix}_q{i}", None)
            st.rerun()
