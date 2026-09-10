import streamlit as st

from src import assistant_rag
from src.manifest import EXERCISE_GROUPS, EXERCISES, PORTAL_PAGES

st.set_page_config(
    page_title="Centro de Modelos de Machine Learning",
    page_icon=":material/model_training:",
    layout="wide",
    initial_sidebar_state="expanded",
)

portal_pages = [
    st.Page(
        p["page"],
        title=p["title"],
        icon=p["icon"],
        default=p.get("default", False),
        url_path=p.get("url_path"),
    )
    for p in PORTAL_PAGES
]

navigation: dict[str, list] = {"Portal": portal_pages}
for group in EXERCISE_GROUPS:
    navigation[group] = [
        st.Page(ex["page"], title=ex["title"], icon=ex["icon"], url_path=ex["url_path"])
        for ex in EXERCISES
        if ex["group"] == group
    ]

pg = st.navigation(navigation, expanded=True)
pg.run()


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.caption("Dónde encontrarlo en el sitio:")
    seen = set()
    for s in sources:
        key = (s["page"], s["tab"])
        if key in seen:
            continue
        seen.add(key)
        st.markdown(f"- **{s['page']}** → {s['tab']}")


@st.dialog("Asistente del portal", width="large", icon=":material/chat:")
def assistant_dialog() -> None:
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Hola! Pregúntame sobre los seis ejercicios, sus métricas, modelos o conceptos de Machine Learning. "
                    "Te responderé y te diré en qué parte del sitio encontrarlo."
                ),
            }
        ]

    if not assistant_rag.has_llm():
        st.caption("Modo sin LLM: respondo con búsqueda sobre el contenido del portal. Configura DEEPSEEK_API_KEY para respuestas generadas.")

    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Escribe tu pregunta sobre el portal...")
    if prompt:
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en el contenido del portal..."):
                result = assistant_rag.answer(prompt, st.session_state.assistant_messages)
            st.markdown(result["answer"])
            _render_sources(result["sources"])

        st.session_state.assistant_messages.append({"role": "assistant", "content": result["answer"]})


with st.container(key="assistant_fab"):
    if st.button("Asistente", icon=":material/chat:", key="open_assistant", type="primary"):
        assistant_dialog()
