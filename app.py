import streamlit as st

from src import assistant_rag

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


st.html(
    """
    <style>
      .st-key-assistant_fab {
        position: fixed;
        bottom: 1.25rem;
        right: 1.25rem;
        z-index: 9999;
        width: auto;
      }
      .st-key-assistant_fab button {
        border-radius: 999px !important;
        padding: .65rem 1.1rem !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, .25) !important;
        font-weight: 600;
      }
    </style>
    """
)


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
