import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.docs import apply_theme, render_standard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "nlp_intent_model.joblib"
REPORT_PATH = PROJECT_ROOT / "models" / "nlp_intent_report.json"

SAMPLE_MESSAGES = {
    "Horarios": "Hola, a que hora abren manana?",
    "Ventas": "Hola, cuanto cuesta el plan premium?",
    "Reclamos": "Me cobraron de mas en mi ultima factura.",
    "Soporte_Tecnico": "Hola, mi internet no funciona desde ayer.",
}

INTENT_LABELS = {
    "Soporte_Tecnico": "Soporte Técnico",
    "Ventas": "Ventas / Comercial",
    "Reclamos": "Reclamos",
    "Horarios": "Horarios de atención",
}


@st.cache_resource(show_spinner="Cargando modelo de NLP...")
def load_intent_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h")
def load_intent_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def render_intent_table() -> None:
    examples = [
        ("Soporte_Tecnico", "Problemas técnicos: fallas, errores, soporte de producto/servicio"),
        ("Ventas", "Consultas comerciales: precios, contratación, planes, cotizaciones"),
        ("Reclamos", "Inconformidades: cobros, devoluciones, quejas, atención al cliente"),
        ("Horarios", "Preguntas sobre horarios de apertura y atención"),
    ]
    st.dataframe(
        pd.DataFrame(examples, columns=["Intención", "Qué cubre"]),
        hide_index=True,
    )


apply_theme()

st.title("Clasificador de Intenciones (NLP)")
st.caption("TF-IDF + Regresión Logística sobre mensajes de chat para detectar la intención del cliente: Soporte Técnico, Ventas, Reclamos u Horarios.")

tab_pred, tab_docs = st.tabs(["Clasificar Mensaje", "Documentación del Ejercicio"], default="Clasificar Mensaje")

with tab_pred:
    with st.container(border=True):
        st.subheader("Mensaje entrante")
        ejemplo = st.selectbox("Cargar ejemplo", list(SAMPLE_MESSAGES.keys()), format_func=lambda k: INTENT_LABELS[k])
        texto = st.text_area("Pega aquí el mensaje del cliente", value=SAMPLE_MESSAGES[ejemplo], height=150)
        st.caption(f"{len(texto.split())} palabras · {len(texto)} caracteres")

    if st.button("Clasificar Intención", type="primary", icon=":material/smart_toy:", key="evaluate"):
        if texto.strip():
            model = load_intent_model()
            proba = model.predict_proba([texto])[0]
            clf = model.named_steps["clf"]
            intencion = clf.classes_[int(proba.argmax())]
            confianza = float(proba.max())

            with st.container(border=True):
                st.markdown(f"### Intención detectada: :teal[**{INTENT_LABELS[intencion]}**]")
                st.metric("Confianza de la clasificación", f"{confianza * 100:.1f}%")

                st.subheader("Distribución de probabilidades por intención")
                prob_df = pd.DataFrame(
                    {
                        "Intención": [INTENT_LABELS[c] for c in clf.classes_],
                        "Probabilidad": proba,
                    }
                ).sort_values("Probabilidad", ascending=False)
                st.bar_chart(prob_df.set_index("Intención"), y="Probabilidad")
        else:
            st.warning("Escribe un mensaje antes de clasificar.")
    else:
        st.markdown("<div class='empty-hint'>Pega un mensaje y pulsa \u201CClasificar Intención\u201D para ver la categoría y la confianza.</div>", unsafe_allow_html=True)

with tab_docs:
    st.subheader("Categorías de intención")
    render_intent_table()

    st.divider()
    st.subheader("Cómo funciona el modelo")
    report = load_intent_report()

    with st.container(border=True):
        st.markdown("**¿Qué hace en general?**")
        st.write(
            "Clasifica **texto libre** (mensajes de chat, tickets, redes sociales) en una de cuatro intenciones de negocio. "
            "Convierte el texto en números (TF-IDF) y aprende qué palabras distinguen cada intención."
        )
        st.markdown("**¿Cómo funciona técnicamente?**")
        st.write(
            "Pipeline de scikit-learn: TfidfVectorizer (n-gramas 1-2, sublinear_tf) transforma el mensaje en una matriz de pesos "
            "por término, y LogisticRegression multinomial estima P(intención | texto). Se usan predict() para la clase y "
            "predict_proba() para la confianza (probabilidad máxima)."
        )
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy (test)", f"{report['test']['accuracy']:.4f}")
        col_b.metric("Muestras de entrenamiento", f"{report['n_samples']}")
        st.markdown("**Casos de uso adicionales**")
        st.write(
            "Chatbots y enrutamiento automático de tickets, priorización de reclamos, análisis de mensajes en redes sociales, "
            "clasificación de correos entrantes y detección de urgencias."
        )
        st.markdown("**Consideraciones**")
        st.write(
            "Dataset simulado de frases cortas en español; en producción se necesita más variedad, léxico real y gestión de "
            "idiomas/errores ortográficos. La accuracy es muy alta por la separación léxica del corpus simulado; en producción "
            "conviene monitorear la distribución de mensajes y reentrenar ante deriva."
        )

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard(
        "En este ejercicio se aplica una versión de los pasos 1-9 (objetivo de enrutamiento, corpus simulado de intenciones, "
        "split 80/20 estratificado, TF-IDF + Regresión Logística y exactitud por clase) y el paso 11 con el despliegue en este dashboard."
    )
