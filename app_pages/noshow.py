import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.docs import render_confusion_theory, render_standard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "noshow_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSION_PATH = PROJECT_ROOT / "models" / "confusion_report_noshow.json"

FEATURES = ["Age", "WaitTime_Days", "Scholarship", "Hipertension", "Diabetes", "SMS_received"]

LOW_BAND_PCT = 25
HIGH_BAND_PCT = 40


def apply_clean_ui() -> None:
    st.html(
        """
        <style>
          :root {
            --teal-900: #004d40;
            --teal-700: #00796b;
            --text-dark: #263238;
            --text-muted: #546e7a;
            --surface: #ffffff;
            --shadow-1: 0 1px 2px rgba(0, 0, 0, .05), 0 1px 3px rgba(0, 0, 0, .10);
            --shadow-2: 0 2px 4px rgba(0, 0, 0, .08), 0 4px 12px rgba(0, 0, 0, .12);
          }

          [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f0f7f6 0%, #e8f1f0 100%);
          }

          h1, h2, h3 { color: var(--teal-900) !important; letter-spacing: .2px; }

          [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--teal-700);
            font-weight: 600;
          }

          [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid rgba(0, 121, 107, .18) !important;
            border-radius: 16px;
            box-shadow: var(--shadow-2);
            padding: .35rem .6rem;
          }

          .result-card {
            background: var(--surface);
            border-radius: 16px;
            box-shadow: var(--shadow-2);
            padding: 1.5rem 2rem;
            text-align: center;
          }
          .result-card.low      { border-top: 5px solid #1b7f3b; }
          .result-card.moderate { border-top: 5px solid #b45309; }
          .result-card.high     { border-top: 5px solid #c3352b; }

          .verdict { font-size: 1.5rem; font-weight: 700; letter-spacing: .3px; }
          .verdict.low      { color: #1b7f3b; }
          .verdict.moderate { color: #b45309; }
          .verdict.high     { color: #c3352b; }

          .verdict-sub {
            color: var(--text-muted);
            font-size: 1rem;
            font-weight: 500;
            margin-top: .15rem;
          }

          .prob-value { font-size: 3.2rem; font-weight: 700; line-height: 1.1; margin: .25rem 0; }
          .prob-value.low      { color: #1b7f3b; }
          .prob-value.moderate { color: #b45309; }
          .prob-value.high     { color: #c3352b; }

          .prob-label {
            color: var(--text-muted);
            font-size: .9rem;
            letter-spacing: .4px;
            text-transform: uppercase;
          }

          .empty-hint { color: var(--text-muted); }

          .st-key-evaluate button[kind="primary"] {
            background: var(--teal-700);
            border-radius: 12px;
            box-shadow: var(--shadow-1);
            font-weight: 600;
          }
          .st-key-evaluate button[kind="primary"]:hover {
            background: var(--teal-900);
            box-shadow: var(--shadow-2);
          }
        </style>
        """
    )


@st.cache_resource(show_spinner="Cargando el modelo de no-show...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h")
def load_confusion_report() -> dict:
    return json.loads(CONFUSION_PATH.read_text(encoding="utf-8"))


def render_verdict(prob: float) -> None:
    pct = prob * 100
    if pct < LOW_BAND_PCT:
        tone, title, subtitle = "low", "ASISTENCIA ESPERADA", "Riesgo bajo de inasistencia a la cita"
        icon = ":material/verified:"
    elif pct < HIGH_BAND_PCT:
        tone, title, subtitle = "moderate", "RIESGO MODERADO DE INASISTENCIA", "Considere un recordatorio adicional o seguimiento"
        icon = ":material/warning:"
    else:
        tone, title, subtitle = "high", "ALTO RIESGO DE INASISTENCIA", "Se recomienda contactar al paciente o reprogramar la cita"
        icon = ":material/gpp_bad:"

    st.html(
        f"""
        <div class="result-card {tone}">
          <div class="verdict {tone}">{icon} {title}</div>
          <div class="verdict-sub">{subtitle}</div>
          <div class="prob-label">Probabilidad de no-show (inasistencia)</div>
          <div class="prob-value {tone}">{pct:.2f}%</div>
          <div class="prob-label">Umbrales: menor a {LOW_BAND_PCT}% bajo · {LOW_BAND_PCT}–{HIGH_BAND_PCT}% moderado · mayor a {HIGH_BAND_PCT}% alto</div>
        </div>
        """
    )


def render_features_table() -> None:
    features = [
        ("Age", "Numérica (entera)", "0 – 115", "Edad del paciente"),
        ("WaitTime_Days", "Numérica (entera)", "0 – 179", "Días entre la programación y la cita (AppointmentDay − ScheduledDay)"),
        ("Scholarship", "Binaria (0/1)", "0, 1", "Paciente pertenece al programa de becas (Bolsa Família)"),
        ("Hipertension", "Binaria (0/1)", "0, 1", "Paciente con hipertensión"),
        ("Diabetes", "Binaria (0/1)", "0, 1", "Paciente con diabetes"),
        ("SMS_received", "Binaria (0/1)", "0, 1", "Recibió al menos un recordatorio por SMS"),
    ]
    features_df = pd.DataFrame(features, columns=["Variable", "Tipo", "Valores / Rango", "Descripción"])
    target_row = pd.DataFrame([("No-show (objetivo)", "Binaria", "No (0), Yes (1)", "Inasistencia a la cita; se codifica 1 = no-show para el entrenamiento")], columns=features_df.columns)
    st.dataframe(pd.concat([features_df, target_row], ignore_index=True), hide_index=True)
    st.caption(
        "Calidad de datos: ScheduledDay y AppointmentDay se convierten a datetime y se calcula WaitTime_Days; "
        "se eliminan filas con edad o espera negativas. Preprocesamiento: StandardScaler sobre Age y WaitTime_Days; resto en escala original (passthrough)."
    )


apply_clean_ui()

st.title("Predicción de Inasistencia Médica (No-Show)")
st.caption("Medical Appointment No Shows (110,527 citas) — modelo XGBoost para estimar la probabilidad de que un paciente falte a su cita.")

tab_pred, tab_docs = st.tabs(
    ["Predicción de Inasistencia", "Documentación del Ejercicio"],
    default="Predicción de Inasistencia",
)

with tab_pred:
    st.subheader("Ficha del paciente")
    with st.container(border=True):
        with st.form("noshow_form"):
            col1, col2 = st.columns(2)
            age = col1.number_input("Edad (Age)", min_value=0, max_value=115, value=35, step=1)
            wait_time = col2.number_input("Días de espera (WaitTime_Days)", min_value=0, max_value=180, value=4, step=1,
                                          help="Días transcurridos entre la programación de la cita y la cita en sí.")

            col3, col4 = st.columns(2)
            with col3:
                scholarship = st.checkbox("Paciente con beca (Scholarship)", value=False)
                hipertension = st.checkbox("Hipertensión (Hipertension)", value=False)
            with col4:
                diabetes = st.checkbox("Diabetes (Diabetes)", value=False)
                sms_received = st.checkbox("Recibió recordatorio SMS (SMS_received)", value=True)

            submitted = st.form_submit_button(
                "Evaluar Probabilidad de Inasistencia",
                type="primary",
                icon=":material/event_busy:",
                key="evaluate",
            )

    with st.container(border=True):
        st.subheader("Resultado")
        if submitted:
            row = pd.DataFrame(
                [
                    {
                        "Age": int(age),
                        "WaitTime_Days": int(wait_time),
                        "Scholarship": int(scholarship),
                        "Hipertension": int(hipertension),
                        "Diabetes": int(diabetes),
                        "SMS_received": int(sms_received),
                    }
                ]
            )
            model = load_model()
            prob_noshow = float(model.predict_proba(row)[0, 1])
            render_verdict(prob_noshow)

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Edad", f"{age} años")
            col_m2.metric("Días de espera", f"{wait_time}")
            col_m3.metric("Recordatorio SMS", "Sí" if sms_received else "No")
            st.caption("La probabilidad de clase 1 (No-show) se obtiene con predict_proba. Bandas orientativas sobre una tasa base de no-show ≈ 20%.")
        else:
            st.markdown(
                "<div class='empty-hint'>Completa la ficha del paciente y pulsa \u201CEvaluar Probabilidad de Inasistencia\u201D para obtener la alerta.</div>",
                unsafe_allow_html=True,
            )

with tab_docs:
    st.subheader("Variables predictoras")
    render_features_table()

    st.divider()
    st.subheader("Cómo funciona el modelo")
    st.markdown("### XGBoost con balanceo de clases")
    report = load_confusion_report()

    with st.container(border=True):
        st.markdown("**¿Qué hace en general?**")
        st.write(
            "Es un ensamble de 300 árboles de decisión entrenados en secuencia (gradient boosting). Aprende qué perfil de paciente "
            "(edad, tiempo de espera, condiciones de salud, beca, recordatorios) se asocia con mayor probabilidad de faltar a la cita, "
            "y emite una probabilidad continua de no-show para priorizar recordatorios o intervenciones."
        )
        st.markdown("**¿Cómo funciona técnicamente?**")
        st.write(
            "Pipeline de scikit-learn: StandardScaler sobre Age y WaitTime_Days y el resto pasa tal cual (todas numéricas). "
            "El clasificador XGBClassifier (n_estimators=200, learning_rate=0.05, max_depth=4 y scale_pos_weight≈4.9 para compensar "
            "el desbalance de ≈20% no-show) se envuelve en CalibratedClassifierCV (sigmoid) para que las probabilidades sean fiables: "
            "su promedio coincide con la tasa real observada. El punto de decisión se fija con el criterio de Youden (máximo TPR−FPR, "
            "≈19%) y la probabilidad P(No-show=1 | x) sale de predict_proba."
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ROC-AUC (test)", f"{report['test']['roc_auc']:.4f}")
        col_b.metric("Recall No-Show (test)", f"{report['test']['recall_no_show']:.4f}")
        col_c.metric("Precision No-Show (test)", f"{report['test']['precision_no_show']:.4f}")
        st.markdown("**Casos de uso adicionales**")
        st.write(
            "Programas de recordatorio inteligente (enviar SMS solo a perfiles de alto riesgo), gestión de agenda y overbooking basado en "
            "probabilidad de no-show, priorización de pacientes para llamadas de confirmación y planificación de recursos médicos."
        )
        st.markdown("**Consideraciones**")
        st.write(
            "El scale_pos_weight mejora la detección de no-shows a costa de la accuracy (≈58% en test) frente a un baseline de ≈80% "
            "si se predijera siempre 'asiste': en este dominio interesa más el recall de la clase minoritaria (~84% en test). "
            "Las probabilidades están calibradas (media ≈ tasa real ≈20%) pero la precisión es baja (~30%): muchos pacientes marcados "
            "como riesgo finalmente asisten. El umbral debe fijarse según el costo del recordatorio vs el de la cita perdida; aquí se usó Youden (≈19%) "
            "y la interfaz presenta bandas de riesgo sobre la probabilidad."
        )

    st.divider()
    st.subheader("Matriz de confusión — datos de entrenamiento")
    st.image(str(REPORTS_DIR / "confusion_noshow_model.png"), width="stretch", caption="XGBoost No-Show — datos de entrenamiento")
    st.dataframe(
        pd.DataFrame(
            report["matrix"],
            index=["Real: Asiste (0)", "Real: No-Show (1)"],
            columns=["Pred: Asiste (0)", "Pred: No-Show (1)"],
        )
    )
    st.write(
        f"Aciertos: {report['TN'] + report['TP']} de {report['samples']} (accuracy {report['accuracy']:.4f}) · "
        f"Recall (No-Show) {report['recall_class_1']:.4f} · Precision (No-Show) {report['precision_class_1']:.4f}"
    )
    st.warning(
        "Conteos sobre el split de entrenamiento (80%). La accuracy baja (~59%) refleja el balanceo hacia la clase minoritaria: "
        "para decidir conviene usar ROC-AUC y recall/precision de la clase No-Show (ver pestaña del modelo)."
    )

    st.divider()
    st.subheader("Teoría de la matriz de confusión")
    render_confusion_theory("No-Show (1 = el paciente no asiste a su cita)")

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard(
        "En este ejercicio se aplican los pasos 1–10 (dataset público de citas médicas, limpieza con WaitTime_Days y descarte de edades/esperas "
        "negativas, EDA de tasa de no-show, split estratificado 80/20, XGBoost balanceado, métricas de test y matriz de confusión) y el "
        "despliegue vía dashboard (paso 11)."
    )
