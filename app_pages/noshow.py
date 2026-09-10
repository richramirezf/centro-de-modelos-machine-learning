from pathlib import Path

import pandas as pd
import streamlit as st
from src.academy_ui import render_concept

from src.docs import (
    render_confusion_matrix,
    render_confusion_theory,
    render_features_table as render_features_table_shared,
    render_model_doc,
    render_standard,
)
from src.labs import render_pipeline_overview, render_threshold_lab
from src.loaders import load_model, load_report
from src.theme import apply_theme

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURES = ["Age", "WaitTime_Days", "Scholarship", "Hipertension", "Diabetes", "SMS_received"]

LOW_BAND_PCT = 25
HIGH_BAND_PCT = 40


def load_noshow_model():
    return load_model("noshow_model.joblib")


def load_noshow_confusion_report() -> dict:
    return load_report("confusion_report_noshow.json")


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
    rows = [
        ("Age", "Numérica (entera)", "0 – 115", "Edad del paciente"),
        ("WaitTime_Days", "Numérica (entera)", "0 – 179", "Días entre la programación y la cita (AppointmentDay − ScheduledDay)"),
        ("Scholarship", "Binaria (0/1)", "0, 1", "Paciente pertenece al programa de becas (Bolsa Família)"),
        ("Hipertension", "Binaria (0/1)", "0, 1", "Paciente con hipertensión"),
        ("Diabetes", "Binaria (0/1)", "0, 1", "Paciente con diabetes"),
        ("SMS_received", "Binaria (0/1)", "0, 1", "Recibió al menos un recordatorio por SMS"),
        ("No-show (objetivo)", "Binaria", "No (0), Yes (1)", "Inasistencia a la cita; se codifica 1 = no-show para el entrenamiento"),
    ]
    render_features_table_shared(
        rows,
        "Calidad de datos: ScheduledDay y AppointmentDay se convierten a datetime y se calcula WaitTime_Days; "
        "se eliminan filas con edad o espera negativas. Preprocesamiento: StandardScaler sobre Age y WaitTime_Days; resto en escala original (passthrough).",
    )


def render_sensitivity_probe() -> None:
    st.markdown("#### ¿Cómo cambia el riesgo con cada variable? (sonda)")
    st.caption(
        "Fijamos un perfil base (edad 35, espera 4 días, sin beca ni condiciones, con SMS) y movemos una variable a la vez "
        "para observar el efecto sobre la probabilidad de no-show."
    )
    model = load_noshow_model()
    base = {"Age": 35, "WaitTime_Days": 4, "Scholarship": 0, "Hipertension": 0, "Diabetes": 0, "SMS_received": 1}

    def prob_of(**overrides) -> float:
        row = {**base, **overrides}
        frame = pd.DataFrame([{k: int(row[k]) for k in FEATURES}])[FEATURES]
        return float(model.predict_proba(frame)[0, 1])

    ages = list(range(0, 101, 2))
    age_probs = [prob_of(Age=a) for a in ages]
    st.markdown("**Efecto de la edad** (resto fijo):")
    st.line_chart(pd.DataFrame({"Probabilidad de no-show": age_probs}, index=pd.Index(ages, name="Edad")))

    waits = list(range(0, 61, 2))
    wait_probs = [prob_of(WaitTime_Days=w) for w in waits]
    st.markdown("**Efecto de los días de espera** (resto fijo):")
    st.line_chart(pd.DataFrame({"Probabilidad de no-show": wait_probs}, index=pd.Index(waits, name="Días de espera")))

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Sin SMS recordatorio", f"{prob_of(SMS_received=0) * 100:.1f}%")
    col_s2.metric("Con SMS recordatorio", f"{prob_of(SMS_received=1) * 100:.1f}%")
    st.markdown(
        "Observa cómo el recordatorio SMS reduce la probabilidad estimada: estás viendo el efecto que el modelo aprendió de los datos."
    )
    render_concept("calibration")


apply_theme("noshow")

st.title("Predicción de Inasistencia Médica (No-Show)")
st.caption("Medical Appointment No Shows (110,527 citas) — modelo XGBoost para estimar la probabilidad de que un paciente falte a su cita.")

tab_pred, tab_docs, tab_lab = st.tabs(
    ["Predicción de Inasistencia", "Documentación del Ejercicio", "Laboratorio interactivo"],
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
            model = load_noshow_model()
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
    report = load_noshow_confusion_report()

    render_model_doc(
        title="XGBoost con balanceo de clases",
        general=(
            "Es un ensamble de 300 árboles de decisión entrenados en secuencia (gradient boosting). Aprende qué perfil de paciente "
            "(edad, tiempo de espera, condiciones de salud, beca, recordatorios) se asocia con mayor probabilidad de faltar a la cita, "
            "y emite una probabilidad continua de no-show para priorizar recordatorios o intervenciones."
        ),
        technical=(
            "Pipeline de scikit-learn: StandardScaler sobre Age y WaitTime_Days y el resto pasa tal cual (todas numéricas). "
            "El clasificador XGBClassifier (n_estimators=200, learning_rate=0.05, max_depth=4 y scale_pos_weight≈4.9 para compensar "
            "el desbalance de ≈20% no-show) se envuelve en CalibratedClassifierCV (sigmoid) para que las probabilidades sean fiables: "
            "su promedio coincide con la tasa real observada. El punto de decisión se fija con el criterio de Youden (máximo TPR−FPR, "
            "≈19%) y la probabilidad P(No-show=1 | x) sale de predict_proba."
        ),
        metrics={
            "ROC-AUC (test)": f"{report['test']['roc_auc']:.4f}",
            "Recall No-Show (test)": f"{report['test']['recall_no_show']:.4f}",
            "Precision No-Show (test)": f"{report['test']['precision_no_show']:.4f}",
        },
        use_cases=(
            "Programas de recordatorio inteligente (enviar SMS solo a perfiles de alto riesgo), gestión de agenda y overbooking basado en "
            "probabilidad de no-show, priorización de pacientes para llamadas de confirmación y planificación de recursos médicos."
        ),
        considerations=(
            "El scale_pos_weight mejora la detección de no-shows a costa de la accuracy (≈58% en test) frente a un baseline de ≈80% "
            "si se predijera siempre 'asiste': en este dominio interesa más el recall de la clase minoritaria (~84% en test). "
            "Las probabilidades están calibradas (media ≈ tasa real ≈20%) pero la precisión es baja (~30%): muchos pacientes marcados "
            "como riesgo finalmente asisten. El umbral debe fijarse según el costo del recordatorio vs el de la cita perdida; aquí se usó Youden (≈19%) "
            "y la interfaz presenta bandas de riesgo sobre la probabilidad."
        ),
    )

    st.divider()
    st.subheader("Matriz de confusión — datos de entrenamiento")
    render_confusion_matrix(
        report,
        labels=["Asiste (0)", "No-Show (1)"],
        image_path=REPORTS_DIR / "confusion_noshow_model.png",
        caption="XGBoost No-Show — datos de entrenamiento",
        positive_name="No-Show",
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

with tab_lab:
    st.subheader("Laboratorio de umbral y sensibilidad")
    st.caption(
        "Mueve el umbral sobre el split de test (22,105 citas) para ver el equilibrio entre detectar no-shows y generar falsas alarmas. "
        "Las probabilidades están calibradas: el punto de Youden (~19%) y las bandas de la pestaña Predicción se basan en esta misma curva."
    )
    render_threshold_lab(
        prefix="noshow_lab",
        eval_name="noshow",
        model_label="XGBoost calibrado (No-Show)",
        neg_label="Asiste (0)",
        pos_label="No-Show (1)",
        default_threshold=0.19,
    )

    st.divider()
    render_sensitivity_probe()

    st.divider()
    render_pipeline_overview(load_noshow_model(), title="Pipeline en vivo del modelo No-Show")
