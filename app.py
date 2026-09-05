import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_LABELS = {
    "Regresión Logística (Interpretable)": "logistic_model.joblib",
    "XGBoost (Avanzado)": "xgb_model.joblib",
}

DEFAULT_RISK_THRESHOLD_PCT = 50
MIN_RISK_THRESHOLD_PCT = 10
MAX_RISK_THRESHOLD_PCT = 90

CHECKING_OPTIONS = ["None", "little", "moderate", "rich"]
SAVING_OPTIONS = ["None", "little", "moderate", "quite rich", "rich"]
HOUSING_OPTIONS = ["own", "rent", "free"]
PURPOSE_OPTIONS = [
    "car",
    "furniture/equipment",
    "radio/TV",
    "domestic appliances",
    "repairs",
    "education",
    "business",
    "vacation/others",
]
SEX_OPTIONS = ["male", "female"]

TEST_METRICS = {
    "Regresión Logística (Interpretable)": {"ROC-AUC": 0.7615, "Recall (clase 1)": 0.3833},
    "XGBoost (Avanzado)": {"ROC-AUC": 0.7440, "Recall (clase 1)": 0.5333},
}


def apply_clean_ui() -> None:
    """Teal (M3) corporate look: white elevated cards, dark-gray/teal text."""
    st.html(
        """
        <style>
          :root {
            --teal-900: #004d40;
            --teal-700: #00796b;
            --teal-500: #00897b;
            --teal-100: #b2dfdb;
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

          /* White M3-elevated cards */
          [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid rgba(0, 121, 107, .18) !important;
            border-radius: 16px;
            box-shadow: var(--shadow-2);
            padding: .35rem .6rem;
          }

          /* Prominent engine selector card */
          [data-testid="stVerticalBlockBorderWrapper"].st-key-engine {
            border-left: 5px solid var(--teal-500) !important;
            box-shadow: var(--shadow-1);
          }

          .st-key-engine [data-testid="stWidgetLabel"] p {
            color: var(--teal-700);
            font-size: 1.05rem;
            font-weight: 600;
          }

          /* Result card */
          .result-card {
            background: var(--surface);
            border-radius: 16px;
            box-shadow: var(--shadow-2);
            padding: 1.5rem 2rem;
            text-align: center;
          }
          .result-card.approved { border-top: 5px solid #1b7f3b; }
          .result-card.denied   { border-top: 5px solid #c3352b; }

          .verdict {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: .3px;
          }
          .verdict.approved { color: #1b7f3b; }
          .verdict.denied   { color: #c3352b; }

          .prob-value {
            font-size: 3.2rem;
            font-weight: 700;
            line-height: 1.1;
            margin: .25rem 0;
          }
          .prob-value.approved { color: #1b7f3b; }
          .prob-value.denied   { color: #c3352b; }

          .prob-label {
            color: var(--text-muted);
            font-size: .9rem;
            letter-spacing: .4px;
            text-transform: uppercase;
          }

          .empty-hint { color: var(--text-muted); }

          /* Primary button M3 feel */
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


@st.cache_resource(show_spinner="Cargando motores predictivos...")
def load_models() -> dict[str, object]:
    return {
        label: joblib.load(MODELS_DIR / filename)
        for label, filename in MODEL_LABELS.items()
    }


@st.cache_data(ttl="1h")
def load_confusion_report() -> dict:
    return json.loads((MODELS_DIR / "confusion_report.json").read_text(encoding="utf-8"))


def render_result(prob: float, threshold_pct: float) -> None:
    denied = prob > threshold_pct / 100
    tone = "denied" if denied else "approved"
    verdict = "CRÉDITO DENEGADO" if denied else "CRÉDITO APROBADO"
    icon = ":material/gpp_bad:" if denied else ":material/verified:"

    st.html(
        f"""
        <div class="result-card {tone}">
          <div class="verdict {tone}">{icon} {verdict}</div>
          <div class="prob-label">Probabilidad de default (clase 1)</div>
          <div class="prob-value {tone}">{prob * 100:.2f}%</div>
          <div class="prob-label">Umbral de decisión: {threshold_pct:.0f}%</div>
        </div>
        """
    )


def render_features_table() -> None:
    features = [
        ("Age", "Numérica (entera)", "18 – 100", "Edad del solicitante"),
        ("Sex", "Categórica", "male, female", "Género del solicitante"),
        ("Housing", "Categórica", "own, rent, free", "Régimen de vivienda"),
        ("Saving accounts", "Categórica", "None, little, moderate, quite rich, rich", "Nivel de cuenta de ahorros ('None' = sin saldo)" ),
        ("Checking account", "Categórica", "None, little, moderate, rich", "Nivel de cuenta corriente ('None' = sin saldo)"),
        ("Credit amount", "Numérica (entera)", "100 – 100000", "Monto del crédito solicitado"),
        ("Duration", "Numérica (entera)", "1 – 72", "Plazo del crédito en meses"),
        ("Purpose", "Categórica", "car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others", "Finalidad del crédito"),
    ]
    features_df = pd.DataFrame(features, columns=["Variable", "Tipo", "Valores / Rango", "Descripción"])
    risk_row = pd.DataFrame([("Risk (objetivo)", "Binaria", "good (0), bad (1)", "Etiqueta de default; se codifica como 'Risk_num' para el entrenamiento")], columns=features_df.columns)
    st.dataframe(pd.concat([features_df, risk_row], ignore_index=True), hide_index=True)
    st.caption("Preprocesamiento aplicado en el pipeline: OneHotEncoder sobre las categóricas y StandardScaler sobre las numéricas. La columna 'Job' no se utiliza.")


def render_model_card(label: str, general: str, technical: str, use_cases: str, considerations: str) -> None:
    st.markdown(f"### {label}")
    m1, m2 = TEST_METRICS[label].values()

    with st.container(border=True):
        st.markdown("**¿Qué hace en general?**")
        st.write(general)
        st.markdown("**¿Cómo funciona técnicamente?**")
        st.write(technical)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ROC-AUC (test)", f"{m1:.4f}")
        col_b.metric("Recall clase 1 (test)", f"{m2:.4f}")
        col_c.metric("Clase de riesgo", "bad (1) = default")
        st.markdown("**Casos de uso adicionales**")
        st.write(use_cases)
        st.markdown("**Consideraciones**")
        st.write(considerations)


def render_confusion_section() -> None:
    report = load_confusion_report()
    by_model = {m["model_name"]: m for m in report["models"]}

    col_lr, col_xgb = st.columns(2)
    with col_lr:
        lr = by_model["logistic_model"]
        st.image(str(REPORTS_DIR / "confusion_logistic_model.png"), width="stretch", caption="Regresión Logística — datos de entrenamiento")
        st.dataframe(
            pd.DataFrame(lr["matrix"], index=["Real: good (0)", "Real: bad (1)"], columns=["Pred: good (0)", "Pred: bad (1)"])
        )
        st.write(f"Aciertos: {lr['TN'] + lr['TP']} de {lr['samples']} (accuracy {lr['accuracy']:.4f}) · Precision (bad) {lr['precision_class_1']:.4f} · Recall (bad) {lr['recall_class_1']:.4f}")
    with col_xgb:
        xgb = by_model["xgb_model"]
        st.image(str(REPORTS_DIR / "confusion_xgb_model.png"), width="stretch", caption="XGBoost — datos de entrenamiento")
        st.dataframe(
            pd.DataFrame(xgb["matrix"], index=["Real: good (0)", "Real: bad (1)"], columns=["Pred: good (0)", "Pred: bad (1)"])
        )
        st.write(f"Aciertos: {xgb['TN'] + xgb['TP']} de {xgb['samples']} (accuracy {xgb['accuracy']:.4f}) · Precision (bad) {xgb['precision_class_1']:.4f} · Recall (bad) {xgb['recall_class_1']:.4f}")

    st.warning(
        "Conteos sobre el split de entrenamiento (80%). XGBoost llega a 1.0 en entrenamiento porque los modelos de boosting memorizan datasets pequeños; "
        "el rendimiento que importa es el de test (ROC-AUC / Recall en la pestaña de cada modelo)."
    )


def render_confusion_theory() -> None:
    st.markdown(
        """
        La matriz de confusión resume **aciertos y errores** de un clasificador binario. Aquí la clase **positiva es `bad (1)`** (default / moroso), porque es el evento de riesgo que queremos detectar:

        - **TN — Verdadero Negativo:** buen pagador aprobado correctamente. *Negocio sano y cliente satisfecho.*
        - **FP — Falso Positivo:** buen pagador denegado por error (falso alarmado). *Costo de oportunidad: perdemos a un cliente bueno.*
        - **FN — Falso Negativo:** moroso aprobado por error (el error más caro). *Pérdida financiera directa por impago.*
        - **TP — Verdadero Positivo:** moroso detectado y denegado. *Riesgo evitado.*

        **Métricas derivadas:**
        - **Accuracy** = (TN + TP) / Total — aciertos globales; engaña con clases desbalanceadas.
        - **Precision (clase 1)** = TP / (TP + FP) — de lo que denegué, cuánto era realmente riesgo.
        - **Recall (clase 1)** = TP / (TP + FN) — del riesgo real, cuánto logré detectar.
        - **Specificity** = TN / (TN + FP) — de los buenos, cuántos fueron aprobados.
        - **F1** = media armónica de precision y recall.
        - **FPR (False Positive Rate)** = FP / (FP + TN) — buenos pagadores rechazados.

        **¿Por qué importa el umbral?** El clasificador nunca "decide" solo: emite una probabilidad y nosotros fijamos el corte. Bajar el umbral deniega más solicitudes (suben TP y FP): detectamos más morosos pero sacrificamos clientes buenos. Subirlo aprueba más (suben TN y FN): ganamos volumen pero asumimos más impagos. El deslizador **"Apetito de Riesgo"** de este dashboard materializa exactamente esa decisión de negocio.
        """
    )


def render_standard() -> None:
    steps = [
        ("1. Entendimiento del negocio", "Objetivo medible (p. ej. minimizar pérdida esperada), definición de default, unidad de análisis y aceptación regulatoria."),
        ("2. Recolección y entendimiento de datos", "Fuentes, población objetivo, ventana temporal, volumen y representatividad de las muestras históricas."),
        ("3. Calidad y limpieza", "Nulos, valores atípicos, inconsistencias e imputación con criterio documentado."),
        ("4. Análisis exploratorio (EDA)", "Balance del target, correlaciones, perfil de riesgo por atributo e hipótesis de negocio."),
        ("5. Ingeniería de atributos", "Codificación de categóricas, escalado de numéricas y creación de variables de negocio cuando aplique."),
        ("6. Diseño experimental", "Split estratificado y ventanas out-of-time; validación cruzada (StratifiedKFold); baseline simple de referencia."),
        ("7. Entrenamiento", "Comparar familias distintas (lineal vs ensamble) con pipelines idénticos, misma codificación e hiperparámetros controlados."),
        ("8. Evaluación", "Métricas discriminativas (ROC-AUC), de calibración y de negocio; matriz de confusión y curvas ROC/PR. Nunca decidir solo con accuracy."),
        ("9. Calibración del umbral", "Fijar el corte de probabilidad según apetito de riesgo y costos FN vs FP."),
        ("10. Validación y riesgo", "Out-of-time, estabilidad de variables, sesgo y deriva (drift)."),
        ("11. Despliegue y monitoreo", "Exposición vía API, versionado de modelo y datos, monitoreo de rendimiento y reentrenamiento programado."),
        ("12. Gobernanza y documentación", "Interpretabilidad (coeficientes / SHAP), trazabilidad de decisiones y auditoría regulatoria."),
    ]
    st.markdown(
        "El estándar de referencia para proyectos de analítica/ML es **CRISP-DM**, y la práctica de industria lo extiende con MLOps. "
        "Secuencia recomendada:"
    )
    for title, desc in steps:
        st.markdown(f"- **{title}:** {desc}")

    st.info(
        "En este proyecto aplicamos los pasos 1–9 (dataset, EDA, split estratificado 80/20, comparación LogReg vs XGBoost, "
        "AUC/Recall, matriz de confusión y umbral dinámico) y el paso 11 iniciado con la API FastAPI (`src/api.py`)."
    )


st.set_page_config(
    page_title="Scoring de Crédito",
    page_icon=":material/credit_score:",
    layout="wide",
)

apply_clean_ui()

st.title("Evaluación de Riesgo Crediticio")
st.caption("German Credit Dataset — modelo comparativo de scoring. Evalúa solicitudes con umbral dinámico y explora la documentación técnica.")

tab_eval, tab_docs = st.tabs(
    ["Evaluación de Riesgo", "Documentación de Modelos"],
    default="Evaluación de Riesgo",
)

with tab_eval:
    with st.container(border=True, key="engine"):
        model_choice = st.selectbox(
            "Seleccionar Motor Predictivo",
            options=list(MODEL_LABELS.keys()),
            help="Motor de inferencia usado para calcular la probabilidad de default.",
        )
        risk_appetite_pct = st.slider(
            "Apetito de Riesgo (Límite de Probabilidad)",
            min_value=MIN_RISK_THRESHOLD_PCT,
            max_value=MAX_RISK_THRESHOLD_PCT,
            value=DEFAULT_RISK_THRESHOLD_PCT,
            step=1,
            help="Porcentaje máximo de probabilidad de default para aprobar: si la probabilidad lo supera, la solicitud se deniega.",
        )

    with st.container(border=True):
        st.subheader("Panel del solicitante")
        with st.form("application_form"):
            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age (Edad)", min_value=18, max_value=100, value=35, step=1)
            credit_amount = col2.number_input(
                "Credit amount (Monto solicitado)", min_value=100, max_value=100000, value=3000, step=100, format="%d"
            )
            duration = col3.number_input("Duration (Plazo en meses)", min_value=1, max_value=72, value=24, step=1)

            col4, col5 = st.columns(2)
            sex = col4.selectbox("Sex (Género)", options=SEX_OPTIONS)
            housing = col5.selectbox(
                "Housing (Vivienda)", options=HOUSING_OPTIONS,
                format_func=lambda v: {"own": "Own (Propia)", "rent": "Rent (Alquiler)", "free": "Free (Gratuita)"}[v],
            )

            col6, col7 = st.columns(2)
            saving_accounts = col6.selectbox(
                "Saving accounts (Cuenta de ahorro)", options=SAVING_OPTIONS,
                help="'None' indica que el solicitante no tiene cuenta de ahorro.",
            )
            checking_account = col7.selectbox(
                "Checking account (Cuenta corriente)", options=CHECKING_OPTIONS,
                help="'None' indica que el solicitante no tiene cuenta corriente.",
            )

            purpose = st.selectbox(
                "Purpose (Propósito del crédito)", options=PURPOSE_OPTIONS,
                format_func=str.title,
            )

            submitted = st.form_submit_button(
                "Evaluar Riesgo de Solicitud",
                type="primary",
                icon=":material/play_arrow:",
                key="evaluate",
            )

    with st.container(border=True):
        st.subheader("Resultado de evaluación")
        if submitted:
            row = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Sex": sex,
                        "Housing": housing,
                        "Saving accounts": saving_accounts,
                        "Checking account": checking_account,
                        "Credit amount": credit_amount,
                        "Duration": duration,
                        "Purpose": purpose,
                    }
                ]
            )
            model = load_models()[model_choice]
            prob_default = float(model.predict_proba(row)[0, 1])
            render_result(prob_default, risk_appetite_pct)

            st.metric(
                "Motor", model_choice,
                help="Modelo utilizado para esta evaluación.",
            )
        else:
            st.markdown(
                "<div class='empty-hint'>Completa el formulario y pulsa \u201CEvaluar Riesgo de Solicitud\u201D para obtener el dictamen.</div>",
                unsafe_allow_html=True,
            )

with tab_docs:
    st.subheader("Variables predictoras")
    render_features_table()

    st.divider()
    st.subheader("Cómo funciona cada modelo")

    col_lr, col_xgb = st.columns(2)
    with col_lr:
        render_model_card(
            "Regresión Logística (Interpretable)",
            "Es un modelo estadístico clásico de clasificación binaria. Calcula la probabilidad de default mediante una combinación lineal de las variables transformada con la función sigmoide, que acota el resultado entre 0 y 1. Es una fórmula matemática cerrada: cada variable aporta un peso (coeficiente) legible e interpretable.",
            "P(y=1 | x) = sigmoide(w·x + b). Los pesos w se estiman por máxima verosimilitud; por ser lineal, requiere variables numéricas, por eso el pipeline aplica OneHotEncoder a las categóricas y StandardScaler a las numéricas. La decisión final se obtiene comparando la probabilidad con el umbral (apetito de riesgo).",
            "Suele ser el estándar en scorecards regulatorios y PD (probability of default) retail, como modelo baseline, o cuando el negocio exige explicar cada decisión ante auditoría. Funciona bien cuando hay pocas variables y relaciones aproximadamente lineales.",
            "Asume linealidad e independencia entre predictores; es sensible a multicolinealidad y outliers; pierde rendimiento ante no linealidades fuertes; su interpretabilidad depende de una correcta codificación de las variables.",
        )
    with col_xgb:
        render_model_card(
            "XGBoost (Avanzado)",
            "Es un ensamble de cientos de árboles de decisión construidos en secuencia: cada árbol nuevo se concentra en corregir los errores de los anteriores usando el descenso de gradiente. Entrega predicciones tabulares de muy alta precisión sin requerir relaciones lineales.",
            "Implementa Gradient Boosting regularizado (RL1/L2) con shrinkage (learning_rate), control de profundidad y poda de árboles; combina 100 árboles por defecto (n_estimators). Captura interacciones no lineales y, por defecto, se guarda la probabilidad de clase 1 vía predict_proba.",
            "Adecuado para fraude, churn, scoring de alto volumen, estimación de LGD/EAD, ranking y datasets tabulares de alta dimensión; también cuando se dispone de herramientas de explicabilidad (SHAP) para respaldar las decisiones.",
            "Riesgo de sobreajuste: en este dataset memoriza el entrenamiento (accuracy 1.0) y por eso se decide con métricas de test. Exige validación cruzada y early stopping; es menos interpretable de forma nativa (usar SHAP); con clases desbalanceadas hay que ajustar scale_pos_weight y priorizar recall/precision de la clase minoritaria.",
        )

    st.divider()
    st.subheader("Matriz de confusión — datos de entrenamiento")
    render_confusion_section()

    st.divider()
    st.subheader("Teoría de la matriz de confusión")
    render_confusion_theory()

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard()