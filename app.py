from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

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


st.set_page_config(
    page_title="Scoring de Crédito",
    page_icon=":material/credit_score:",
    layout="wide",
)

apply_clean_ui()
models = load_models()

st.title("Evaluación de Riesgo Crediticio")
st.caption("German Credit Dataset — modelo comparativo de scoring. Selecciona el motor, completa los datos y evalúa la solicitud.")

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
        help=f"Porcentaje máximo de probabilidad de default para aprobar: si la probabilidad lo supera, la solicitud se deniega.",
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
        model = models[model_choice]
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