from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

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


@st.cache_resource(show_spinner="Cargando motores predictivos...")
def load_models() -> dict[str, object]:
    return {label: load_model(filename) for label, filename in MODEL_LABELS.items()}


def load_confusion_report() -> dict:
    return load_report("confusion_report.json")


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
    rows = [
        ("Age", "Numérica (entera)", "18 – 100", "Edad del solicitante"),
        ("Sex", "Categórica", "male, female", "Género del solicitante"),
        ("Housing", "Categórica", "own, rent, free", "Régimen de vivienda"),
        ("Saving accounts", "Categórica", "None, little, moderate, quite rich, rich", "Nivel de cuenta de ahorros ('None' = sin saldo)" ),
        ("Checking account", "Categórica", "None, little, moderate, rich", "Nivel de cuenta corriente ('None' = sin saldo)"),
        ("Credit amount", "Numérica (entera)", "100 – 100000", "Monto del crédito solicitado"),
        ("Duration", "Numérica (entera)", "1 – 72", "Plazo del crédito en meses"),
        ("Purpose", "Categórica", "car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others", "Finalidad del crédito"),
        ("Risk (objetivo)", "Binaria", "good (0), bad (1)", "Etiqueta de default; se codifica como 'Risk_num' para el entrenamiento"),
    ]
    render_features_table_shared(
        rows,
        "Preprocesamiento aplicado en el pipeline: OneHotEncoder sobre las categóricas y StandardScaler sobre las numéricas. La columna 'Job' no se utiliza.",
    )


def render_model_card(label: str, general: str, technical: str, use_cases: str, considerations: str) -> None:
    roc_auc, recall = TEST_METRICS[label].values()
    render_model_doc(
        title=label,
        general=general,
        technical=technical,
        metrics={
            "ROC-AUC (test)": f"{roc_auc:.4f}",
            "Recall clase 1 (test)": f"{recall:.4f}",
            "Clase de riesgo": "bad (1) = default",
        },
        use_cases=use_cases,
        considerations=considerations,
    )


def render_confusion_section() -> None:
    report = load_confusion_report()
    by_model = {m["model_name"]: m for m in report["models"]}
    labels = ["good (0)", "bad (1)"]

    col_lr, col_xgb = st.columns(2)
    with col_lr:
        render_confusion_matrix(
            by_model["logistic_model"],
            labels=labels,
            image_path=REPORTS_DIR / "confusion_logistic_model.png",
            caption="Regresión Logística — datos de entrenamiento",
            positive_name="bad",
        )
    with col_xgb:
        render_confusion_matrix(
            by_model["xgb_model"],
            labels=labels,
            image_path=REPORTS_DIR / "confusion_xgb_model.png",
            caption="XGBoost — datos de entrenamiento",
            positive_name="bad",
        )

    st.warning(
        "Conteos sobre el split de entrenamiento (80%). XGBoost llega a 1.0 en entrenamiento porque los modelos de boosting memorizan datasets pequeños; "
        "el rendimiento que importa es el de test (ROC-AUC / Recall en la pestaña de cada modelo)."
    )


apply_theme("credit")

st.title("Evaluación de Riesgo Crediticio")
st.caption("German Credit Dataset — modelo comparativo de scoring. Evalúa solicitudes con umbral dinámico y explora la documentación técnica.")

tab_eval, tab_docs, tab_lab = st.tabs(
    ["Evaluación de Riesgo", "Documentación de Modelos", "Laboratorio interactivo"],
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

            if "XGBoost" in model_choice:
                show_shap = st.checkbox("Explicar esta decisión con SHAP", key="credit_shap_waterfall")
                if show_shap:
                    with st.spinner("Calculando valores SHAP..."):
                        preprocessor = model.named_steps["preprocessor"]
                        xgb = model.named_steps["classifier"]
                        x_transformed = preprocessor.transform(row)
                        feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
                        explainer = shap.TreeExplainer(xgb)
                        shap_values = explainer(x_transformed)
                        shap_values.feature_names = list(feature_names)
                        fig, _ = plt.subplots(figsize=(9, 5))
                        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
                        st.pyplot(fig)
                        plt.close(fig)
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
    render_confusion_theory("bad (1) = default / morosidad")

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard(
        "En este ejercicio se aplican los pasos 1–9 (dataset, EDA, split estratificado 80/20, comparación LogReg vs XGBoost, "
        "AUC/Recall, matriz de confusión y umbral dinámico) y el paso 11 iniciado con la API FastAPI (`src/api.py`)."
    )

with tab_lab:
    st.subheader("Laboratorio de umbral y curva ROC")
    st.caption(
        "Explora cómo cambia la matriz de confusión y sus métricas al mover el umbral de decisión sobre el split de test "
        "(200 solicitudes, 20% del dataset). Conecta la teoría de la pestaña Documentación con la decisión de negocio."
    )
    lab_model = st.selectbox(
        "Motor a analizar", list(MODEL_LABELS.keys()),
        index=1, key="lab_model_credit",
        help="LogReg y XGBoost tienen distinta curva de calibración de riesgo.",
    )
    lab_eval = "credit_logreg" if "Log" in lab_model else "credit_xgb"
    render_threshold_lab(
        prefix="credit_lab",
        eval_name=lab_eval,
        model_label=lab_model,
        neg_label="good (0)",
        pos_label="bad (1)",
        default_threshold=0.5,
        context="Compara qué tan distinto responde cada motor al umbral; el 'Apetito de Riesgo' de la pestaña Evaluación usa exactamente este mecanismo.",
    )

    st.divider()
    pipeline_model = load_models()[lab_model]
    render_pipeline_overview(pipeline_model, title="Pipeline en vivo del motor seleccionado")
