import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.docs import render_confusion_theory, render_standard
from src.labs import render_pipeline_overview, render_threshold_lab
from src_telco.data_loader import load_telco_data
from src_telco.quality import run_quality_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSION_PATH = PROJECT_ROOT / "models" / "confusion_report_telco.json"


def apply_clean_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Plus+Jakarta+Sans:wght@600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #042f2e;
        }

        .stButton>button {
            background-color: #0d9488;
            color: white;
            border-radius: 8px;
            border: none;
            box-shadow: 0px 4px 6px -1px rgba(0, 0, 0, 0.1), 0px 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.2s ease;
        }

        .stButton>button:hover {
            background-color: #0f766e;
            box-shadow: 0px 10px 15px -3px rgba(0, 0, 0, 0.1), 0px 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-1px);
        }

        div[data-testid="stForm"], div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
        }

        div[data-testid="stForm"] p, div[data-testid="stForm"] label,
        div[data-testid="stExpander"] p, div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] li {
            color: #374151 !important;
        }

        div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary span {
            color: #042f2e !important;
            font-weight: 600;
        }

        div[data-testid="stExpander"] svg {
            fill: #042f2e !important;
            color: #042f2e !important;
        }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando el modelo de churn...")
def load_churn_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h")
def load_clean_data():
    df = load_telco_data()
    df = run_quality_pipeline(df)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


@st.cache_data(ttl="1h")
def load_confusion_report() -> dict:
    return json.loads(CONFUSION_PATH.read_text(encoding="utf-8"))


apply_clean_ui()

st.title("Predicción de Churn en Telecomunicaciones")
st.caption("Telco Customer Churn (IBM, 7,043 clientes) — modelo XGBoost con explicabilidad SHAP. Detecta clientes en riesgo de abandono antes de que se vayan.")

df = load_clean_data()

tab_eda, tab_pred, tab_batch, tab_docs, tab_how, tab_lab = st.tabs(
    ["📊 EDA & Calidad", "🔮 Predicción Interactiva", "📁 Predicción Masiva (CSV)", "📚 Documentación de Modelos", "🧠 Cómo funciona", "🔬 Laboratorio"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EDA & Calidad
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Exploratory Data Analysis")

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(20))

    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())

    st.subheader("Distribución de Churn")
    churn_counts = df["Churn"].value_counts().rename({0: "No", 1: "Yes"})
    fig_churn = px.pie(
        churn_counts,
        values=churn_counts.values,
        names=churn_counts.index,
        title="Proporción de Churn",
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    st.plotly_chart(fig_churn)

    st.subheader("Distribución de Tenure por Churn")
    fig_tenure = px.histogram(
        df,
        x="tenure",
        color=df["Churn"].map({0: "No", 1: "Yes"}),
        barmode="overlay",
        opacity=0.65,
        color_discrete_sequence=["#636EFA", "#EF553B"],
        labels={"color": "Churn"},
    )
    st.plotly_chart(fig_tenure)

    st.subheader("Distribución de MonthlyCharges por Churn")
    fig_monthly = px.histogram(
        df,
        x="MonthlyCharges",
        color=df["Churn"].map({0: "No", 1: "Yes"}),
        barmode="overlay",
        opacity=0.65,
        color_discrete_sequence=["#636EFA", "#EF553B"],
        labels={"color": "Churn"},
    )
    st.plotly_chart(fig_monthly)

    st.subheader("Correlación entre variables numéricas")
    numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
    corr = numeric_df.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Matriz de correlación",
    )
    st.plotly_chart(fig_corr)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Predicción Interactiva
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.header("Predicción de Churn para un Cliente")

    st.sidebar.header("Datos del cliente")

    tenure = st.sidebar.slider("Tenure (meses)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Cargo mensual ($)", 18.0, 120.0, 50.0, step=1.0)
    total_charges = tenure * monthly_charges

    contract = st.sidebar.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox(
        "Método de pago",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    gender = st.sidebar.selectbox("Género", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    partner = st.sidebar.selectbox("Pareja", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependientes", ["Yes", "No"])

    if st.sidebar.button("Predecir"):
        input_df = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [senior_citizen],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": ["Yes"],
                "MultipleLines": ["No"],
                "InternetService": [internet_service],
                "OnlineSecurity": ["No"],
                "OnlineBackup": ["No"],
                "DeviceProtection": ["No"],
                "TechSupport": ["No"],
                "StreamingTV": ["No"],
                "StreamingMovies": ["No"],
                "Contract": [contract],
                "PaperlessBilling": ["Yes"],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
            }
        )

        model = load_churn_model()
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ El cliente **ABANDONARÁ** (Churn)")
        else:
            st.success("✅ El cliente **NO abandonará** (No Churn)")

        st.metric(
            label="Probabilidad de Churn",
            value=f"{probability[1] * 100:.1f}%",
        )

        st.subheader("Distribución de probabilidades")
        prob_df = pd.DataFrame(
            {"Resultado": ["No Churn", "Churn"], "Probabilidad": [probability[0], probability[1]]}
        )
        fig_prob = px.bar(
            prob_df,
            x="Resultado",
            y="Probabilidad",
            color="Resultado",
            color_discrete_map={"No Churn": "#636EFA", "Churn": "#EF553B"},
            text_auto=".1%",
        )
        st.plotly_chart(fig_prob)

        st.subheader("Explicabilidad SHAP (Waterfall)")
        with st.spinner("Calculando valores SHAP..."):
            preprocessor = model.named_steps["preprocessor"]
            xgb_model = model.named_steps["classifier"]

            X_transformed = preprocessor.transform(input_df)
            raw_names = preprocessor.get_feature_names_out()
            feature_names = [name.split("__", 1)[-1] for name in raw_names]

            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(X_transformed)
            shap_values.feature_names = list(feature_names)

            fig_shap, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig_shap)
            plt.close(fig_shap)
    else:
        st.info("Ajusta los parámetros en la barra lateral y pulsa **Predecir**.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Predicción Masiva (CSV)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Predicción Masiva por Lotes")

    uploaded_file = st.file_uploader("Sube un archivo CSV con datos de clientes", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.subheader("Vista previa del archivo cargado")
        st.dataframe(df_batch.head(20))

        if st.button("Ejecutar Predicción Masiva"):
            with st.spinner("Procesando predicciones..."):
                df_clean = df_batch.copy()

                if "customerID" in df_clean.columns:
                    df_clean.drop(columns=["customerID"], inplace=True)

                df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
                df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())

                model_batch = load_churn_model()
                predictions = model_batch.predict(df_clean)
                probabilities = model_batch.predict_proba(df_clean)[:, 1]

                df_batch["Churn_Prediction"] = np.where(predictions == 1, "Yes", "No")
                df_batch["Churn_Probability_%"] = (probabilities * 100).round(1)

            st.subheader("Resultados de la predicción")
            st.dataframe(df_batch)

            st.success(f"Predicciones completadas para {len(df_batch)} clientes.")

            csv_output = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar resultados como CSV",
                data=csv_output,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Documentación de Modelos
# ═══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.subheader("Variables predictoras")
    telco_features = [
        ("gender", "Categórica", "Male, Female", "Género del cliente"),
        ("SeniorCitizen", "Numérica (binaria)", "0, 1", "1 si es adulto mayor"),
        ("Partner", "Categórica", "Yes, No", "Tiene pareja"),
        ("Dependents", "Categórica", "Yes, No", "Tiene dependientes"),
        ("tenure", "Numérica (entera)", "0 – 72", "Meses de antigüedad con la empresa"),
        ("PhoneService", "Categórica", "Yes, No", "Tiene servicio telefónico"),
        ("MultipleLines", "Categórica", "No service, No, Yes", "Líneas telefónicas múltiples"),
        ("InternetService", "Categórica", "DSL, Fiber optic, No", "Tipo de servicio de internet"),
        ("OnlineSecurity", "Categórica", "No service, No, Yes", "Seguridad en línea contratada"),
        ("OnlineBackup", "Categórica", "No service, No, Yes", "Respaldo en línea contratado"),
        ("DeviceProtection", "Categórica", "No service, No, Yes", "Protección de dispositivos"),
        ("TechSupport", "Categórica", "No service, No, Yes", "Soporte técnico contratado"),
        ("StreamingTV", "Categórica", "No service, No, Yes", "Streaming de TV"),
        ("StreamingMovies", "Categórica", "No service, No, Yes", "Streaming de películas"),
        ("Contract", "Categórica", "Month-to-month, One year, Two year", "Tipo de contrato"),
        ("PaperlessBilling", "Categórica", "Yes, No", "Facturación sin papel"),
        ("PaymentMethod", "Categórica", "Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)", "Método de pago"),
        ("MonthlyCharges", "Numérica (decimal)", "~18 – 120 (USD)", "Cargo mensual"),
        ("TotalCharges", "Numérica (decimal)", "Acumulado (USD)", "Cargo total acumulado"),
    ]
    features_df = pd.DataFrame(telco_features, columns=["Variable", "Tipo", "Valores / Rango", "Descripción"])
    target_row = pd.DataFrame([("Churn (objetivo)", "Binaria", "No (0), Yes (1)", "Abandono del servicio; se codifica 1 = churn para el entrenamiento")], columns=features_df.columns)
    st.dataframe(pd.concat([features_df, target_row], ignore_index=True), hide_index=True)
    st.caption("Preprocesamiento aplicado en el pipeline: StandardScaler sobre las 3 numéricas y OneHotEncoder sobre las 16 categóricas. Se descarta 'customerID' por no tener valor predictivo.")

    st.divider()
    st.subheader("Cómo funciona el modelo")
    st.markdown("### XGBoost con explicabilidad SHAP")
    report = load_confusion_report()

    with st.container(border=True):
        st.markdown("**¿Qué hace en general?**")
        st.write(
            "Es un ensamble de 200 árboles de decisión entrenados en secuencia: cada árbol nuevo corrige los errores de los anteriores "
            "mediante descenso de gradiente. El modelo aprende qué perfiles de cliente (contrato, antigüedad, cargos, servicios contratados) "
            "se asocian con mayor probabilidad de abandono, y emite una probabilidad continua de churn."
        )
        st.markdown("**¿Cómo funciona técnicamente?**")
        st.write(
            "Pipeline de scikit-learn con dos etapas: (1) preprocesador ColumnTransformer con StandardScaler (numéricas) y "
            "OneHotEncoder (categóricas); (2) XGBClassifier con n_estimators=200, max_depth=5 y learning_rate=0.1. "
            "La probabilidad P(Churn=1 | x) sale de predict_proba. La explicabilidad se obtiene con shap.TreeExplainer "
            "(valores de Shapley): un gráfico waterfall muestra cuánto desplaza cada variable a cada cliente desde la predicción base."
        )
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Accuracy (test)", f"{report['test']['accuracy']:.4f}")
        col_b.metric("ROC-AUC (test)", f"{report['test']['roc_auc']:.4f}")
        col_c.metric("Precision churn (test)", f"{report['test']['precision_churn']:.4f}")
        col_d.metric("Recall churn (test)", f"{report['test']['recall_churn']:.4f}")
        st.markdown("**Casos de uso adicionales**")
        st.write(
            "Programas de retención proactiva (marketing preventivo), segmentación por riesgo de abandono, priorización de campañas, "
            "cálculo de LTV ajustado por riesgo y modelos análogos de churn en banca, seguros o SaaS."
        )
        st.markdown("**Consideraciones**")
        st.write(
            "Requiere validación cruzada y control de hiperparámetros para evitar sobreajuste (en entrenamiento alcanza ~88% de accuracy vs ~79% en test). "
            "Es menos interpretable de forma nativa: se complementa con SHAP. Con desbalance (27% churn) conviene ajustar scale_pos_weight y priorizar "
            "recall/precision de la clase minoritaria. El umbral de decisión debe fijarse según el costo de retener vs el de perder al cliente."
        )

    st.divider()
    st.subheader("Matriz de confusión — datos de entrenamiento")
    st.image(str(REPORTS_DIR / "confusion_churn_model.png"), width="stretch", caption="XGBoost Churn — datos de entrenamiento")
    st.dataframe(
        pd.DataFrame(
            report["matrix"],
            index=["Real: No Churn (0)", "Real: Churn (1)"],
            columns=["Pred: No Churn (0)", "Pred: Churn (1)"],
        )
    )
    st.write(
        f"Aciertos: {report['TN'] + report['TP']} de {report['samples']} (accuracy {report['accuracy']:.4f}) · "
        f"Precision (churn) {report['precision_class_1']:.4f} · Recall (churn) {report['recall_class_1']:.4f}"
    )
    st.warning(
        "Conteos sobre el split de entrenamiento (80%). La brecha entre accuracy de entrenamiento (~88%) y de test (~79%) "
        "evidencia algo de sobreajuste: las métricas que importan para decidir son las de test."
    )

    st.divider()
    st.subheader("Teoría de la matriz de confusión")
    render_confusion_theory("Churn (1 = el cliente abandona el servicio)")

    st.divider()
    st.subheader("Paso a paso recomendado (estándar de industria)")
    render_standard(
        "En este ejercicio se aplican los pasos 1–10 (dataset IBM, limpieza de TotalCharges, EDA interactivo, split estratificado 80/20, "
        "XGBoost con SHAP, métricas de test y matriz de confusión) y el despliegue vía dashboard (paso 11)."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Cómo funciona (Metodología)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_how:
    st.header("🧠 ¿Cómo funciona esta Inteligencia Artificial?")
    st.markdown("Esta herramienta no es magia; es el resultado de un marco de trabajo estructurado. Aquí te explicamos paso a paso cómo transformamos datos crudos en predicciones de negocio:")

    st.markdown("---")

    with st.expander("1. Entender el problema (Enfoque de negocio)", expanded=True):
        st.write('''
        **El objetivo:** Antes de tocar una sola línea de código, definimos qué queríamos resolver.
        En este caso, el reto era detectar qué clientes están a punto de cancelar su servicio (fenómeno conocido como *Churn*). La meta de negocio es identificar este riesgo a tiempo para que el equipo de marketing pueda actuar, ofreciendo alternativas o promociones antes de que el cliente tome la decisión final de irse.
        ''')
        st.info('''
        **Detalle técnico:** El problema se formula como una **clasificación binaria supervisada**. Dado un vector de características *x ∈ ℝⁿ* que representa el perfil de un cliente, el modelo debe estimar *P(Churn=1 | x)*, la probabilidad condicional de abandono.
        El costo asimétrico del Churn justifica un enfoque basado en probabilidades: un falso negativo (cliente churn que no fue detectado) tiene un costo 5-25x mayor que un falso positivo. Por eso el modelo devuelve una **probabilidad continua** que permite segmentar intervenciones por nivel de riesgo.
        Dataset: *Telco Customer Churn* de IBM, con 7,043 registros y 21 variables, incluyendo la variable target binaria `Churn` (Yes/No).
        ''')

    with st.expander("2. Identificar variables predictoras (Feature Engineering)"):
        st.write('''
        **La estrategia:** La Inteligencia Artificial necesita "pistas" para adivinar el futuro.
        Seleccionamos factores clave del comportamiento del cliente: ¿Qué tipo de contrato tienen (mensual o anual)? ¿Cuánto pagan al mes? ¿Cuánto tiempo llevan con la empresa? ¿Tienen soporte técnico contratado? Estas características son las piezas del rompecabezas que alimentan al modelo.
        ''')
        st.info('''
        **Detalle técnico:** Construimos una matriz de diseño *X ∈ ℝ^{7043 × 19}* excluyendo la variable target y `customerID`.

        **Variables numéricas (3):** `tenure` (antigüedad, rango 0-72), `MonthlyCharges` (cargo mensual, rango ~18-120), `TotalCharges` (cargo acumulado, convertida de string a float con coerción de errores).

        **Variables categóricas (16):** Cada una se codifica con **One-Hot Encoding**. Ejemplos: `Contract` tiene 3 niveles → genera 3 columnas binarias; `PaymentMethod` tiene 4 niveles → genera 4 columnas binarias.

        **Transformación final:** Después del preprocesamiento, el número de columnas se expande de 19 a ~30 features numéricas, todas en escala compatible para el algoritmo.
        ''')

    with st.expander("3. Estructurar la base de datos (Ingeniería de datos)"):
        st.write('''
        **La organización:** Recopilamos toda la información histórica dispersa de los clientes y la organizamos en una estructura tabular centralizada.
        Imagínalo como un archivo maestro consolidado donde cada fila es un cliente único y cada columna es una de las "pistas" que identificamos en el paso anterior.
        ''')
        st.info('''
        **Detalle técnico:** El CSV se carga con `pandas.read_csv()` y se almacena en un DataFrame tipado. Cada columna se infiere automáticamente: `object` para strings, `int64` para enteros, `float64` para decimales.

        El dataset es una matriz donde cada **fila** (i) es un cliente único, cada **columna** (j) es una característica observable, y la **variable target** `y ∈ {0, 1}` indica si el cliente abandonó (1) o permanece (0).

        **Desbalance de clases:** El dataset tiene aproximadamente 73% No Churn vs 27% Churn. Este desbalance se maneja con **stratified split** durante la partición train/test para preservar la proporción en ambos conjuntos.
        ''')

    with st.expander("4. Garantizar calidad del dato (Data Cleaning Pipeline)"):
        st.write('''
        **La purificación:** Los datos del mundo real siempre vienen con errores, vacíos o formatos incorrectos.
        En esta fase crítica, rellenamos huecos de información (como cargos mensuales no registrados), eliminamos datos inútiles (como el ID del cliente, que no ayuda a predecir nada) y transformamos textos a un lenguaje numérico que la computadora pueda procesar correctamente.
        ''')
        st.info('''
        **Detalle técnico:** El pipeline de limpieza (`quality.py`) ejecuta 3 operaciones en cascada:

        **Coerción de tipos:** `TotalCharges = pd.to_numeric(TotalCharges, errors='coerce')` — valores no numéricos se convierten a `NaN`. El CSV original tiene 11 registros con espacios vacíos en esta columna.

        **Imputación por mediana:** `TotalCharges.fillna(TotalCharges.median())` — se usa la mediana (robusta a outliers) en lugar de la media, ya que la distribución de cargos totales tiene sesgo positivo.

        **Eliminación de features no informativas:** Se descarta `customerID` porque es un identificador nominal sin poder predictivo. Incluirlo causaría **data leakage** o overfitting espurio.

        **Resultado:** DataFrame limpio con 0 valores nulos y todas las columnas en tipos correctos.
        ''')

    with st.expander("5. Estadística descriptiva (EDA cuantitativo)"):
        st.write('''
        **El análisis histórico:** Antes de predecir el futuro, miramos el pasado.
        En esta etapa cruzamos las variables para visualizar el panorama general y confirmar hipótesis. Por ejemplo, al graficar los datos, confirmamos visualmente que los clientes que pagan mes a mes tienen una tasa de abandono drásticamente mayor que aquellos con contratos anuales.
        ''')
        st.info('''
        **Detalle técnico:**

        **Distribución de la variable target:** Se calcula la tasa de Churn global: *P(Churn=1) ≈ 0.265*. Esto establece el **baseline**: un clasificador trivial que siempre prediga "No Churn" tendría ~73.5% de accuracy.

        **Correlaciones lineales (Pearson):**
        `ρ(Xᵢ, Xⱼ) = Cov(Xᵢ, Xⱼ) / (σᵢ · σⱼ)`
        Hallazgos clave:
        - `tenure` ↔ `TotalCharges`: ρ ≈ 0.83 (positiva fuerte, los cargos crecen con el tiempo)
        - `tenure` ↔ `Churn`: ρ ≈ -0.35 (negativa, mayor antigüedad = menos abandono)
        - `MonthlyCharges` ↔ `Churn`: ρ ≈ 0.19 (positiva débil, cargos altos = más abandono)

        **Distribuciones por clase:** Clientes con tenure < 12 meses tienen tasa de Churn > 40%; contrato "Month-to-month" > 42%; "Fiber optic" > 30%.
        ''')

    with st.expander("6. Predicción con IA (El motor predictivo)"):
        st.write('''
        **El aprendizaje:** Aquí entra en acción el Machine Learning. Usamos un algoritmo avanzado que analizó miles de perfiles de clientes del pasado, aprendiendo de forma autónoma las reglas y patrones ocultos de los que decidieron irse.
        Hoy, cuando evalúas a un cliente nuevo en esta aplicación, la IA compara su perfil con todo lo que aprendió y emite un veredicto matemático: la probabilidad exacta de abandono.
        ''')
        st.info('''
        **Detalle técnico:**

        **Arquitectura del Pipeline** (`sklearn.pipeline.Pipeline` con 2 etapas):

        *Etapa 1 — Preprocesamiento (`ColumnTransformer`):*
        - `StandardScaler` para numéricas: *x' = (x - μ) / σ* — centra cada feature en media 0 y desviación estándar 1
        - `OneHotEncoder(handle_unknown='ignore')` para categóricas: convierte cada nivel en un vector binario

        *Etapa 2 — Modelo (`XGBClassifier`):*
        XGBoost es un **Gradient Boosting** que construye secuencialmente árboles débiles, donde cada nuevo árbol corrige los errores residuales del anterior.

        **Hiperparámetros:** `n_estimators=200` (200 árboles), `max_depth=5` (profundidad máxima), `learning_rate=0.1` (tasa de aprendizaje), `eval_metric='logloss'`.

        **Función de pérdida (Log Loss):**
        `L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`
        Donde *ŷ = P(Churn=1 | x)* es la probabilidad predicha.

        **Partición:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` — 80% entrenamiento (5,634 muestras), 20% prueba (1,409 muestras), stratified para preservar proporción de clases.

        **Serialización:** El pipeline completo se exporta con `joblib.dump()` a `models/churn_model.joblib`, conteniendo scaler, OneHotEncoder y los 200 árboles del XGBoost.

        **Explicabilidad (SHAP):** Se utiliza `shap.TreeExplainer` basado en **valores de Shapley** de la teoría de juegos cooperativos:
        `φᵢ = Σ [|S|!(n-|S|-1)!/n!] · [f(S ∪ {i}) - f(S)]`
        Donde *S* es cualquier subconjunto de features sin *i*, y *f(S)* es la predicción del modelo. El gráfico de cascada (waterfall) muestra cómo cada feature contribuye a desplazar la predicción desde el valor base hasta la predicción final para un cliente específico.
        ''')

with tab_lab:
    st.subheader("Laboratorio de umbral y curva ROC")
    st.caption(
        "Mueve el umbral sobre el split de test (1,409 clientes) y observa el equilibrio entre detectar churn y generar falsas alarmas. "
        "El deslizador no cambia el modelo: cambia la decisión de negocio sobre sus probabilidades."
    )
    render_threshold_lab(
        prefix="churn_lab",
        eval_name="churn",
        model_label="XGBoost Churn",
        neg_label="No Churn (0)",
        pos_label="Churn (1)",
        default_threshold=0.4,
        context="La tasa base de churn es ~27%: con un umbral de 0.5 se detecta poco. Prueba umbrales bajos y compara recall vs precision.",
    )

    st.divider()
    render_pipeline_overview(load_churn_model(), title="Pipeline en vivo del modelo de churn")
