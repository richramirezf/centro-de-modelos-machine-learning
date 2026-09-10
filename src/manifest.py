"""Fuente única de verdad de las páginas y ejercicios del portal.

Lo consumen ``app.py`` (navegación), ``app_pages/inicio.py`` (tarjetas) y
``src/assistant_corpus.py`` (corpus del asistente), evitando que los metadatos
se desincronicen entre sí.
"""

PORTAL_PAGES = [
    {
        "title": "Inicio",
        "page": "app_pages/inicio.py",
        "icon": ":material/home:",
        "default": True,
    },
    {
        "title": "Aprende ML",
        "page": "app_pages/aprende.py",
        "icon": ":material/school:",
        "url_path": "aprende",
    },
]

EXERCISE_GROUPS = ["Clasificación Binaria", "Regresión & NLP"]

EXERCISES = [
    {
        "id": "credito",
        "title": "Scoring de Crédito",
        "page": "app_pages/credit_scoring.py",
        "url_path": "credito",
        "icon": ":material/credit_score:",
        "group": "Clasificación Binaria",
        "tabs": ["Evaluación de Riesgo", "Documentación de Modelos", "Laboratorio interactivo"],
        "model": "Regresión Logística (interpretable) y XGBoost",
        "metrics": "ROC-AUC 0.7615 (LogReg) y 0.7440 (XGBoost) en test; dataset German Credit (1,000 solicitudes).",
        "dataset": "German Credit (1,000 solicitudes)",
        "description": "Riesgo de **default crediticio** (German Credit, 1,000). LogReg + XGBoost, umbral dinámico y API FastAPI.",
        "detail": "Estima la probabilidad de default (good/bad) de una solicitud. Incluye umbral dinámico ('Apetito de Riesgo'), API FastAPI en src/api.py y explicabilidad SHAP para XGBoost.",
    },
    {
        "id": "churn",
        "title": "Churn de Telecomunicaciones",
        "page": "app_pages/telco_churn.py",
        "url_path": "churn",
        "icon": ":material/support_agent:",
        "group": "Clasificación Binaria",
        "tabs": ["EDA & Calidad", "Predicción Interactiva", "Predicción Masiva (CSV)", "Documentación de Modelos", "Cómo funciona", "Laboratorio"],
        "model": "XGBoost con explicabilidad SHAP",
        "metrics": "ROC-AUC 0.8338 y accuracy 0.7942 en test; dataset Telco Customer Churn de IBM (7,043 clientes).",
        "dataset": "Telco Customer Churn (IBM, 7,043 clientes)",
        "description": "Abandono de clientes (IBM Telco, 7,043). XGBoost con **SHAP**, EDA y predicción por lotes.",
        "detail": "Predice el abandono de clientes (Churn). Incluye EDA con Plotly, waterfall SHAP por cliente y predicción masiva subiendo un CSV.",
    },
    {
        "id": "noshow",
        "title": "Ausentismo Médico (No-Show)",
        "page": "app_pages/noshow.py",
        "url_path": "noshow",
        "icon": ":material/event_busy:",
        "group": "Clasificación Binaria",
        "tabs": ["Predicción de Inasistencia", "Documentación del Ejercicio", "Laboratorio interactivo"],
        "model": "XGBoost calibrado (CalibratedClassifierCV)",
        "metrics": "ROC-AUC 0.7227 y recall de No-Show 0.8351 en test; 110,527 citas médicas.",
        "dataset": "Medical Appointment No Shows (110,527 citas)",
        "description": "Inasistencias a citas (110,527). XGBoost **calibrado** con alertas por bandas de riesgo.",
        "detail": "Estima la probabilidad de que un paciente no asista a su cita. Trabaja el desbalance de clases, la calibración de probabilidades y alertas por bandas de riesgo.",
    },
    {
        "id": "demanda",
        "title": "Pronóstico de Demanda",
        "page": "app_pages/demand.py",
        "url_path": "demanda",
        "icon": ":material/storefront:",
        "group": "Regresión & NLP",
        "tabs": ["Pronóstico de Demanda", "Documentación del Ejercicio", "Laboratorio (A vs B)"],
        "model": "XGBRegressor",
        "metrics": "R² 0.9117, RMSE 5.87 y MAE 4.08 en test; dataset simulado de ventas (20,000 registros).",
        "dataset": "Simulado de ventas (20,000 registros)",
        "description": "**Regresión**: volumen de ventas por día, producto, precio y promoción. XGBRegressor.",
        "detail": "Regresión continua: proyecta el volumen de ventas (unidades) según día de la semana, tipo de producto, precio y promoción activa.",
    },
    {
        "id": "vivienda",
        "title": "Valuación Inmobiliaria",
        "page": "app_pages/housing.py",
        "url_path": "vivienda",
        "icon": ":material/home_work:",
        "group": "Regresión & NLP",
        "tabs": ["Valuación de Propiedad", "Documentación del Ejercicio", "Laboratorio (A vs B)"],
        "model": "XGBRegressor con StandardScaler",
        "metrics": "R² 0.9011, RMSE ≈ 25,309 USD y MAE ≈ 20,298 USD en test; dataset simulado (15,000 propiedades).",
        "dataset": "Simulado inmobiliario (15,000 propiedades)",
        "description": "**Regresión**: precio (USD) por superficie, habitaciones, antigüedad y garaje. XGBRegressor.",
        "detail": "Regresión continua: estima el precio de una propiedad (USD) según metros cuadrados, habitaciones, antigüedad y garaje.",
    },
    {
        "id": "intent",
        "title": "Clasificador de Textos (NLP)",
        "page": "app_pages/intent.py",
        "url_path": "nlp",
        "icon": ":material/forum:",
        "group": "Regresión & NLP",
        "tabs": ["Clasificar Mensaje", "Documentación del Ejercicio", "Laboratorio NLP"],
        "model": "TF-IDF + Regresión Logística multinomial",
        "metrics": "Accuracy 1.0000 en test (corpus simulado de 522 mensajes con separación léxica clara).",
        "dataset": "Simulado de chat en español (522 mensajes)",
        "description": "**NLP**: intención de mensajes de chat (Soporte/Ventas/Reclamos/Horarios). TF-IDF + LogReg.",
        "detail": "NLP: clasifica mensajes de chat en cuatro intenciones (Soporte_Tecnico, Ventas, Reclamos, Horarios) y muestra la confianza con predict_proba.",
    },
]
