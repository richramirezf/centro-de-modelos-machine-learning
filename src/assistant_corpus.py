"""Corpus del asistente RAG: convierte el contenido educativo y la
documentación de cada ejercicio en fragmentos con metadatos de ubicación
(página, pestaña, sección) para poder citar dónde encontrar cada cosa."""

from src.academy_content import (
    CHOOSE_MODEL_INTRO,
    CHOOSE_MODEL_TABLE,
    CONCEPTS,
    FAQ,
    GLOSSARY,
    ROUTE,
)

EXERCISES = [
    {
        "id": "credito",
        "title": "Scoring de Crédito",
        "page": "app_pages/credit_scoring.py",
        "tabs": ["Evaluación de Riesgo", "Documentación de Modelos", "Laboratorio interactivo"],
        "model": "Regresión Logística (interpretable) y XGBoost",
        "metrics": "ROC-AUC 0.7615 (LogReg) y 0.7440 (XGBoost) en test; dataset German Credit (1,000 solicitudes).",
        "desc": "Estima la probabilidad de default (good/bad) de una solicitud. Incluye umbral dinámico ('Apetito de Riesgo'), API FastAPI en src/api.py y explicabilidad SHAP para XGBoost.",
    },
    {
        "id": "churn",
        "title": "Churn de Telecomunicaciones",
        "page": "app_pages/telco_churn.py",
        "tabs": ["EDA & Calidad", "Predicción Interactiva", "Predicción Masiva (CSV)", "Documentación de Modelos", "Cómo funciona", "Laboratorio"],
        "model": "XGBoost con explicabilidad SHAP",
        "metrics": "ROC-AUC 0.8338 y accuracy 0.7942 en test; dataset Telco Customer Churn de IBM (7,043 clientes).",
        "desc": "Predice el abandono de clientes (Churn). Incluye EDA con Plotly, waterfall SHAP por cliente y predicción masiva subiendo un CSV.",
    },
    {
        "id": "noshow",
        "title": "Ausentismo Médico (No-Show)",
        "page": "app_pages/noshow.py",
        "tabs": ["Predicción de Inasistencia", "Documentación del Ejercicio", "Laboratorio interactivo"],
        "model": "XGBoost calibrado (CalibratedClassifierCV)",
        "metrics": "ROC-AUC 0.7227 y recall de No-Show 0.8351 en test; 110,527 citas médicas.",
        "desc": "Estima la probabilidad de que un paciente no asista a su cita. Trabaja el desbalance de clases, la calibración de probabilidades y alertas por bandas de riesgo.",
    },
    {
        "id": "demanda",
        "title": "Pronóstico de Demanda",
        "page": "app_pages/demand.py",
        "tabs": ["Pronóstico de Demanda", "Documentación del Ejercicio", "Laboratorio (A vs B)"],
        "model": "XGBRegressor",
        "metrics": "R² 0.9117, RMSE 5.87 y MAE 4.08 en test; dataset simulado de ventas (20,000 registros).",
        "desc": "Regresión continua: proyecta el volumen de ventas (unidades) según día de la semana, tipo de producto, precio y promoción activa.",
    },
    {
        "id": "vivienda",
        "title": "Valuación Inmobiliaria",
        "page": "app_pages/housing.py",
        "tabs": ["Valuación de Propiedad", "Documentación del Ejercicio", "Laboratorio (A vs B)"],
        "model": "XGBRegressor con StandardScaler",
        "metrics": "R² 0.9011, RMSE ≈ 25,309 USD y MAE ≈ 20,298 USD en test; dataset simulado (15,000 propiedades).",
        "desc": "Regresión continua: estima el precio de una propiedad (USD) según metros cuadrados, habitaciones, antigüedad y garaje.",
    },
    {
        "id": "intent",
        "title": "Clasificador de Textos (NLP)",
        "page": "app_pages/intent.py",
        "tabs": ["Clasificar Mensaje", "Documentación del Ejercicio", "Laboratorio NLP"],
        "model": "TF-IDF + Regresión Logística multinomial",
        "metrics": "Accuracy 1.0000 en test (corpus simulado de 522 mensajes con separación léxica clara).",
        "desc": "NLP: clasifica mensajes de chat en cuatro intenciones (Soporte_Tecnico, Ventas, Reclamos, Horarios) y muestra la confianza con predict_proba.",
    },
]

SITE_INFO = [
    {
        "id": "site_overview",
        "text": (
            "El sitio es el 'Centro de Modelos de Machine Learning': un portal en Streamlit con seis ejercicios "
            "(clasificación, regresión y NLP) más la página educativa 'Aprende ML'. La navegación está en la barra lateral, "
            "agrupada en Portal (Inicio, Aprende ML), Clasificación Binaria (Scoring de Crédito, Churn Telco, Ausentismo Médico) "
            "y Regresión & NLP (Pronóstico de Demanda, Valuación Inmobiliaria, Clasificador de Textos)."
        ),
        "page": "Inicio",
        "tab": "Hub",
        "section": "Descripción general del sitio",
    },
    {
        "id": "site_aprende",
        "text": (
            "La página 'Aprende ML' (barra lateral → Portal → Aprende ML) reúne la ruta de aprendizaje, un glosario con buscador, "
            "la guía '¿Qué modelo elegir?', quizzes con corrección en vivo (uno por ejercicio y uno general) y un FAQ técnico."
        ),
        "page": "Aprende ML",
        "tab": "Ruta de aprendizaje / Glosario / Quizzes / FAQ",
        "section": "Ubicación de los recursos educativos",
    },
    {
        "id": "site_labs",
        "text": (
            "Cada ejercicio tiene una pestaña 'Laboratorio' (o 'Laboratorio interactivo') donde puedes mover el umbral de decisión "
            "y ver la matriz de confusión y la curva ROC en vivo, además de una vista del pipeline. En No-Show hay una sonda de "
            "sensibilidad; en Demanda y Vivienda un comparador A vs B; y en NLP los términos con mayor peso por intención."
        ),
        "page": "Cada ejercicio",
        "tab": "Laboratorio",
        "section": "Laboratorios interactivos",
    },
    {
        "id": "site_run",
        "text": (
            "Para ejecutar el portal: 'python -m streamlit run app.py'. La API de crédito se levanta con "
            "'python -m uvicorn src.api:app --port 8600'. Para regenerar artefactos de los laboratorios: 'python src/eval_export.py'."
        ),
        "page": "README",
        "tab": "Uso",
        "section": "Cómo ejecutar",
    },
]


def build_corpus() -> list[dict]:
    chunks: list[dict] = []

    def add(cid, text, page, tab, section):
        chunks.append({"id": cid, "text": text, "page": page, "tab": tab, "section": section})

    for item in SITE_INFO:
        add(item["id"], item["text"], item["page"], item["tab"], item["section"])

    for ex in EXERCISES:
        text = (
            f"Ejercicio '{ex['title']}'. Problema: {ex['desc']} Modelo: {ex['model']}. "
            f"Métricas: {ex['metrics']} Pestañas disponibles: {', '.join(ex['tabs'])}."
        )
        add(f"ex_{ex['id']}", text, ex["title"], "Documentación del Ejercicio", "Resumen del ejercicio")

    for g in GLOSSARY:
        add(f"glossary_{g['term']}", f"Glosario — {g['term']} ({g['category']}): {g['definition']}",
            "Aprende ML", "Glosario", g["term"])

    for key, (title, text) in CONCEPTS.items():
        add(f"concept_{key}", f"Concepto clave — {title}: {text}", "Aprende ML", "Conceptos", title)

    for i, item in enumerate(FAQ):
        add(f"faq_{i}", f"FAQ — {item['q']} Respuesta: {item['a']}", "Aprende ML", "FAQ técnico", item["q"])

    for r in ROUTE:
        text = (
            f"Ruta de aprendizaje — Paso {r['order']}: {r['title']} (nivel {r['level']}, tipo {r['type']}). "
            f"Objetivo: {r['goal']} Se aprende: {'; '.join(r['learn'])}."
        )
        add(f"route_{r['id']}", text, "Aprende ML", "Ruta de aprendizaje", r["title"])

    add("guide_intro", f"Guía para elegir modelo: {CHOOSE_MODEL_INTRO}", "Aprende ML", "¿Qué modelo elegir?", "Introducción")
    for i, row in enumerate(CHOOSE_MODEL_TABLE):
        text = " | ".join(f"{k}: {v}" for k, v in row.items())
        add(f"guide_{i}", f"Guía de elección de modelo — {text}", "Aprende ML", "¿Qué modelo elegir?", row["Modelo base sugerido"])

    return chunks
