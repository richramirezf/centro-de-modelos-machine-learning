"""Corpus del asistente RAG: convierte el contenido educativo y la
documentación de cada ejercicio en fragmentos con metadatos de ubicación
(página, pestaña, sección) para poder citar dónde encontrar cada cosa.

Los ejercicios se derivan de ``src/manifest.py`` (fuente única de verdad).
"""

from src.academy_content import (
    CHOOSE_MODEL_INTRO,
    CHOOSE_MODEL_TABLE,
    CONCEPTS,
    FAQ,
    GLOSSARY,
    ROUTE,
)
from src.manifest import EXERCISES as _MANIFEST_EXERCISES

EXERCISES = [
    {
        "id": ex["id"],
        "title": ex["title"],
        "page": ex["page"],
        "tabs": ex["tabs"],
        "model": ex["model"],
        "metrics": ex["metrics"],
        "desc": ex["detail"],
    }
    for ex in _MANIFEST_EXERCISES
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
