"""Asistente RAG del portal: recuperación TF-IDF sobre el corpus y
generación con DeepSeek (API compatible con OpenAI). Si no hay clave,
responde en modo 'solo recuperación' con las fuentes."""

import os
import unicodedata
from functools import lru_cache

import numpy as np
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.assistant_corpus import build_corpus

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
TOP_K = 5


def _normalize(text: str) -> str:
    """Minúsculas y sin acentos, para que 'calibracion' encuentre 'Calibración'."""
    text = text.lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))

SYSTEM_PROMPT = (
    "Eres el asistente del 'Centro de Modelos de Machine Learning', un portal educativo con seis ejercicios "
    "(clasificación, regresión y NLP). Respondes SIEMPRE en español, de forma breve y clara (máximo ~120 palabras). "
    "Usa EXCLUSIVAMENTE la información del CONTEXTO. Si la respuesta no está en el contexto, dilo y sugiere la sección "
    "más cercana. Cuando des un dato, indica dónde encontrarlo en el sitio citando las fuentes entre corchetes, "
    "por ejemplo [Scoring de Crédito → Documentación de Modelos] o [Aprende ML → Glosario]."
)


def get_api_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("DEEPSEEK_API_KEY")
    except Exception:
        return None


def has_llm() -> bool:
    return bool(get_api_key())


@lru_cache(maxsize=1)
def _build_index() -> dict:
    corpus = build_corpus()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([_normalize(c["text"]) for c in corpus])
    return {"corpus": corpus, "vectorizer": vectorizer, "matrix": matrix}


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    index = _build_index()
    query_vec = index["vectorizer"].transform([_normalize(query)])
    scores = cosine_similarity(query_vec, index["matrix"])[0]
    order = np.argsort(scores)[::-1][:k]
    results = []
    for idx in order:
        if scores[idx] <= 0:
            continue
        chunk = dict(index["corpus"][idx])
        chunk["score"] = float(scores[idx])
        results.append(chunk)
    return results


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({c['page']} → {c['tab']} → {c['section']}) {c['text']}")
    return "\n".join(lines)


def _call_deepseek(query: str, context: str, history: list[dict] | None, key: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {query}",
        }
    )
    response = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 700,
            "stream": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _fallback_answer(chunks: list[dict]) -> str:
    if not chunks:
        return "No encontré información sobre eso en el portal. Prueba a preguntar por un ejercicio, una métrica o un concepto del glosario."
    lines = [
        "No hay una clave de LLM configurada, así que te muestro lo más relevante que encontré en el sitio:",
    ]
    for c in chunks[:3]:
        snippet = c["text"]
        if len(snippet) > 260:
            snippet = snippet[:260].rsplit(" ", 1)[0] + "..."
        lines.append(f"- **{c['page']} → {c['tab']}**: {snippet}")
    return "\n\n".join(lines)


def answer(query: str, history: list[dict] | None = None) -> dict:
    """Devuelve {'answer', 'sources', 'used_llm', 'error'}."""
    query = (query or "").strip()
    if not query:
        return {"answer": "Escribe una pregunta sobre el portal.", "sources": [], "used_llm": False, "error": None}

    chunks = retrieve(query)
    sources = [
        {"page": c["page"], "tab": c["tab"], "section": c["section"], "score": c["score"]}
        for c in chunks
    ]

    key = get_api_key()
    if not key:
        return {"answer": _fallback_answer(chunks), "sources": sources, "used_llm": False, "error": None}

    try:
        text = _call_deepseek(query, _format_context(chunks), history, key)
        return {"answer": text, "sources": sources, "used_llm": True, "error": None}
    except Exception as exc:  # noqa: BLE001 - degradar siempre a recuperación
        fallback = _fallback_answer(chunks)
        return {
            "answer": f"No pude contactar al modelo ({type(exc).__name__}). {fallback}",
            "sources": sources,
            "used_llm": False,
            "error": str(exc),
        }
