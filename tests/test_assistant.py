"""Tests del asistente RAG: recuperación y modo sin LLM (fallback)."""

import src.assistant_rag as rag


def test_corpus_has_chunks() -> None:
    from src.assistant_corpus import build_corpus

    corpus = build_corpus()
    assert len(corpus) > 50
    assert all({"id", "text", "page", "tab", "section"} <= set(chunk) for chunk in corpus)


def test_retrieve_is_accent_insensitive() -> None:
    hits = rag.retrieve("que es la calibracion", k=5)
    assert hits
    joined = " ".join(f"{h['text']} {h['section']}" for h in hits).lower()
    assert "calibraci" in joined


def test_retrieve_finds_topic() -> None:
    hits = rag.retrieve("que es tf-idf", k=5)
    assert any("tf-idf" in h["text"].lower() for h in hits)


def test_fallback_answer_without_llm(monkeypatch) -> None:
    monkeypatch.setattr(rag, "get_api_key", lambda: None)
    result = rag.answer("que es la calibracion")
    assert result["used_llm"] is False
    assert result["answer"]
    assert result["sources"]
