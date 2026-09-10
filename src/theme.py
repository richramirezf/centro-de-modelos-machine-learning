"""Tema visual único (teal / Material 3) para todas las páginas del portal.

Centraliza el CSS que antes estaba duplicado en ``src/docs.py``,
``app_pages/credit_scoring.py``, ``app_pages/noshow.py`` y
``app_pages/telco_churn.py``, e incluye el botón flotante del asistente.
"""

import streamlit as st

_BASE_CSS = """
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

  [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid rgba(0, 121, 107, .18) !important;
    border-radius: 16px;
    box-shadow: var(--shadow-2);
    padding: .35rem .6rem;
  }

  [data-testid="stVerticalBlockBorderWrapper"].st-key-engine {
    border-left: 5px solid var(--teal-500) !important;
    box-shadow: var(--shadow-1);
  }

  .st-key-engine [data-testid="stWidgetLabel"] p {
    color: var(--teal-700);
    font-size: 1.05rem;
    font-weight: 600;
  }

  .empty-hint { color: var(--text-muted); }

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

  .result-card {
    background: var(--surface);
    border-radius: 16px;
    box-shadow: var(--shadow-2);
    padding: 1.5rem 2rem;
    text-align: center;
  }

  .verdict {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: .3px;
  }

  .verdict-sub {
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 500;
    margin-top: .15rem;
  }

  .prob-value {
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: .25rem 0;
  }

  .prob-label {
    color: var(--text-muted);
    font-size: .9rem;
    letter-spacing: .4px;
    text-transform: uppercase;
  }

  .st-key-assistant_fab {
    position: fixed;
    bottom: 1.25rem;
    right: 1.25rem;
    z-index: 9999;
    width: auto;
  }
  .st-key-assistant_fab button {
    border-radius: 999px !important;
    padding: .65rem 1.1rem !important;
    box-shadow: 0 6px 16px rgba(0, 0, 0, .25) !important;
    font-weight: 600;
  }
</style>
"""

_CREDIT_CARDS = """
<style>
  .result-card.approved { border-top: 5px solid #1b7f3b; }
  .result-card.denied   { border-top: 5px solid #c3352b; }
  .verdict.approved { color: #1b7f3b; }
  .verdict.denied   { color: #c3352b; }
  .prob-value.approved { color: #1b7f3b; }
  .prob-value.denied   { color: #c3352b; }
</style>
"""

_NOSHOW_CARDS = """
<style>
  .result-card.low      { border-top: 5px solid #1b7f3b; }
  .result-card.moderate { border-top: 5px solid #b45309; }
  .result-card.high     { border-top: 5px solid #c3352b; }
  .verdict.low      { color: #1b7f3b; }
  .verdict.moderate { color: #b45309; }
  .verdict.high     { color: #c3352b; }
  .prob-value.low      { color: #1b7f3b; }
  .prob-value.moderate { color: #b45309; }
  .prob-value.high     { color: #c3352b; }
</style>
"""

_CARD_VARIANTS = {"credit": _CREDIT_CARDS, "noshow": _NOSHOW_CARDS}


def apply_theme(result_cards: str | None = None) -> None:
    """Inyecta el tema base y, opcionalmente, una variante de tarjeta de resultado.

    ``result_cards`` acepta ``"credit"`` (approved/denied) o ``"noshow"``
    (low/moderate/high). Cualquier otro valor inyecta solo el tema base.
    """
    css = _BASE_CSS
    variant = _CARD_VARIANTS.get(result_cards or "")
    if variant:
        css += variant
    st.html(css)
