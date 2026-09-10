"""Smoke tests: cada página del portal debe renderizar sin excepciones."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAGES = [
    "app_pages/inicio.py",
    "app_pages/aprende.py",
    "app_pages/credit_scoring.py",
    "app_pages/noshow.py",
    "app_pages/demand.py",
    "app_pages/housing.py",
    "app_pages/intent.py",
]

TELCO_PAGE = "app_pages/telco_churn.py"
TELCO_CSV = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page: str) -> None:
    at = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=240)
    at.run()
    assert not at.exception, [f"{e.type}: {e.value}" for e in at.exception]


def test_telco_page_renders_without_exceptions() -> None:
    if not TELCO_CSV.exists():
        pytest.skip("Dataset de Telco no disponible en data/")
    at = AppTest.from_file(str(PROJECT_ROOT / TELCO_PAGE), default_timeout=240)
    at.run()
    assert not at.exception, [f"{e.type}: {e.value}" for e in at.exception]


def test_router_renders_without_exceptions() -> None:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception, [f"{e.type}: {e.value}" for e in at.exception]
    assert any(button.label == "Asistente" for button in at.button)
