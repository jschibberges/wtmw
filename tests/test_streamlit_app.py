"""Smoke-Test: jede Seite des Streamlit-Dashboards läuft mit den Daten aus data/ ohne Fehler."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
from streamlit.util import calc_md5  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"
DATA = APP.parent / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "all_data_with_topics.xlsx").exists() or not (DATA / "guests_with_topics.xlsx").exists(),
    reason="Auswertungsdateien fehlen",
)


def _run(url_path: str = "", **query: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    if url_path:
        # AppTest.switch_page kennt nur Seiten-Dateien; st.Page(callable) wird über
        # den Hash seines url_path ausgewählt.
        at._page_hash = calc_md5(url_path)
    for key, value in query.items():
        at.query_params[key] = value
    return at.run()


def _html(at: AppTest) -> str:
    return " ".join(el.proto.body for el in at.get("html"))


@pytest.mark.parametrize(
    ("url_path", "query", "expected"),
    [
        ("", {}, "Erfasste Episoden pro Jahr"),
        ("gaeste", {}, "Rangliste"),
        ("themen", {}, "Häufigste Gäste zum Thema"),
        ("netzwerke", {}, "Stärkste Verbindungen"),
        ("gaeste", {"zeitraum": "2 Jahre", "kategorie": "Politik"}, "Rangliste"),
    ],
)
def test_pages_render(url_path: str, query: dict[str, str], expected: str) -> None:
    at = _run(url_path, **query)
    assert not at.exception
    assert expected in _html(at)


def test_person_profile_and_filters_from_url() -> None:
    at = _run("gaeste", zeitraum="2 Jahre", sendung="Markus Lanz")
    first = re.search(r'class="rank-row"[^>]*href="([^"]+)"', _html(at)).group(1)
    # Links tragen die aktiven Filter weiter.
    assert "zeitraum=2+Jahre" in first and "sendung=Markus+Lanz" in first and "person=" in first

    at = _run("gaeste", person="Markus Söder")
    assert not at.exception
    assert "Häufigste Mitgäste" in _html(at)

    at = _run("gaeste", person="Niemand Unbekannt")
    assert "Person nicht gefunden" in _html(at)


def test_invalid_filter_value_falls_back_to_default() -> None:
    at = _run("", sendung="Gibt es nicht")
    assert not at.exception
    assert "sendung" not in at.query_params
