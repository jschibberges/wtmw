from __future__ import annotations

import pandas as pd

import reflex_dashboard_redesign as rd


def _shows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": ["a", "b", "c", "d"],
            "date": pd.to_datetime(["2023-03-01", "2023-06-01", "2024-01-10", "2025-02-01"]),
            "show_display": ["Markus Lanz", "Anne Will", "Markus Lanz", "Markus Lanz"],
            "topic": [0.0, -1.0, float("nan"), 0.0],
            "topic_label": ["0 corona impfstoff", "-1 sonstige", None, "0 corona impfstoff"],
        }
    )


def _guests() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": ["a", "a", "b", "b", "c", "d"],
            "name": ["Anna Muster", "Bert Beispiel", "Anna Muster", "Bert Beispiel", "Anna Muster", "Anna Muster"],
            "date": pd.to_datetime(
                ["2023-03-01", "2023-03-01", "2023-06-01", "2023-06-01", "2024-01-10", "2025-02-01"]
            ),
            "show_display": ["Markus Lanz", "Markus Lanz", "Anne Will", "Anne Will", "Markus Lanz", "Markus Lanz"],
            "party_norm": ["SPD", "CDU", "SPD", "CDU", "BSW", "BSW"],
            "category_display": ["Politik"] * 6,
            "topic": [0.0, 0.0, -1.0, -1.0, float("nan"), 0.0],
            "topic_label": ["0 corona impfstoff", "0 corona impfstoff", "-1 sonstige", "-1 sonstige", None, "0 corona impfstoff"],
        }
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Anna Muster", "Bert Beispiel"], "known_roles": ["Politikerin", "Politiker"]})


def test_person_href_encodes_umlauts_and_spaces():
    assert rd.person_href("Markus Söder") == "/gaeste/Markus%20S%C3%B6der"


def test_stacked_timeline_marks_incomplete_years_with_partial_keys(monkeypatch):
    monkeypatch.setattr(rd, "INCOMPLETE_YEARS", {2025})

    rows, series = rd.build_stacked_timeline(_shows())

    by_period = {row["period"]: row for row in rows}
    assert by_period["2023"]["Markus Lanz"] == 1
    assert "Markus Lanz__p" not in by_period["2023"]
    assert by_period["2025"]["Markus Lanz__p"] == 1
    partial_flags = {(s["key"], s["partial"]) for s in series}
    assert ("Markus Lanz", "0") in partial_flags and ("Markus Lanz__p", "1") in partial_flags


def test_stacked_timeline_defaults_to_latest_year_incomplete(monkeypatch):
    monkeypatch.setattr(rd, "INCOMPLETE_YEARS", None)

    rows, _ = rd.build_stacked_timeline(_shows())

    assert any(key.endswith("__p") for key in rows[-1])
    assert not any(key.endswith("__p") for key in rows[0])


def test_classified_share_counts_outliers_and_missing_topics():
    share, note = rd.classified_share(_shows())

    assert share == "50 %"
    assert note == "2 Episoden ohne Thema"


def test_latest_party_map_uses_most_recent_party():
    assert rd.latest_party_map(_guests())["Anna Muster"] == "BSW"


def test_guest_rank_rows_sorted_with_latest_party_and_profile_link():
    rows = rd.build_guest_rank_rows(_guests(), _metadata(), "appearances", limit=5)

    assert [r["name"] for r in rows] == ["Anna Muster", "Bert Beispiel"]
    assert rows[0]["party"] == "BSW"
    assert rows[0]["value"] == "4"
    assert rows[0]["width"] == "100.0%"
    assert rows[0]["href"] == "/gaeste/Anna%20Muster"


def test_pair_rows_are_strongest_first_and_split_names():
    rows = rd.build_pair_rows(_guests(), limit=3)

    assert rows == [
        {
            "a": "Anna Muster",
            "b": "Bert Beispiel",
            "a_href": "/gaeste/Anna%20Muster",
            "b_href": "/gaeste/Bert%20Beispiel",
            "value": "2",
        }
    ]


def test_topic_list_flags_format_clusters_without_prefix(monkeypatch):
    monkeypatch.setattr(
        rd,
        "format_topic_label",
        lambda raw: "Format-Cluster: Talkrunde" if isinstance(raw, str) and raw.startswith("0") else "Sonstige",
    )

    rows = rd.build_topic_list_rows(_shows())

    assert len(rows) == 1
    assert rows[0]["label"] == "Talkrunde"
    assert rows[0]["is_format"] == "1"
    assert rows[0]["episodes"] == "2"


def test_default_topic_skips_format_clusters(monkeypatch):
    shows = pd.DataFrame(
        {
            "uid": ["a", "b", "c"],
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
            "topic": [1.0, 1.0, 2.0],
            "topic_label": ["1 runde", "1 runde", "2 klima"],
        }
    )
    monkeypatch.setattr(
        rd,
        "format_topic_label",
        lambda raw: "Format-Cluster: Runde" if raw.startswith("1") else "Klima",
    )

    assert rd.default_topic_id(shows) == "2"


def test_topic_detail_reports_share_peak_and_guests():
    detail = rd.build_topic_detail(_shows(), _guests(), "0")

    assert detail["episodes"] == "2"
    assert detail["share"] == "50 %"
    assert detail["peak"] == "2023 (1)"
    assert detail["guests"][0]["name"] == "Anna Muster"
    assert rd.build_topic_detail(_shows(), _guests(), "99") == {}


def test_person_profile_summarises_appearances_and_coguests():
    profile = rd.build_person_profile(_guests(), _metadata(), "Anna Muster")

    assert profile["appearances"] == "4"
    assert profile["party"] == "BSW"
    assert profile["rank_note"] == "Rang 1 von 2"
    assert profile["span"] == "2023–2025"
    assert profile["co_guests"] == [
        {"name": "Bert Beispiel", "href": "/gaeste/Bert%20Beispiel", "value": "2"}
    ]
    assert rd.build_person_profile(_guests(), _metadata(), "Niemand") == {}


def test_search_suggestions_match_case_insensitively_and_ignore_blank_query():
    hits = rd.search_suggestions(_guests(), "MUST")

    assert [h["name"] for h in hits] == ["Anna Muster"]
    assert hits[0]["party"] == "BSW"
    assert rd.search_suggestions(_guests(), "  ") == []


def test_topic_label_mismatches_detects_stale_keywords_and_missing_ids(tmp_path, monkeypatch):
    (tmp_path / "talkshow_topic_model").mkdir()
    (tmp_path / "talkshow_topic_model" / "topics.json").write_text(
        '{"topic_representations": {"0": [["corona", 1], ["impfstoff", 1]], "1": [["klima", 1]]}}',
        encoding="utf-8",
    )
    (tmp_path / "topic_labels.json").write_text(
        '{"-1": "Sonstige",'
        ' "0": {"label": "Corona", "keywords": ["corona"]},'
        ' "1": {"label": "Ukraine", "keywords": ["putin"]},'
        ' "7": {"label": "Weg", "keywords": ["x"]}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(rd, "DATA_DIR", tmp_path)

    stale = {m["id"] for m in rd.topic_label_mismatches()}

    assert stale == {"1", "7"}


def test_short_party_abbreviates_long_names_and_defaults_to_dash():
    assert rd._short_party("Bündnis 90/Die Grünen") == "Grüne"
    assert rd._short_party("SPD") == "SPD"
    assert rd._short_party(None) == "–"


def test_kpis_use_singular_for_a_single_show():
    only_lanz = _shows()[lambda d: d["show_display"] == "Markus Lanz"]

    kpis = rd.build_kpis(only_lanz, _guests())

    assert kpis[0] == {"label": "Episoden", "value": "3", "note": "1 Sendung"}
    assert rd.build_kpis(_shows(), _guests())[0]["note"] == "2 Sendungen"
