from __future__ import annotations

import pandas as pd

import app_helpers
from app_helpers import (
    filter_by_timeframe,
    format_topic_label,
    prepare_guest_metadata,
    summarize_topic_counts,
)


def test_format_topic_label_handles_topic_prefixes_and_empty_values(monkeypatch):
    monkeypatch.setattr(app_helpers, "_load_topic_labels", lambda: {})

    assert format_topic_label("13 klima energie wirtschaft") == "Topic 13: klima energie wirtschaft"
    assert format_topic_label("-1 irgendwas") == "Nicht klassifiziert"
    assert format_topic_label("") == "Nicht klassifiziert"
    assert format_topic_label(None) == "Nicht klassifiziert"


def test_format_topic_label_prefers_custom_labels(monkeypatch):
    monkeypatch.setattr(
        app_helpers,
        "_load_topic_labels",
        lambda: {"13": "Ernährung & Gesundheit", "-1": "Sonstige / Nicht zugeordnet"},
    )

    assert format_topic_label("13 klima energie wirtschaft") == "Ernährung & Gesundheit"
    assert format_topic_label("-1 irgendwas") == "Sonstige / Nicht zugeordnet"


def test_filter_by_timeframe_limits_rows_relative_to_latest_date():
    df = pd.DataFrame(
        {
            "uid": ["a", "b", "c"],
            "date": ["2024-01-15", "2024-08-01", "2025-01-15"],
        }
    )

    filtered, start_date, latest_date = filter_by_timeframe(df, "6 Monate")

    assert latest_date == pd.Timestamp("2025-01-15")
    assert start_date == pd.Timestamp("2024-07-15")
    assert filtered["uid"].tolist() == ["b", "c"]


def test_summarize_topic_counts_preserves_unclassified_buckets(monkeypatch):
    monkeypatch.setattr(app_helpers, "_load_topic_labels", lambda: {})

    df = pd.DataFrame(
        {
            "uid": ["a", "b", "c", "d"],
            "topic": [13, -1, None, 13],
            "topic_label": ["13 klima energie", "12 ignored", None, "13 klima energie"],
        }
    )

    result = summarize_topic_counts(df)
    counts = dict(zip(result["Thema"], result["Episoden"]))

    assert counts["Topic 13: klima energie"] == 2
    assert counts["Nicht klassifiziert"] == 2


def test_prepare_guest_metadata_deduplicates_party_tokens_and_sorts_roles_by_recency():
    df = pd.DataFrame(
        {
            "name": ["Karl Lauterbach"] * 5,
            "party": ["SPD"] * 5,
            "role": [
                "SPD",
                "SPD, Mitglied des Deutschen Bundestages, Gesundheitsökonom und Epidemiologe",
                "Bundesminister für Gesundheit, SPD",
                "SPD, Bundesminister für Gesundheit",
                "SPD, Mitglied des Deutschen Bundestags, Gesundheitsökonom und Epidemiologe",
            ],
            "date": [
                "2021-01-01",
                "2021-05-01",
                "2024-01-15",
                "2024-05-01",
                "2022-01-01",
            ],
        }
    )

    metadata = prepare_guest_metadata(df)

    assert metadata.loc[0, "primary_party"] == "SPD"
    assert (
        metadata.loc[0, "known_roles"]
        == "Bundesminister für Gesundheit, Mitglied des Deutschen Bundestages, Gesundheitsökonom und Epidemiologe"
    )
