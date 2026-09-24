from __future__ import annotations

import importlib

import networkx as nx
import pandas as pd

import analyze_talkshows


def test_analyze_talkshows_imports_cleanly():
    module = importlib.import_module("analyze_talkshows")
    assert hasattr(module, "perform_guest_analysis")


def test_perform_guest_analysis_writes_canonicalized_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_talkshows, "DATA_DIR", tmp_path)

    all_data = [
        {
            "uid": "ep-1",
            "show": "Anne Will",
            "date": "01.12.2024",
            "guests": [
                {
                    "name": "Alice Example - SPD",
                    "party": "SPD",
                    "role": "Bundesministerin",
                    "description": "Bundesministerin und SPD-Politikerin",
                },
                {
                    "name": "Bob Example",
                    "party": None,
                    "role": "Journalist",
                    "description": "Journalist bei der Zeitung",
                },
            ],
        },
        {
            "uid": "ep-2",
            "show": "Anne Will",
            "date": "08.12.2024",
            "guests": [
                {
                    "name": "Alice Example - SPD",
                    "party": "SPD",
                    "role": "Bundesministerin",
                    "description": "Bundesministerin und SPD-Politikerin",
                }
            ],
        },
    ]
    df_with_topics = pd.DataFrame(
        {
            "uid": ["ep-1", "ep-2"],
            "topic": [13, 7],
            "topic_label": ["13 klima energie", "7 haushalt wirtschaft"],
        }
    )

    df_guests_with_topics, _, df_consolidated = analyze_talkshows.perform_guest_analysis(
        all_data, df_with_topics
    )

    assert "name" in df_guests_with_topics.columns
    assert "name_clean" not in df_guests_with_topics.columns
    assert sorted(df_guests_with_topics["name"].unique().tolist()) == ["Alice Example", "Bob Example"]

    saved_guest_topics = pd.read_excel(tmp_path / "guests_with_topics.xlsx")
    assert "name" in saved_guest_topics.columns
    assert "name_clean" not in saved_guest_topics.columns
    assert {"CategoryPrimary", "Categories", "Confidence", "CategoryScores"}.issubset(
        saved_guest_topics.columns
    )
    assert sorted(saved_guest_topics["name"].unique().tolist()) == ["Alice Example", "Bob Example"]
    alice_topics_row = saved_guest_topics.loc[saved_guest_topics["name"] == "Alice Example"].iloc[0]
    assert alice_topics_row["CategoryPrimary"] == "Politics & Government"

    saved_consolidated = pd.read_excel(tmp_path / "guests_consolidated.xlsx")
    assert {"Categories", "CategoryPrimary", "Confidence", "CategoryScores"}.issubset(saved_consolidated.columns)
    alice_row = saved_consolidated.loc[saved_consolidated["name"] == "Alice Example"].iloc[0]
    assert alice_row["CategoryPrimary"] == "Politics & Government"
    assert "CategoryPrimary" in df_consolidated.columns


def test_load_all_show_data_normalizes_two_digit_year_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_talkshows, "DATA_DIR", tmp_path)

    show_payload = [
        {
            "uid": "ep-1",
            "title": "Test",
            "description": "Beschreibung",
            "show": "Caren Miosga",
            "date": "01.09.24",
            "guests": [],
        }
    ]

    for file_name in [
        "AnneWill_data.json",
        "CarenMiosga_data.json",
        "HartAberFair_data.json",
        "MarkusLanz_data.json",
        "Maischberger_data.json",
        "Illner_data.json",
    ]:
        payload = show_payload if file_name == "CarenMiosga_data.json" else []
        (tmp_path / file_name).write_text(importlib.import_module("json").dumps(payload), encoding="utf-8")

    all_data, df = analyze_talkshows.load_all_show_data()

    assert all_data[0]["date"] == "01.09.2024"
    assert df.loc[0, "date"] == "01.09.2024"


def test_load_all_show_data_excludes_episodes_before_2015(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_talkshows, "DATA_DIR", tmp_path)

    payload = [
        {
            "uid": "ep-old",
            "title": "Alt",
            "description": "Beschreibung",
            "show": "Caren Miosga",
            "date": "31.12.2014",
            "guests": [],
        },
        {
            "uid": "ep-new",
            "title": "Neu",
            "description": "Beschreibung",
            "show": "Caren Miosga",
            "date": "01.01.2015",
            "guests": [],
        },
    ]

    for file_name in [
        "AnneWill_data.json",
        "CarenMiosga_data.json",
        "HartAberFair_data.json",
        "MarkusLanz_data.json",
        "Maischberger_data.json",
        "Illner_data.json",
    ]:
        file_payload = payload if file_name == "CarenMiosga_data.json" else []
        (tmp_path / file_name).write_text(importlib.import_module("json").dumps(file_payload), encoding="utf-8")

    all_data, df = analyze_talkshows.load_all_show_data()

    assert [row["uid"] for row in all_data] == ["ep-new"]
    assert df["uid"].tolist() == ["ep-new"]
    assert df.loc[0, "date"] == "01.01.2015"


def test_prepare_guest_dataframe_normalizes_missing_and_non_string_text_fields():
    all_data = [
        {
            "uid": "ep-1",
            "show": "Anne Will",
            "date": "01.12.2024",
            "guests": [
                {
                    "name": "Alice Example",
                    "role": None,
                    "description": float("nan"),
                },
                {
                    "name": "Bob Example",
                },
            ],
        }
    ]

    df_guests = analyze_talkshows.prepare_guest_dataframe(all_data)

    assert df_guests["name"].tolist() == ["Alice Example", "Bob Example"]
    assert df_guests["role"].tolist() == ["", ""]
    assert df_guests["description"].tolist() == ["", ""]


def test_prepare_static_network_view_filters_weak_edges_and_focuses_on_largest_component():
    G = nx.Graph()
    G.add_node("A", appearances=10)
    G.add_node("B", appearances=9)
    G.add_node("C", appearances=8)
    G.add_node("D", appearances=7)
    G.add_edge("A", "B", weight=3)
    G.add_edge("B", "C", weight=1)
    G.add_edge("C", "D", weight=4)

    H = analyze_talkshows._prepare_static_network_view(G, min_edge_weight=2, max_nodes=10)

    assert set(H.nodes()) in ({"A", "B"}, {"C", "D"})
    assert sorted(H.edges(data="weight")) in ([("A", "B", 3)], [("C", "D", 4)])


def test_prepare_static_network_view_raises_edge_threshold_until_node_budget_is_met():
    G = nx.Graph()
    for node, appearances in [("A", 10), ("B", 9), ("C", 8), ("D", 7)]:
        G.add_node(node, appearances=appearances)
    G.add_edge("A", "B", weight=2)
    G.add_edge("B", "C", weight=3)
    G.add_edge("C", "D", weight=4)

    H = analyze_talkshows._prepare_static_network_view(G, min_edge_weight=2, max_nodes=2)

    assert set(H.nodes()) == {"C", "D"}
    assert sorted(H.edges(data="weight")) == [("C", "D", 4)]


def test_select_label_nodes_prefers_high_appearance_nodes():
    G = nx.Graph()
    G.add_node("A", appearances=2)
    G.add_node("B", appearances=10)
    G.add_node("C", appearances=6)
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)

    labels = analyze_talkshows._select_label_nodes(G, top_n=2)

    assert labels == ["B", "C"]
