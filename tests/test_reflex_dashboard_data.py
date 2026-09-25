from __future__ import annotations

import networkx as nx

from reflex_dashboard_data import _build_network_summary


def test_build_network_summary_extracts_metrics_and_highlights(tmp_path):
    graph = nx.Graph()
    graph.add_edge("Alice Example", "Bob Example", weight=3)
    graph.add_edge("Bob Example", "Carla Example", weight=1)
    graph.add_node("Dana Example")

    gexf_path = tmp_path / "network.gexf"
    nx.write_gexf(graph, gexf_path)

    summary = _build_network_summary(
        gexf_path,
        {
            "node_kind": "Gaeste",
            "edge_kind": "gemeinsame Auftritte",
            "hub_kind": "gewichtete Verknuepfungen",
            "scope_note": "Globaler Export",
        },
    )

    assert summary["scope_note"] == "Globaler Export"
    assert summary["insight_badges"] == [
        "Gesamtdatensatz",
        "Knoten: Gaeste",
        "Kanten: gemeinsame Auftritte",
    ]
    assert summary["metrics"] == [
        {"label": "Gaeste", "value": "4"},
        {"label": "Verbindungen", "value": "2"},
        {"label": "Hauptcluster", "value": "75 %"},
    ]
    assert summary["highlights"] == [
        {
            "label": "Stärkster Hub",
            "value": "Bob Example (4 gewichtete Verknuepfungen)",
        },
        {
            "label": "Stärkste Verbindung",
            "value": "Alice Example <-> Bob Example (3 gemeinsame Auftritte)",
        },
    ]
    assert summary["exported_at"] != "–"


def test_build_network_summary_returns_empty_dict_for_missing_file(tmp_path):
    summary = _build_network_summary(
        tmp_path / "missing.gexf",
        {"node_kind": "Themen", "edge_kind": "geteilte Gaeste"},
    )

    assert summary == {}
