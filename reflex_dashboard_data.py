from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd

from app_helpers import (
    TIMEFRAME_OPTIONS,
    filter_by_timeframe,
    format_topic_label,
    prepare_guest_metadata,
    summarize_topic_counts,
)
from date_utils import coerce_mixed_date_series, filter_to_analysis_window

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
NETWORK_ASSETS_DIR = ASSETS_DIR / "generated_networks"

SHOW_FILE = DATA_DIR / "all_data_with_topics.xlsx"
GUEST_FILE = DATA_DIR / "guests_with_topics.xlsx"

ALL_SHOWS = "Alle Sendungen"
ALL_CATEGORIES = "Alle Kategorien"
DEFAULT_TIMEFRAME = "Gesamter Zeitraum"

SHOW_LABELS = {
    "Anne Will": "Anne Will",
    "Caren Miosga": "Caren Miosga",
    "Markus Lanz": "Markus Lanz",
    "Maybrit Illner": "Maybrit Illner",
    "hart aber fair": "Hart aber Fair",
    "Hart aber Fair": "Hart aber Fair",
    "maischberger": "Maischberger",
    "Maischberger": "Maischberger",
}

CATEGORY_LABELS = {
    "Academia & Expertise": "Wissenschaft & Expertise",
    "Arts & Culture": "Kultur",
    "Business & Economy": "Wirtschaft",
    "Citizens & Everyday Voices": "Buergerliche Stimmen",
    "Civil Society & Advocacy": "Zivilgesellschaft",
    "Influencers & Digital Creators": "Digitale Oeffentlichkeit",
    "Media & Communication": "Medien",
    "Politics & Government": "Politik",
    "Religion & Spirituality": "Religion",
    "Sports": "Sport",
}

NETWORK_ASSET_SPECS: dict[str, dict[str, str]] = {
    "guest_network": {
        "title": "Gaeste-Netzwerk",
        "description": (
            "Ko-Auftritte von Gaesten. Die statische Ansicht betont Cluster, "
            "die interaktive Version eignet sich fuer Exploration und Hover-Details."
        ),
        "image": "cooccurrence_network.png",
        "html": "cooccurrence_network.html",
        "gexf": "cooccurrence_network_with_weights.gexf",
        "node_kind": "Gaeste",
        "edge_kind": "gemeinsame Auftritte",
        "hub_kind": "gewichtete Verknuepfungen",
        "scope_note": "Globaler Export aus dem Gesamtdatensatz; reagiert nicht auf die Filter oben.",
    },
    "topic_network": {
        "title": "Themen-Netzwerk",
        "description": (
            "Themencluster und ihre geteilten Gaeste. Knoten stehen fuer Themen, "
            "Verbindungen fuer personelle Ueberlappungen."
        ),
        "image": "topic_cooccurrence_network.png",
        "html": "topic_cooccurrence_network.html",
        "gexf": "topic_cooccurrence_network.gexf",
        "node_kind": "Themen",
        "edge_kind": "geteilte Gaeste",
        "hub_kind": "gewichtete Verknuepfungen",
        "scope_note": "Globaler Export aus dem Gesamtdatensatz; reagiert nicht auf die Filter oben.",
    },
    "topics_visualization": {
        "title": "BERTopic-Cluster",
        "description": "Interaktive Themenkarte des trainierten Modells.",
        "html": "topics_visualization.html",
    },
    "topics_over_time": {
        "title": "Themen im Zeitverlauf",
        "description": "Entwicklung der Themen ueber die Jahre hinweg.",
        "html": "topics_over_time.html",
    },
    "topics_hierarchy": {
        "title": "Themen-Hierarchie",
        "description": "Hierarchische Verdichtung und Verwandtschaft der Topics.",
        "html": "topics_hierarchy.html",
    },
    "topics_barchart": {
        "title": "Themen-Haeufigkeit",
        "description": "Interaktives Balkendiagramm der Themenverteilung.",
        "html": "topics_barchart.html",
    },
}


@dataclass(frozen=True)
class DashboardBundle:
    shows: pd.DataFrame
    guests: pd.DataFrame
    guest_metadata: pd.DataFrame
    show_options: list[str]
    category_options: list[str]
    error: str | None = None


def display_show_name(show: object) -> str:
    if pd.isna(show):
        return "Unbekannte Sendung"
    text = "" if show is None else str(show).strip()
    return SHOW_LABELS.get(text, text or "Unbekannte Sendung")


def display_category_name(category: object) -> str:
    if pd.isna(category):
        return "Nicht zugeordnet"
    text = "" if category is None else str(category).strip()
    return CATEGORY_LABELS.get(text, text or "Nicht zugeordnet")


def compact_label(text: object, max_length: int = 34) -> str:
    if text is None:
        return "–"
    value = " ".join(str(text).split())
    if len(value) <= max_length:
        return value
    return value[: max_length - 3].rstrip() + "..."


def format_count(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(value):,}".replace(",", ".")


def format_date_range(
    start_date: pd.Timestamp | None, end_date: pd.Timestamp | None
) -> str:
    if start_date is None or pd.isna(start_date):
        return "–"
    if end_date is None or pd.isna(end_date):
        return pd.Timestamp(start_date).strftime("%d.%m.%Y")
    return (
        f"{pd.Timestamp(start_date).strftime('%d.%m.%Y')} – "
        f"{pd.Timestamp(end_date).strftime('%d.%m.%Y')}"
    )


def _format_share(part: int, total: int) -> str:
    if total <= 0:
        return "0 %"
    return f"{round((part / total) * 100):.0f} %"


def _format_metric_number(value: object) -> str:
    if isinstance(value, (int, float)) and not pd.isna(value):
        numeric = float(value)
        if numeric.is_integer():
            return format_count(int(numeric))
        return f"{numeric:.1f}".replace(".", ",")
    return "0"


def _build_network_summary(gexf_path: Path, spec: dict[str, str]) -> dict[str, object]:
    if not gexf_path.exists():
        return {}

    try:
        graph = nx.read_gexf(gexf_path)
    except Exception:
        return {}

    if graph.number_of_nodes() == 0:
        return {}

    analysis_graph = graph.to_undirected() if graph.is_directed() else graph
    largest_component = max((len(nodes) for nodes in nx.connected_components(analysis_graph)), default=0)

    weighted_degrees = {
        str(node): float(graph.degree(node, weight="weight"))
        for node in graph.nodes()
    }
    top_node, top_node_weight = max(weighted_degrees.items(), key=lambda item: (item[1], item[0]))

    top_edge = max(
        graph.edges(data=True),
        key=lambda item: (float(item[2].get("weight", 0) or 0), str(item[0]), str(item[1])),
        default=None,
    )

    highlights = [
        {
            "label": "Staerkster Hub",
            "value": (
                f"{compact_label(top_node, 34)} "
                f"({_format_metric_number(top_node_weight)} "
                f"{spec.get('hub_kind', 'Verknuepfungen')})"
            ),
        }
    ]
    if top_edge is not None:
        source, target, edge_data = top_edge
        highlights.append(
            {
                "label": "Staerkste Verbindung",
                "value": (
                    f"{compact_label(source, 28)} <-> {compact_label(target, 28)} "
                    f"({_format_metric_number(edge_data.get('weight', 0))} "
                    f"{spec.get('edge_kind', 'Verbindungen')})"
                ),
            }
        )

    return {
        "scope_note": spec.get(
            "scope_note",
            "Globaler Export aus dem Gesamtdatensatz; reagiert nicht auf die Filter oben.",
        ),
        "insight_badges": [
            "Gesamtdatensatz",
            f"Knoten: {spec.get('node_kind', 'Elemente')}",
            f"Kanten: {spec.get('edge_kind', 'Verbindungen')}",
        ],
        "metrics": [
            {"label": spec.get("node_kind", "Elemente"), "value": format_count(graph.number_of_nodes())},
            {"label": "Verbindungen", "value": format_count(graph.number_of_edges())},
            {"label": "Hauptcluster", "value": _format_share(largest_component, graph.number_of_nodes())},
        ],
        "highlights": highlights,
        "exported_at": format_timestamp_label(gexf_path.stat().st_mtime),
    }


def ensure_network_assets() -> dict[str, dict[str, object]]:
    NETWORK_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    synced_assets: dict[str, dict[str, object]] = {}
    for asset_key, spec in NETWORK_ASSET_SPECS.items():
        entry: dict[str, object] = {
            "title": spec["title"],
            "description": spec["description"],
        }
        for field in ("image", "html"):
            file_name = spec.get(field)
            if not file_name:
                continue
            source_path = DATA_DIR / file_name
            if not source_path.exists():
                entry[f"has_{field}"] = False
                continue

            target_path = NETWORK_ASSETS_DIR / file_name
            if (
                not target_path.exists()
                or source_path.stat().st_mtime > target_path.stat().st_mtime
                or source_path.stat().st_size != target_path.stat().st_size
            ):
                shutil.copy2(source_path, target_path)

            entry[f"has_{field}"] = True
            entry[f"{field}_src"] = f"/generated_networks/{file_name}"

        gexf_name = spec.get("gexf")
        if gexf_name:
            entry.update(_build_network_summary(DATA_DIR / gexf_name, spec))

        synced_assets[asset_key] = entry

    return synced_assets


def format_timestamp_label(value: float | datetime | None) -> str:
    if value is None:
        return "–"
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromtimestamp(value)
    return dt.strftime("%d.%m.%Y, %H:%M")


def current_timestamp_label() -> str:
    return format_timestamp_label(datetime.now())


def data_updated_at_label() -> str:
    timestamps: list[float] = []
    for path in (SHOW_FILE, GUEST_FILE):
        if path.exists():
            timestamps.append(path.stat().st_mtime)
    if not timestamps:
        return "–"
    return format_timestamp_label(max(timestamps))


def load_dashboard_bundle() -> DashboardBundle:
    if not SHOW_FILE.exists() or not GUEST_FILE.exists():
        return DashboardBundle(
            shows=pd.DataFrame(),
            guests=pd.DataFrame(),
            guest_metadata=pd.DataFrame(),
            show_options=[ALL_SHOWS],
            category_options=[ALL_CATEGORIES],
            error=(
                "Auswertungsdateien fehlen. Bitte zuerst "
                "'python analyze_talkshows.py' ausfuehren."
            ),
        )

    df_shows = pd.read_excel(SHOW_FILE)
    df_guests = pd.read_excel(GUEST_FILE)

    if "date" in df_shows.columns:
        df_shows["date"] = coerce_mixed_date_series(df_shows["date"])
        df_shows = filter_to_analysis_window(df_shows)

    if "show" in df_shows.columns:
        df_shows["show_display"] = df_shows["show"].apply(display_show_name)
    else:
        df_shows["show_display"] = "Unbekannte Sendung"

    show_info_cols = [
        column
        for column in ["uid", "date", "show_display", "station", "title"]
        if column in df_shows.columns
    ]
    if show_info_cols:
        df_guests = df_guests.merge(df_shows[show_info_cols], on="uid", how="left")

    if "date" in df_guests.columns:
        df_guests["date"] = coerce_mixed_date_series(df_guests["date"])
        df_guests = filter_to_analysis_window(df_guests)

    if "show_display" not in df_guests.columns:
        df_guests["show_display"] = "Unbekannte Sendung"
    if "CategoryPrimary" in df_guests.columns:
        df_guests["category_display"] = df_guests["CategoryPrimary"].apply(display_category_name)
    else:
        df_guests["category_display"] = "Nicht zugeordnet"

    guest_metadata = prepare_guest_metadata(df_guests)

    show_options = [ALL_SHOWS] + sorted(
        show for show in df_shows["show_display"].dropna().unique().tolist() if show
    )
    category_options = [ALL_CATEGORIES] + sorted(
        category
        for category in df_guests["category_display"].dropna().unique().tolist()
        if category
    )

    return DashboardBundle(
        shows=df_shows,
        guests=df_guests,
        guest_metadata=guest_metadata,
        show_options=show_options,
        category_options=category_options,
    )


def filter_shows(
    shows: pd.DataFrame,
    timeframe_label: str,
    show_label: str = ALL_SHOWS,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    filtered, start_date, end_date = filter_by_timeframe(shows, timeframe_label)
    if show_label != ALL_SHOWS and "show_display" in filtered.columns:
        filtered = filtered[filtered["show_display"] == show_label].copy()
    return filtered, start_date, end_date


def filter_guests(
    guests: pd.DataFrame,
    timeframe_label: str,
    show_label: str = ALL_SHOWS,
    category_label: str = ALL_CATEGORIES,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    filtered, start_date, end_date = filter_by_timeframe(guests, timeframe_label)
    if show_label != ALL_SHOWS and "show_display" in filtered.columns:
        filtered = filtered[filtered["show_display"] == show_label].copy()
    if category_label != ALL_CATEGORIES and "category_display" in filtered.columns:
        filtered = filtered[filtered["category_display"] == category_label].copy()
    return filtered, start_date, end_date


def build_timeline_rows(df_shows: pd.DataFrame) -> list[dict[str, int | str]]:
    if df_shows.empty or "date" not in df_shows.columns:
        return []

    source = df_shows.dropna(subset=["date"]).copy()
    if source.empty:
        return []

    if "uid" in source.columns:
        source = source.drop_duplicates(subset=["uid"])

    start_date = source["date"].min()
    end_date = source["date"].max()
    span_days = int((end_date - start_date).days) if pd.notna(start_date) and pd.notna(end_date) else 0

    if span_days > 730:
        source["period"] = source["date"].dt.year.astype(str)
    else:
        source["period"] = source["date"].dt.to_period("M").astype(str)

    timeline = (
        source.groupby("period")["uid"]
        .nunique()
        .reset_index(name="episodes")
        .sort_values("period")
    )
    return timeline.to_dict(orient="records")


def build_show_mix_rows(df_shows: pd.DataFrame) -> list[dict[str, int | str]]:
    if df_shows.empty or "show_display" not in df_shows.columns:
        return []

    grouped = (
        df_shows.groupby("show_display")["uid"]
        .nunique()
        .reset_index(name="episodes")
        .sort_values("episodes", ascending=False)
    )
    grouped["show_short"] = grouped["show_display"].apply(lambda value: compact_label(value, 18))
    return grouped.to_dict(orient="records")


def build_topic_rows(df_shows: pd.DataFrame, limit: int = 8) -> list[dict[str, int | str]]:
    topic_counts = summarize_topic_counts(df_shows)
    if topic_counts.empty:
        return []

    topic_counts = topic_counts[topic_counts["Thema"] != "Nicht klassifiziert"].copy()
    if topic_counts.empty:
        return []

    topic_counts = topic_counts.head(limit)
    topic_counts["topic_short"] = topic_counts["Thema"].apply(lambda value: compact_label(value, 36))
    topic_counts.rename(columns={"Thema": "topic", "Episoden": "episodes"}, inplace=True)
    return topic_counts[["topic", "topic_short", "episodes"]].to_dict(orient="records")


def build_category_rows(df_guests: pd.DataFrame, limit: int = 8) -> list[dict[str, int | str]]:
    if df_guests.empty or "category_display" not in df_guests.columns or "name" not in df_guests.columns:
        return []

    source = df_guests.dropna(subset=["name"]).drop_duplicates(subset=["name"]).copy()
    if source.empty:
        return []

    grouped = (
        source.groupby("category_display")
        .size()
        .reset_index(name="guests")
        .sort_values("guests", ascending=False)
        .head(limit)
    )
    grouped["category_short"] = grouped["category_display"].apply(lambda value: compact_label(value, 22))
    return grouped.to_dict(orient="records")


def build_appearance_rows(
    df_guests: pd.DataFrame,
    guest_metadata: pd.DataFrame,
    limit: int = 12,
) -> list[list[str]]:
    if df_guests.empty or "name" not in df_guests.columns:
        return []

    appearances = (
        df_guests.groupby("name")
        .size()
        .reset_index(name="appearances")
        .sort_values("appearances", ascending=False)
    )
    appearances = appearances.merge(guest_metadata, on="name", how="left")

    rows: list[list[str]] = []
    for row in appearances.head(limit).itertuples(index=False):
        rows.append(
            [
                str(row.name),
                compact_label(getattr(row, "primary_party", None) or "–", 18),
                compact_label(getattr(row, "known_roles", None) or "–", 42),
                format_count(getattr(row, "appearances", 0)),
            ]
        )
    return rows


def build_topic_diversity_rows(
    df_guests: pd.DataFrame,
    guest_metadata: pd.DataFrame,
    limit: int = 12,
) -> list[list[str]]:
    if df_guests.empty or "name" not in df_guests.columns:
        return []

    source = df_guests.dropna(subset=["name"]).copy()
    if "topic" in source.columns:
        source = source[source["topic"] != -1]
    if "topic_label" not in source.columns:
        return []
    source = source[source["topic_label"].notna()]
    if source.empty:
        return []

    diversity = (
        source.groupby("name")["topic_label"]
        .nunique()
        .reset_index(name="unique_topics")
        .sort_values("unique_topics", ascending=False)
    )
    diversity = diversity.merge(guest_metadata, on="name", how="left")

    rows: list[list[str]] = []
    for row in diversity.head(limit).itertuples(index=False):
        rows.append(
            [
                str(row.name),
                compact_label(getattr(row, "primary_party", None) or "–", 18),
                compact_label(getattr(row, "known_roles", None) or "–", 42),
                format_count(getattr(row, "unique_topics", 0)),
            ]
        )
    return rows


def build_show_cards(df_shows: pd.DataFrame) -> list[list[str]]:
    if df_shows.empty or "show_display" not in df_shows.columns:
        return []

    grouped = (
        df_shows.groupby("show_display")
        .agg(
            episodes=("uid", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
        .sort_values("episodes", ascending=False)
    )

    cards: list[list[str]] = []
    for row in grouped.itertuples(index=False):
        cards.append(
            [
                str(row.show_display),
                format_count(getattr(row, "episodes", 0)),
                format_date_range(getattr(row, "first_date", None), getattr(row, "last_date", None)),
            ]
        )
    return cards


def build_top_topic_label(df_shows: pd.DataFrame) -> str:
    topic_rows = build_topic_rows(df_shows, limit=1)
    if not topic_rows:
        return "Kein Thema"
    return str(topic_rows[0]["topic"])


def build_top_guest_label(df_guests: pd.DataFrame) -> str:
    if df_guests.empty or "name" not in df_guests.columns:
        return "Kein Gast"
    appearances = (
        df_guests.groupby("name")
        .size()
        .reset_index(name="appearances")
        .sort_values("appearances", ascending=False)
    )
    if appearances.empty:
        return "Kein Gast"
    top_row = appearances.iloc[0]
    return f"{top_row['name']} ({format_count(top_row['appearances'])})"


def build_top_diversity_label(df_guests: pd.DataFrame) -> str:
    if df_guests.empty or "name" not in df_guests.columns or "topic_label" not in df_guests.columns:
        return "Keine Themenvielfalt"
    source = df_guests.dropna(subset=["name", "topic_label"]).copy()
    if "topic" in source.columns:
        source = source[source["topic"] != -1]
    if source.empty:
        return "Keine Themenvielfalt"
    diversity = (
        source.groupby("name")["topic_label"]
        .nunique()
        .reset_index(name="unique_topics")
        .sort_values("unique_topics", ascending=False)
    )
    if diversity.empty:
        return "Keine Themenvielfalt"
    top_row = diversity.iloc[0]
    return f"{top_row['name']} ({format_count(top_row['unique_topics'])})"


def build_episode_count(df_shows: pd.DataFrame) -> str:
    if df_shows.empty:
        return "0"
    if "uid" in df_shows.columns:
        return format_count(df_shows["uid"].nunique())
    return format_count(len(df_shows))


def build_unique_guest_count(df_guests: pd.DataFrame) -> str:
    if df_guests.empty or "name" not in df_guests.columns:
        return "0"
    return format_count(df_guests["name"].dropna().nunique())


def format_topic_text(value: object) -> str:
    return compact_label(format_topic_label(value), 48)


TIMEFRAME_LABELS = list(TIMEFRAME_OPTIONS.keys())

# ── Treemap ──────────────────────────────────────────────────────────────────

# Thematic groups with their member topic IDs and display colours.
# Each group maps a set of BERTopic integer IDs to a semantic label + colour.
_TOPIC_GROUPS: list[dict[str, object]] = [
    {
        "label": "Außenpolitik & Sicherheit",
        "color": "#2563eb",  # blue
        "topics": {0, 9, 11, 14, 19, 24},
    },
    {
        "label": "Innenpolitik & Wahlen",
        "color": "#7c3aed",  # violet
        "topics": {1, 3, 6, 15, 28, 31},
    },
    {
        "label": "Wirtschaft & Klima",
        "color": "#0e7490",  # cyan/teal
        "topics": {4, 12, 16, 18, 21, 25, 30},
    },
    {
        "label": "Gesellschaft & Migration",
        "color": "#059669",  # emerald
        "topics": {7, 8, 10, 17, 27, 29},
    },
    {
        "label": "Gesundheit",
        "color": "#d97706",  # amber
        "topics": {2, 5, 13},
    },
    {
        "label": "Kultur & Sport",
        "color": "#e11d48",  # rose
        "topics": {20, 22, 23, 26},
    },
]

# Fallback colour for topics not assigned to any group
_TOPIC_GROUP_FALLBACK = "#94a3b8"  # slate-400


def _topic_group_color(topic_id: int) -> str:
    """Return the thematic group colour for a given topic ID."""
    for group in _TOPIC_GROUPS:
        if topic_id in group["topics"]:  # type: ignore[operator]
            return str(group["color"])
    return _TOPIC_GROUP_FALLBACK


def build_treemap_data(df_shows: pd.DataFrame) -> list[dict[str, object]]:
    """
    Returns a flat list of dicts suitable for ``rx.recharts.treemap``.

    Each item: ``{"name": str, "size": int, "fill": str, "group": str}``
    Topics are coloured by thematic group (not by rank), so the treemap
    gives a visual sense of which topics belong together.
    """
    if df_shows.empty or "topic" not in df_shows.columns:
        return []

    source = df_shows[df_shows["topic"] >= 0].copy()
    if source.empty:
        return []

    # One episode counts once per topic (guard against guest-level explosion)
    if "uid" in source.columns:
        source = source.drop_duplicates(subset=["uid", "topic"])

    label_map: dict = {}
    if "topic_label" in source.columns:
        label_map = (
            source[["topic", "topic_label"]]
            .drop_duplicates()
            .set_index("topic")["topic_label"]
            .to_dict()
        )

    counts = (
        source.groupby("topic")
        .size()
        .reset_index(name="size")
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )

    rows: list[dict[str, object]] = []
    for _, row in counts.iterrows():
        topic_id = int(row["topic"])
        raw_label = label_map.get(topic_id, f"Topic {topic_id}")
        display = compact_label(format_topic_label(raw_label), 26)
        color = _topic_group_color(topic_id)
        # Determine group name for tooltip enrichment
        group_name = next(
            (str(g["label"]) for g in _TOPIC_GROUPS if topic_id in g["topics"]),  # type: ignore[operator]
            "Sonstige",
        )
        rows.append(
            {
                "name": display,
                "size": int(row["size"]),
                "fill": color,
                "group": group_name,
            }
        )

    return rows


def build_treemap_legend() -> list[dict[str, str]]:
    """
    Returns one entry per thematic group for rendering a colour legend.
    Each item: ``{"color": str, "label": str}``
    """
    return [
        {"color": str(g["color"]), "label": str(g["label"])}
        for g in _TOPIC_GROUPS
    ]


# ── Top guest pairs by co-appearance intensity (filter-reactive) ─────────────

from itertools import combinations as _combinations


def build_top_guest_pairs(
    df_guests: pd.DataFrame,
    n: int = 15,
) -> list[dict[str, str | int]]:
    """
    Returns the top N guest *pairs* ranked by how often they appeared in the
    same episode — i.e. the strongest edges in the co-occurrence network.

    Each item: ``{"pair": str, "episodes": int}``
    Sorted ascending so the largest bar renders at the top of a horizontal
    bar chart.
    """
    if df_guests.empty:
        return []
    needed = {"name", "uid"}
    if not needed.issubset(df_guests.columns):
        return []

    # One row per (uid, name) — guard against duplicate guest rows per episode
    src = df_guests[["uid", "name"]].drop_duplicates()

    # Group names by episode
    ep_guests = src.groupby("uid")["name"].apply(list)

    # Count co-occurrences for every pair within each episode
    pair_counts: dict[tuple[str, str], int] = {}
    for guests in ep_guests:
        unique = list(dict.fromkeys(guests))  # stable dedup preserving order
        for a, b in _combinations(sorted(unique), 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return []

    top = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    # Ascending so the bar chart renders strongest pair at the top
    return [
        {"pair": f"{a} & {b}", "episodes": count}
        for (a, b), count in reversed(top)
    ]


# ── Topics over time (custom chart) ─────────────────────────────────────────

_N_CHART_TOPICS = 6


def build_topics_over_time_data(
    df_shows: pd.DataFrame,
    n_topics: int = _N_CHART_TOPICS,
) -> tuple[list[dict[str, object]], list[str]]:
    """
    Build a wide-format dataset for a multi-line Recharts chart.

    Returns
    -------
    rows : list[dict]
        Each dict has keys "period", "t0", "t1", … "t{n-1}".
    labels : list[str]
        Exactly *n_topics* human-readable label strings.
        Unused slots are empty strings.
    """
    empty = ([], [""] * n_topics)

    if df_shows.empty or "topic_label" not in df_shows.columns or "date" not in df_shows.columns:
        return empty

    source = df_shows.dropna(subset=["date", "topic_label"]).copy()
    if "topic" in source.columns:
        source = source[source["topic"] != -1]
    if source.empty:
        return empty

    # Top N topics by total episode count
    top_labels = (
        source.groupby("topic_label")
        .size()
        .nlargest(n_topics)
        .index.tolist()
    )
    if not top_labels:
        return empty

    # Time granularity: year if > 2 years, else month
    valid_dates = source["date"].dropna()
    if valid_dates.empty:
        return empty
    span_days = int((valid_dates.max() - valid_dates.min()).days)

    if span_days > 730:
        source["period"] = source["date"].dt.year.astype(str)
    else:
        source["period"] = source["date"].dt.to_period("M").astype(str)

    # De-duplicate at uid level if available (count episodes, not guests)
    if "uid" in source.columns:
        agg = source.drop_duplicates(subset=["uid", "topic_label"])
    else:
        agg = source

    filtered = agg[agg["topic_label"].isin(top_labels)]

    pivot = (
        filtered.groupby(["period", "topic_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Rename to t0 … t{n-1} so column names are always predictable
    col_map = {raw: f"t{i}" for i, raw in enumerate(top_labels)}
    pivot.rename(columns=col_map, inplace=True)

    for i in range(n_topics):
        key = f"t{i}"
        if key not in pivot.columns:
            pivot[key] = 0

    pivot = pivot.sort_values("period")
    keep = ["period"] + [f"t{i}" for i in range(n_topics)]
    rows = pivot[keep].to_dict(orient="records")

    # Human-readable labels, padded to exactly n_topics
    display = [compact_label(format_topic_label(raw), 32) for raw in top_labels]
    padded = (display + [""] * n_topics)[:n_topics]

    return rows, padded
