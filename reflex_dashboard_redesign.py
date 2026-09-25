"""
Data builders for the redesigned Reflex dashboard.

Sits next to reflex_dashboard_data.py and only adds functions; nothing in the
existing module has to be removed. Every builder returns plain JSON-friendly
structures (str/int values) so Reflex can type the computed vars.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from app_helpers import format_topic_label, is_format_cluster_label
from reflex_dashboard_data import build_top_guest_pairs, compact_label, format_count
from topic_labels import FORMAT_CLUSTER_PREFIX, label_text, load_curated_labels

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Same order/colours as CHART_COLORS in wtmw_reflex.py, fixed per show so a
# show keeps its colour on every page and under every filter.
SHOW_COLORS: dict[str, str] = {
    "Maybrit Illner": "#0e7490",
    "Maischberger": "#2563eb",
    "Markus Lanz": "#d97706",
    "Hart aber Fair": "#7c3aed",
    "Anne Will": "#059669",
    "Caren Miosga": "#e11d48",
}
FALLBACK_COLOR = "#64748b"

CATEGORY_COLORS: list[str] = [
    "#2563eb", "#7c3aed", "#0e7490", "#059669", "#d97706", "#e11d48",
    "#475569", "#94a3b8", "#64748b", "#cbd5e1", "#e2e8f0",
]

# Years whose scrape is known to be incomplete (None = only the latest year in the data).
# 2025 has no Markus Lanz episodes Jan–Jul and no Maischberger episodes Jan–Aug, and
# holds ~40 % of the 2024 volume — a scraping gap, not fewer broadcasts.
INCOMPLETE_YEARS: set[int] | None = {2025, 2026}

UNCLASSIFIED_LABEL = "Nicht klassifiziert"

# Short party names for narrow table columns (anything else is passed through)
PARTY_SHORT: dict[str, str] = {
    "Bündnis 90/Die Grünen": "Grüne",
    "DIE LINKE": "Linke",
}


# ── helpers ──────────────────────────────────────────────────────────────────


def _short_party(party: str | None) -> str:
    if not party:
        return "–"
    return PARTY_SHORT.get(party, party)


def person_href(name: str) -> str:
    return f"/gaeste/{quote(str(name), safe='')}"


def _years(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["date"], errors="coerce").dt.year


def _incomplete_years(df_shows: pd.DataFrame) -> set[int]:
    if INCOMPLETE_YEARS is not None:
        return set(INCOMPLETE_YEARS)
    years = _years(df_shows).dropna()
    return {int(years.max())} if not years.empty else set()


def _topic_id(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip().split(" ", 1)[0].split("_")[0]
        return int(text) if text.lstrip("-").isdigit() else None


def _topic_display(raw_label: object) -> tuple[str, bool]:
    """(label without the 'Format-Cluster: ' prefix, is_format). The UI renders a badge instead."""
    label = format_topic_label(raw_label)
    if is_format_cluster_label(label):
        return label.removeprefix(FORMAT_CLUSTER_PREFIX), True
    return label, False


def _topic_terms(raw_label: object, n: int = 4) -> str:
    """'0 impfpflicht gesundheitsminister inzidenz covid' -> 'impfpflicht · …'"""
    if raw_label is None or pd.isna(raw_label):
        return ""
    parts = str(raw_label).split()
    if parts and parts[0].lstrip("-").isdigit():
        parts = parts[1:]
    return " · ".join(parts[:n])


def _unique_episodes(df_shows: pd.DataFrame) -> pd.DataFrame:
    if "uid" in df_shows.columns:
        return df_shows.drop_duplicates(subset=["uid"])
    return df_shows


def _lighten(hex_color: str, amount: float = 0.6) -> str:
    """Mix a #rrggbb colour with white; used for incomplete years (recharts ignores fill_opacity here)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(*(round(c + (255 - c) * amount) for c in (r, g, b)))


def _spark_path(values: list[int], width: int = 108, height: int = 28) -> str:
    if not values:
        return ""
    peak = max(max(values), 1)
    step = width / max(len(values) - 1, 1)
    points = [
        f"{i * step:.1f},{height - 2 - (v / peak) * (height - 4):.1f}"
        for i, v in enumerate(values)
    ]
    return "M" + " L".join(points)


# ── Übersicht ────────────────────────────────────────────────────────────────


def build_stacked_timeline(df_shows: pd.DataFrame) -> tuple[list[dict[str, int | str]], list[dict[str, str]]]:
    """
    Episodes per year, one key per show. Incomplete years use '<show>__p' keys
    so the chart can draw them with lower opacity in the same stack.
    Returns (rows, series) where series = [{key, show, color, fill, partial}]:
    ``color`` is the show colour (legend), ``fill`` the bar colour (lightened when partial).
    """
    if df_shows.empty or "date" not in df_shows.columns:
        return [], []
    src = _unique_episodes(df_shows.dropna(subset=["date"])).copy()
    src["year"] = _years(src)
    src = src.dropna(subset=["year"])
    if src.empty:
        return [], []
    partial = _incomplete_years(src)
    pivot = src.groupby(["year", "show_display"]).size().unstack(fill_value=0).sort_index()
    shows = sorted(pivot.columns, key=lambda s: -int(pivot[s].sum()))

    rows: list[dict[str, int | str]] = []
    for year, counts in pivot.iterrows():
        row: dict[str, int | str] = {"period": str(int(year))}
        suffix = "__p" if int(year) in partial else ""
        for show in shows:
            row[f"{show}{suffix}"] = int(counts[show])
        rows.append(row)

    series: list[dict[str, str]] = []
    for show in shows:
        color = SHOW_COLORS.get(show, FALLBACK_COLOR)
        series.append({"key": show, "show": show, "color": color, "fill": color, "partial": "0"})
        if partial:
            series.append({"key": f"{show}__p", "show": show, "color": color, "fill": _lighten(color), "partial": "1"})
    return rows, series


def build_coverage_rows(df_shows: pd.DataFrame) -> list[dict[str, str]]:
    """Per show: count + date range positioned on the shared time axis (percent)."""
    if df_shows.empty or "show_display" not in df_shows.columns:
        return []
    src = _unique_episodes(df_shows.dropna(subset=["date"]))
    if src.empty:
        return []
    t0, t1 = src["date"].min(), src["date"].max()
    span = max((t1 - t0).total_seconds(), 1)
    grouped = (
        src.groupby("show_display")
        .agg(episodes=("date", "size"), first=("date", "min"), last=("date", "max"))
        .sort_values("episodes", ascending=False)
        .reset_index()
    )
    rows: list[dict[str, str]] = []
    for r in grouped.itertuples(index=False):
        left = (r.first - t0).total_seconds() / span * 100
        width = max((r.last - r.first).total_seconds() / span * 100, 0.8)
        rows.append({
            "show": str(r.show_display),
            "color": SHOW_COLORS.get(str(r.show_display), FALLBACK_COLOR),
            "episodes": format_count(r.episodes),
            "range": f"{r.first:%d.%m.%Y} – {r.last:%d.%m.%Y}",
            "left": f"{left:.2f}%",
            "width": f"{width:.2f}%",
        })
    return rows


def build_axis_labels(df_shows: pd.DataFrame) -> list[str]:
    years = _years(df_shows).dropna() if not df_shows.empty and "date" in df_shows.columns else pd.Series(dtype=float)
    if years.empty:
        return []
    lo, hi = int(years.min()), int(years.max())
    return [str(lo), str(round((lo + hi) / 2)), str(hi)]


def classified_share(df_shows: pd.DataFrame) -> tuple[str, str]:
    """('84 %', '342 Episoden ohne Thema')"""
    src = _unique_episodes(df_shows)
    if src.empty or "topic" not in src.columns:
        return "–", ""
    ids = src["topic"].apply(_topic_id)
    missing = int((ids.isna() | (ids == -1)).sum())
    share = round((1 - missing / len(src)) * 100)
    return f"{share} %", f"{format_count(missing)} Episoden ohne Thema"


def build_kpis(df_shows: pd.DataFrame, df_guests: pd.DataFrame) -> list[dict[str, str]]:
    episodes = _unique_episodes(df_shows)
    n_eps = len(episodes)
    n_guests = int(df_guests["name"].dropna().nunique()) if "name" in df_guests.columns else 0
    per_ep = (len(df_guests) / n_eps) if n_eps else 0
    years = _years(episodes).dropna() if "date" in episodes.columns else pd.Series(dtype=float)
    span = f"{int(years.min())}–{int(years.max())}" if not years.empty else "–"
    n_shows = episodes["show_display"].nunique() if "show_display" in episodes.columns else 0
    share, share_note = classified_share(df_shows)
    return [
        {"label": "Episoden", "value": format_count(n_eps), "note": f"{n_shows} {'Sendung' if n_shows == 1 else 'Sendungen'}"},
        {"label": "Gäste", "value": format_count(n_guests), "note": f"Ø {per_ep:.1f} pro Episode".replace(".", ",")},
        {"label": "Zeitraum", "value": span, "note": "Abdeckung je Sendung unten"},
        {"label": "Mit Thema", "value": share, "note": share_note},
    ]


# ── Gäste ────────────────────────────────────────────────────────────────────


def latest_party_map(df_guests: pd.DataFrame) -> dict[str, str]:
    """Most recent non-empty party per guest (fixes Wagenknecht → BSW)."""
    col = "party_norm" if "party_norm" in df_guests.columns else "party"
    if df_guests.empty or col not in df_guests.columns or "date" not in df_guests.columns:
        return {}
    src = df_guests.dropna(subset=["name", col]).sort_values("date")
    src = src[src[col].astype(str).str.strip() != ""]
    return src.groupby("name")[col].last().astype(str).to_dict()


def build_guest_rank_rows(
    df_guests: pd.DataFrame,
    guest_metadata: pd.DataFrame,
    by: str = "appearances",
    limit: int = 12,
) -> list[dict[str, str]]:
    if df_guests.empty or "name" not in df_guests.columns:
        return []
    src = df_guests.dropna(subset=["name"])
    if by == "topics":
        ids = src["topic"].apply(_topic_id) if "topic" in src.columns else pd.Series(dtype=float)
        src = src[ids.notna() & (ids != -1)]
        values = src.groupby("name")["topic"].nunique()
    else:
        values = src.groupby("name").size()
    values = values.sort_values(ascending=False).head(limit)
    if values.empty:
        return []
    parties = latest_party_map(df_guests)
    meta = guest_metadata.set_index("name") if "name" in guest_metadata.columns else pd.DataFrame()
    cats = (
        df_guests.dropna(subset=["name"]).groupby("name")["category_display"].agg(lambda s: s.mode().iat[0])
        if "category_display" in df_guests.columns else pd.Series(dtype=str)
    )
    peak = int(values.iloc[0]) or 1
    rows: list[dict[str, str]] = []
    for rank, (name, value) in enumerate(values.items(), start=1):
        roles = meta.at[name, "known_roles"] if name in meta.index and "known_roles" in meta.columns else None
        rows.append({
            "rank": str(rank),
            "name": str(name),
            "href": person_href(name),
            "party": compact_label(_short_party(parties.get(name)), 14),
            "role": compact_label(roles if isinstance(roles, str) else "–", 60),
            "category": str(cats.get(name, "–")),
            "value": format_count(value),
            "width": f"{int(value) / peak * 100:.1f}%",
        })
    return rows


def build_pair_rows(df_guests: pd.DataFrame, limit: int = 6) -> list[dict[str, str]]:
    """Strongest co-appearance pairs, most frequent first, with profile links for both guests."""
    rows: list[dict[str, str]] = []
    # build_top_guest_pairs sorts ascending (for horizontal bar charts) and joins names with " & "
    for item in reversed(build_top_guest_pairs(df_guests, n=limit)):
        a, _, b = str(item["pair"]).partition(" & ")
        rows.append({
            "a": a, "b": b,
            "a_href": person_href(a), "b_href": person_href(b),
            "value": format_count(int(item["episodes"])),
        })
    return rows


def build_category_share_rows(df_guests: pd.DataFrame) -> list[dict[str, str]]:
    if df_guests.empty or "category_display" not in df_guests.columns:
        return []
    per_guest = df_guests.dropna(subset=["name"]).groupby("name")["category_display"].agg(lambda s: s.mode().iat[0])
    counts = per_guest.value_counts()
    total = int(counts.sum()) or 1
    return [
        {
            "label": str(label),
            "count": format_count(n),
            "share": f"{round(n / total * 100)} %",
            "width": f"{n / total * 100:.2f}%",
            "color": CATEGORY_COLORS[i % len(CATEGORY_COLORS)],
        }
        for i, (label, n) in enumerate(counts.items())
    ]


# ── Themen ───────────────────────────────────────────────────────────────────


def build_topic_list_rows(df_shows: pd.DataFrame, limit: int = 14) -> list[dict[str, str]]:
    src = _unique_episodes(df_shows)
    if src.empty or "topic" not in src.columns:
        return []
    src = src.assign(tid=src["topic"].apply(_topic_id), year=_years(src))
    src = src[src["tid"].notna() & (src["tid"] != -1)]
    if src.empty:
        return []
    years = list(range(int(src["year"].min()), int(src["year"].max()) + 1))
    counts = src.groupby("tid").size().sort_values(ascending=False).head(limit)
    raw_labels = src.groupby("tid")["topic_label"].first() if "topic_label" in src.columns else pd.Series(dtype=str)
    rows: list[dict[str, str]] = []
    for rank, (tid, n) in enumerate(counts.items(), start=1):
        per_year = src[src["tid"] == tid].groupby("year").size().reindex(years, fill_value=0).tolist()
        label, is_format = _topic_display(raw_labels.get(tid))
        rows.append({
            "rank": str(rank),
            "id": str(int(tid)),
            "label": compact_label(label, 44),
            "episodes": format_count(n),
            "spark": _spark_path([int(v) for v in per_year]),
            "is_format": "1" if is_format else "",
        })
    return rows


def build_topic_detail(df_shows: pd.DataFrame, df_guests: pd.DataFrame, topic_id: str) -> dict[str, object]:
    src = _unique_episodes(df_shows)
    tid = _topic_id(topic_id)
    if src.empty or tid is None or "topic" not in src.columns:
        return {}
    src = src.assign(tid=src["topic"].apply(_topic_id), year=_years(src))
    sel = src[src["tid"] == tid]
    if sel.empty:
        return {}
    years = list(range(int(src["year"].min()), int(src["year"].max()) + 1))
    per_year = sel.groupby("year").size().reindex(years, fill_value=0)
    peak_year = int(per_year.idxmax())
    raw = sel["topic_label"].iloc[0] if "topic_label" in sel.columns else None
    label, is_format = _topic_display(raw)
    guests: list[dict[str, str]] = []
    if not df_guests.empty and "topic" in df_guests.columns:
        g = df_guests[df_guests["topic"].apply(_topic_id) == tid]
        for name, n in g.groupby("name").size().sort_values(ascending=False).head(5).items():
            guests.append({"name": str(name), "href": person_href(name), "value": format_count(n)})
    return {
        "id": str(tid),
        "label": label,
        "terms": _topic_terms(raw),
        "episodes": format_count(len(sel)),
        "share": f"{round(len(sel) / len(src) * 100)} %",
        "peak": f"{peak_year} ({int(per_year.max())})",
        "is_format": is_format,
        "years": [{"period": f"'{str(y)[2:]}", "episodes": int(v)} for y, v in per_year.items()],
        "guests": guests,
    }


def default_topic_id(df_shows: pd.DataFrame) -> str:
    rows = build_topic_list_rows(df_shows, limit=40)
    for row in rows:
        if not row["is_format"]:
            return row["id"]
    return rows[0]["id"] if rows else ""


# ── Personenprofil ───────────────────────────────────────────────────────────


def build_person_profile(df_guests: pd.DataFrame, guest_metadata: pd.DataFrame, name: str) -> dict[str, object]:
    if not name or df_guests.empty or "name" not in df_guests.columns:
        return {}
    mine = df_guests[df_guests["name"] == name].copy()
    if mine.empty:
        return {}
    mine["year"] = _years(mine)
    meta = guest_metadata[guest_metadata["name"] == name]
    roles = meta["known_roles"].iat[0] if not meta.empty and pd.notna(meta["known_roles"].iat[0]) else "–"
    party = _short_party(latest_party_map(mine).get(name))
    category = mine["category_display"].mode().iat[0] if "category_display" in mine.columns else "–"

    all_counts = df_guests.groupby("name").size().sort_values(ascending=False)
    rank = int(list(all_counts.index).index(name)) + 1

    ids = mine["topic"].apply(_topic_id) if "topic" in mine.columns else pd.Series(dtype=float)
    topic_breadth = int(ids[ids.notna() & (ids != -1)].nunique())

    all_years = _years(df_guests).dropna()
    years = list(range(int(all_years.min()), int(all_years.max()) + 1))
    per_year = mine.groupby("year").size().reindex(years, fill_value=0)

    by_show = mine.groupby("show_display").size().sort_values(ascending=False)
    show_peak = int(by_show.iloc[0]) or 1

    mine["topic_display"] = mine["topic_label"].apply(format_topic_label) if "topic_label" in mine.columns else UNCLASSIFIED_LABEL
    by_topic = mine["topic_display"].value_counts().head(5)
    topic_peak = int(by_topic.iloc[0]) or 1

    co = df_guests[df_guests["uid"].isin(mine["uid"]) & (df_guests["name"] != name)]
    co_counts = co.groupby("name").size().sort_values(ascending=False).head(8)

    recent = mine.sort_values("date", ascending=False).head(8)

    return {
        "name": name,
        "party": str(party),
        "category": str(category),
        "roles": str(roles),
        "appearances": format_count(len(mine)),
        "rank_note": f"Rang {rank} von {format_count(len(all_counts))}",
        "topic_breadth": str(topic_breadth),
        "span": f"{int(mine['year'].min())}–{int(mine['year'].max())}",
        "last_seen": f"zuletzt {mine['date'].max():%d.%m.%Y}",
        "top_show": str(by_show.index[0]),
        "top_show_note": f"{int(by_show.iloc[0])} Auftritte",
        "years": [{"period": f"'{str(y)[2:]}", "episodes": int(v)} for y, v in per_year.items()],
        "shows": [
            {"show": str(s), "value": str(int(v)), "width": f"{int(v) / show_peak * 100:.1f}%", "color": SHOW_COLORS.get(str(s), FALLBACK_COLOR)}
            for s, v in by_show.items()
        ],
        "topics": [
            {"label": compact_label(t, 40), "value": str(int(v)), "width": f"{int(v) / topic_peak * 100:.1f}%"}
            for t, v in by_topic.items()
        ],
        "co_guests": [{"name": str(n), "href": person_href(n), "value": str(int(v))} for n, v in co_counts.items()],
        "recent": [
            {
                "date": f"{r.date:%d.%m.%Y}" if pd.notna(r.date) else "–",
                "show": str(r.show_display),
                "topic": compact_label(format_topic_label(getattr(r, "topic_label", None)), 40),
            }
            for r in recent.itertuples(index=False)
        ],
    }


def search_suggestions(df_guests: pd.DataFrame, query: str, limit: int = 8) -> list[dict[str, str]]:
    q = query.strip().lower()
    if not q or df_guests.empty or "name" not in df_guests.columns:
        return []
    names = df_guests["name"].dropna()
    hits = names[names.str.lower().str.contains(q, regex=False)]
    counts = hits.value_counts().head(limit)
    parties = latest_party_map(df_guests)
    return [
        {"name": str(n), "href": person_href(n), "party": _short_party(parties.get(n)), "value": format_count(v)}
        for n, v in counts.items()
    ]


# ── Topic-label validation ───────────────────────────────────────────────────


def topic_label_mismatches() -> list[dict[str, str]]:
    """
    Compare data/topic_labels.json with the saved model's representations.
    An entry is stale when its id is not in the model or, for entries that carry
    keywords, when none of them appear in the topic's top-10 terms. Logged at startup.
    """
    labels_path = DATA_DIR / "topic_labels.json"
    topics_path = DATA_DIR / "talkshow_topic_model" / "topics.json"
    if not labels_path.exists() or not topics_path.exists():
        return []
    curated = load_curated_labels(labels_path)
    reps = json.loads(topics_path.read_text(encoding="utf-8")).get("topic_representations", {})
    out: list[dict[str, str]] = []
    for tid, entry in curated.items():
        label = label_text(entry)
        if tid.startswith("_") or tid == "-1" or not label:
            continue
        if tid not in reps:
            out.append({"id": tid, "label": label, "terms": "(topic id not in model)"})
            continue
        terms = [t[0].lower() for t in reps[tid][:10]]
        keywords = entry.get("keywords") if isinstance(entry, dict) else None
        if keywords:
            stale = not {k.lower() for k in keywords} & set(terms)
        else:
            words = {w.strip("&/()").lower() for w in label.replace("-", " ").split() if len(w) > 3}
            stale = not any(w[:6] in t for w in words for t in terms)
        if stale:
            out.append({"id": tid, "label": label, "terms": " · ".join(terms[:5])})
    return out
