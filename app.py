"""
Streamlit-Dashboard „Wer talkt mit wem?“
=======================================
Gleicher Aufbau wie das Reflex-Dashboard (wtmw_reflex/wtmw_reflex.py):

  Übersicht (/) · Gäste (/gaeste, Profil: /gaeste?person=<Name>)
  Themen (/themen?thema=<ID>) · Netzwerke (/netzwerke)

Die Seiten nutzen dieselben Daten-Builder wie Reflex (wtmw_reflex.page_data,
wtmw_reflex.dashboard_data), damit beide Dashboards dieselben Zahlen zeigen.

Filter (Zeitraum / Sendung / Gastkategorie) stehen im Session State und
werden in die URL gespiegelt: st.navigation leert die Query-Parameter beim
Seitenwechsel, und Links in HTML-Bausteinen (Namen, Themenliste,
Abdeckungszeilen, Kategorie-Chips) laden die Seite neu – beides behält so
die aktive Auswahl.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote, urlencode

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from wtmw_reflex.dashboard_data import (
    ALL_CATEGORIES,
    ALL_SHOWS,
    DEFAULT_TIMEFRAME,
    NETWORK_ASSET_SPECS,
    TIMEFRAME_LABELS,
    data_updated_at_label,
    filter_guests,
    filter_shows,
    format_count,
    load_dashboard_bundle,
)
from wtmw_reflex.page_data import (
    build_axis_labels,
    build_category_share_rows,
    build_coverage_rows,
    build_guest_rank_rows,
    build_kpis,
    build_pair_rows,
    build_person_profile,
    build_stacked_timeline,
    build_topic_detail,
    build_topic_list_rows,
    default_topic_id,
    latest_party_map,
)

st.set_page_config(
    page_title="Wer talkt mit wem?",
    page_icon="🎙️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# URL-Parameter → Standardwert. Standardwerte erscheinen nicht in der URL.
FILTER_DEFAULTS: dict[str, str] = {
    "zeitraum": DEFAULT_TIMEFRAME,
    "sendung": ALL_SHOWS,
    "kategorie": ALL_CATEGORIES,
}
FILTER_LABELS: dict[str, str] = {
    "zeitraum": "Zeitraum",
    "sendung": "Sendung",
    "kategorie": "Gastkategorie",
}

BLUE = "#2563eb"
AXIS_COLOR = "#94a3b8"
GRID_COLOR = "#f1f5f9"


class Filters(NamedTuple):
    timeframe: str
    show: str
    category: str


# ---------------------------------------------------------------------------
# Styles (Tokens und Klassen aus assets/reflex_dashboard*.css)
# ---------------------------------------------------------------------------

CSS = """
<style>
:root{
  --blue:#2563eb;--blue-light:#eff6ff;--blue-border:#bfdbfe;--blue-dark:#1d4ed8;--slate:#475569;
  --bg-page:#f8fafc;--bg-surface:#fff;--bg-raised:#f1f5f9;--border:#e2e8f0;--border-hi:#cbd5e1;
  --text-hi:#0f172a;--text-mid:#334155;--text-lo:#64748b;--text-mute:#94a3b8;
  --shadow-xs:0 1px 2px rgba(0,0,0,.04);--shadow-sm:0 1px 3px rgba(0,0,0,.06),0 1px 4px rgba(0,0,0,.04);
  --r:12px;--r-sm:8px;--r-xs:6px;
}
[data-testid="stMainBlockContainer"]{max-width:1200px;padding-top:4.5rem;padding-bottom:4rem}
[data-testid="stHeader"]{background:rgba(255,255,255,.94);border-bottom:1px solid var(--border);backdrop-filter:blur(14px)}

/* Karten = st.container(border=True, key="card-…") */
[class*="st-key-card"]{background:var(--bg-surface);border-color:var(--border)!important;border-radius:var(--r)!important;box-shadow:var(--shadow-sm);padding:24px!important}
[class*="st-key-card-tight"]{padding:16px!important}

/* Filterleiste */
.st-key-fbar{background:var(--bg-surface);border-color:var(--border)!important;border-radius:var(--r)!important;box-shadow:var(--shadow-xs);padding:12px 16px!important}
.st-key-fbar [data-testid="stWidgetLabel"] p{font-size:.6875rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-lo)}
.st-key-fbar-active [data-baseweb="select"]>div{background:var(--blue-light);border-color:var(--blue-border)}
.fbar-feedback{font-size:.8125rem;color:var(--text-lo)}
.scope-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;min-height:40px}
.scope-badge{font-size:.6875rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-lo);background:var(--bg-raised);border:1px solid var(--border);border-radius:999px;padding:3px 10px;white-space:nowrap}
.scope-text{font-size:.8125rem;color:var(--text-mid)}

/* Kopfzeilen */
.page-header{display:flex;flex-direction:column;gap:8px;margin:8px 0 4px}
.page-header-eyebrow{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--blue)}
.page-header-title{font-size:2rem;font-weight:800;letter-spacing:-.035em;color:var(--text-hi);line-height:1.1;margin:0}
.page-header-sub{font-size:.9375rem;color:var(--text-lo);line-height:1.65;max-width:600px;margin:0}
.card-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;width:100%}
.panel-title{font-size:.9375rem;font-weight:700;color:var(--text-hi);letter-spacing:-.015em}
.card-hint{font-size:.75rem;color:var(--text-mute)}
a.more-link{font-size:.8125rem;font-weight:500;color:var(--blue);text-decoration:none}
a.more-link:hover{text-decoration:underline}
.body-copy{font-size:.875rem;color:var(--text-mid);line-height:1.55}
.section-copy{font-size:.875rem;color:var(--text-lo);line-height:1.65}
.empty-state{display:flex;align-items:center;justify-content:center;border:1px dashed var(--border);border-radius:var(--r-sm);background:var(--bg-raised);font-size:.875rem;color:var(--text-mute);text-align:center;padding:16px}

/* KPI-Leiste */
.kpi-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow-sm)}
.kpi-cell{display:flex;flex-direction:column;gap:6px;padding:20px 24px;border-left:1px solid var(--bg-raised)}
.kpi-cell:first-child{border-left:0}
.stat-title{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text-lo)}
.kpi-value{font-size:1.75rem;font-weight:700;letter-spacing:-.03em;line-height:1.1;font-variant-numeric:tabular-nums;color:var(--text-hi)}
.kpi-note{font-size:.75rem;color:var(--text-lo)}

/* Legende, Fußnoten */
.legend{display:flex;gap:14px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:6px;font-size:.75rem;color:var(--slate)}
.swatch{width:10px;height:10px;border-radius:3px;flex-shrink:0;display:inline-block}
.swatch-sm{width:8px;height:8px;border-radius:2px}
.swatch-partial{background:var(--blue);opacity:.3}
.chart-footnote{display:flex;gap:24px;flex-wrap:wrap;font-size:.78rem;color:var(--slate);line-height:1.5;border-top:1px solid var(--bg-raised);padding-top:12px}
.chart-footnote span{display:flex;align-items:center;gap:6px}

/* Abdeckung je Sendung */
a.cov-row,.cov-row{display:grid;grid-template-columns:130px minmax(0,1fr) 44px;gap:12px;align-items:center;padding:7px 8px;margin:0 -8px;border-radius:var(--r-xs);text-decoration:none;color:inherit}
a.cov-row:hover{background:var(--bg-page)}
.cov-row-active{background:var(--blue-light)}
.cov-name{display:flex;align-items:center;gap:8px;font-size:.84rem;font-weight:600;color:var(--text-hi)}
.cov-track{position:relative;height:8px;background:var(--bg-raised);border-radius:4px}
.cov-range{position:absolute;top:0;bottom:0;border-radius:4px;opacity:.85}
.cov-count{font-size:.84rem;font-weight:700;text-align:right;font-variant-numeric:tabular-nums;color:var(--text-hi)}
.cov-axis{display:flex;justify-content:space-between;font-size:.6875rem;color:var(--text-mute)}

/* Listen und Tabellen */
.rows{display:flex;flex-direction:column}
.table-name{font-size:.875rem;font-weight:600;color:var(--text-hi)}
a.table-name{text-decoration:none}
a.table-name:hover{color:var(--blue)}
.table-cell{font-size:.8125rem;color:var(--text-lo)}
.table-value{font-size:.875rem;font-weight:700;color:var(--blue);font-variant-numeric:tabular-nums;letter-spacing:-.01em;text-align:right}
.table-value-dark{color:var(--text-hi)}
.rank{font-size:.75rem;color:var(--text-mute);font-variant-numeric:tabular-nums}
.row-meta{font-size:.75rem;color:var(--text-lo)}
.truncate{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tabular{font-variant-numeric:tabular-nums}
.bar-track{flex:1;background:var(--bg-raised);border-radius:3px;overflow:hidden;min-width:40px}
.bar-fill{height:100%;border-radius:3px}
.bar-with-value{display:grid;grid-template-columns:minmax(0,1fr) 32px;gap:10px;align-items:center}
a.guest-row-compact{display:grid;grid-template-columns:20px minmax(0,1.3fr) minmax(0,1fr) 36px;gap:12px;align-items:center;padding:10px 0;border-top:1px solid var(--bg-raised);text-decoration:none;color:inherit}
a.guest-row-compact:hover .table-name{color:var(--blue)}
.list-row{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:8px 0;border-top:1px solid var(--bg-raised);text-decoration:none;color:inherit}
a.list-row:hover .table-name{color:var(--blue)}
.pair-names{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.pair-plus{color:var(--border-hi)}
.subhead{font-size:.8125rem;font-weight:700;color:var(--text-mid);padding-bottom:6px}

/* Gäste: Kategorien und Rangliste */
.cat-bar{display:flex;height:14px;border-radius:7px;overflow:hidden;gap:2px;margin:4px 0 16px}
.cat-seg{height:100%}
.cat-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
a.cat-chip{display:grid;grid-template-columns:10px minmax(0,1fr) auto;gap:8px;align-items:center;padding:8px 10px;border:1px solid var(--border);border-radius:var(--r-sm);background:var(--bg-surface);text-decoration:none;color:inherit}
a.cat-chip:hover{border-color:var(--blue-border)}
a.cat-chip-active{background:var(--blue-light);border-color:var(--blue-border)}
.cat-label{font-size:.8125rem;font-weight:500;color:var(--text-mid);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cat-count{font-size:.78rem;color:var(--text-lo);font-variant-numeric:tabular-nums}
.rank-row{display:grid;grid-template-columns:28px minmax(0,1.2fr) 64px minmax(0,1.6fr) 170px 180px;gap:16px;align-items:center;padding:9px 8px;margin:0 -8px;border-radius:var(--r-xs);border-bottom:1px solid var(--bg-raised);text-decoration:none;color:inherit}
a.rank-row:hover{background:var(--bg-page)}
.rank-head{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-lo);border-bottom-color:var(--border)}

/* Themen */
a.topic-row,.topic-row{display:grid;grid-template-columns:24px minmax(0,1fr) 108px 44px;gap:14px;align-items:center;padding:8px 10px;border-radius:var(--r-sm);text-decoration:none;color:inherit}
a.topic-row:hover{background:var(--bg-page)}
a.topic-row-active,a.topic-row-active:hover{background:var(--blue-light)}
.topic-head{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-lo)}
.topic-label{display:flex;align-items:center;gap:8px;min-width:0}
.spark{display:block}
.topic-name-active{color:var(--blue-dark)}
.topic-footnote{font-size:.8125rem;color:var(--text-lo);padding:10px;margin-top:6px;border-top:1px solid var(--bg-raised)}
.badge-neutral{font-size:.75rem;font-weight:600;color:var(--text-lo);background:var(--bg-raised);border:1px solid var(--border);border-radius:999px;padding:2px 10px;white-space:nowrap}
.badge-xs{font-size:.6875rem;padding:1px 8px}
.badge-party{font-size:.75rem;font-weight:600;color:var(--blue);background:#fff;border:1px solid var(--blue-border);border-radius:20px;padding:2px 10px;white-space:nowrap}
.detail-head{display:flex;flex-direction:column;gap:8px}
.detail-title-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.detail-title{font-size:1.375rem;font-weight:800;letter-spacing:-.025em;color:var(--text-hi);margin:0}
.mono-note{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.75rem;color:var(--text-lo)}
.mini-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border:1px solid var(--bg-raised);border-radius:var(--r-sm);margin-top:12px}
.mini-stat-cell{display:flex;flex-direction:column;gap:4px;padding:12px 14px;border-left:1px solid var(--bg-raised)}
.mini-stat-cell:first-child{border-left:0}
.mini-stat{font-size:1.25rem;font-weight:700;letter-spacing:-.02em;color:var(--text-hi)}

/* Personenprofil */
.crumbs{display:flex;gap:6px;font-size:.8125rem;color:var(--text-mute)}
.crumbs a{color:var(--blue);text-decoration:none}
.profile-title-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.labelled-bar{display:grid;grid-template-columns:120px minmax(0,1fr) 28px;gap:12px;align-items:center;padding:6px 0}
.recent-row{display:grid;grid-template-columns:78px minmax(0,1fr);gap:10px;padding:7px 0;border-top:1px solid var(--bg-raised)}
.recent-main{display:flex;flex-direction:column;min-width:0}

/* Netzwerke */
.lazy-head{display:flex;flex-direction:column;gap:2px}

/* Fußzeile */
.page-footer{display:flex;justify-content:space-between;flex-wrap:wrap;gap:16px;border-top:1px solid var(--border);padding-top:16px;margin-top:16px}
.footer-note{font-size:.78rem;color:var(--text-mute)}

@media (max-width:900px){
  .kpi-strip{grid-template-columns:repeat(2,minmax(0,1fr))}
  .cat-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .rank-row{grid-template-columns:24px minmax(0,1fr) 56px 120px}
  .rank-row > :nth-child(4),.rank-row > :nth-child(5){display:none}
}
</style>
"""


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------


@st.cache_resource(ttl=3600, show_spinner="Lade Daten …")
def get_bundle():
    """Geteilt über alle Sessions; die DataFrames werden nur gelesen."""
    return load_dashboard_bundle()


def _shows(timeframe: str, show: str) -> pd.DataFrame:
    return filter_shows(get_bundle().shows, timeframe, show)[0]


def _guests(timeframe: str, show: str, category: str = ALL_CATEGORIES) -> pd.DataFrame:
    return filter_guests(get_bundle().guests, timeframe, show, category)[0]


@st.cache_data(ttl=3600, show_spinner=False)
def filter_feedback(timeframe: str, show: str, category: str) -> str:
    fs = _shows(timeframe, show)
    fg = _guests(timeframe, show, category)
    n_eps = fs["uid"].nunique() if "uid" in fs.columns else len(fs)
    n_guests = fg["name"].dropna().nunique() if "name" in fg.columns else 0
    return f"{format_count(n_eps)} Episoden · {format_count(n_guests)} Gäste"


@st.cache_data(ttl=3600, show_spinner=False)
def overview_data(timeframe: str, show: str, category: str) -> dict:
    bundle = get_bundle()
    fs = _shows(timeframe, show)
    fg = _guests(timeframe, show, category)
    rows, series = build_stacked_timeline(fs)
    # Abdeckung ignoriert den Sendungsfilter, damit jede Sendung anklickbar bleibt.
    fs_all = _shows(timeframe, ALL_SHOWS)
    return {
        "kpis": build_kpis(fs, fg),
        "timeline_rows": rows,
        "timeline_series": series,
        "coverage": build_coverage_rows(fs_all),
        "coverage_axis": build_axis_labels(fs_all),
        "top_guests": build_guest_rank_rows(fg, bundle.guest_metadata, "appearances", 10),
        "top_pairs": build_pair_rows(_guests(timeframe, show), limit=6),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def guest_page_data(timeframe: str, show: str, category: str, sort: str) -> dict:
    fg_all = _guests(timeframe, show)
    return {
        "categories": build_category_share_rows(fg_all),
        "total": fg_all["name"].dropna().nunique() if "name" in fg_all.columns else 0,
        "ranking": build_guest_rank_rows(_guests(timeframe, show, category), get_bundle().guest_metadata, sort, 25),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def profile_data(timeframe: str, show: str, name: str) -> dict:
    return build_person_profile(_guests(timeframe, show), get_bundle().guest_metadata, name) or {}


@st.cache_data(ttl=3600, show_spinner=False)
def topic_page_data(timeframe: str, show: str, topic_id: str | None) -> dict:
    fs = _shows(timeframe, show)
    rows = build_topic_list_rows(fs)
    ids = {r["id"] for r in rows}
    active = topic_id if topic_id in ids else default_topic_id(fs)
    detail = build_topic_detail(fs, _guests(timeframe, show), active) if active else {}

    note = ""
    if not fs.empty and "topic" in fs.columns:
        eps = fs.drop_duplicates(subset=["uid"]) if "uid" in fs.columns else fs
        outliers = int((eps["topic"] == -1).sum())
        missing = int(eps["topic"].isna().sum())
        note = (
            f"Ohne Thema: {format_count(outliers + missing)} Episoden "
            f"({format_count(outliers)} Ausreißer, {format_count(missing)} ohne Beschreibung)"
        )
    return {"rows": rows, "active": active, "detail": detail or {}, "note": note}


@st.cache_data(ttl=3600, show_spinner=False)
def network_pairs() -> list[dict[str, str]]:
    """Ungefiltert: /netzwerke zeigt immer den Gesamtdatensatz."""
    return build_pair_rows(get_bundle().guests, limit=6)


@st.cache_data(ttl=3600, show_spinner=False)
def search_options() -> tuple[list[str], dict[str, str]]:
    """Alle Gäste, häufigste zuerst, mit Anzeigetext 'Name · Partei · n'."""
    guests = get_bundle().guests
    if guests.empty or "name" not in guests.columns:
        return [], {}
    counts = guests["name"].dropna().value_counts()
    parties = latest_party_map(guests)
    labels = {
        name: " · ".join(part for part in (name, parties.get(name, ""), f"{n}×") if part)
        for name, n in counts.items()
    }
    return list(counts.index), labels


@st.cache_data(ttl=3600, show_spinner=False)
def load_html_content(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Filterzustand
# ---------------------------------------------------------------------------


def _filter_options(param: str) -> list[str]:
    bundle = get_bundle()
    return {
        "zeitraum": TIMEFRAME_LABELS,
        "sendung": bundle.show_options,
        "kategorie": bundle.category_options,
    }[param]


def _store_filters(values: dict[str, str]) -> None:
    st.session_state["_filters"] = dict(values)
    for param, default in FILTER_DEFAULTS.items():
        if values[param] == default:
            st.query_params.pop(param, None)
        else:
            st.query_params[param] = values[param]


def current_filters() -> Filters:
    """URL-Parameter gewinnen (Links); sonst gilt der Session-Wert (Seitenwechsel)."""
    saved = st.session_state.get("_filters", {})
    values: dict[str, str] = {}
    for param, default in FILTER_DEFAULTS.items():
        value = st.query_params.get(param, saved.get(param, default))
        values[param] = value if value in _filter_options(param) else default
    _store_filters(values)
    return Filters(values["zeitraum"], values["sendung"], values["kategorie"])


def _on_filter_change(param: str) -> None:
    values = dict(st.session_state["_filters"])
    values[param] = st.session_state[f"fbar_{param}"]
    _store_filters(values)


def _reset_filters() -> None:
    _store_filters(dict(FILTER_DEFAULTS))


def _on_search() -> None:
    st.session_state["_goto_person"] = st.session_state.get("nav_search")
    st.session_state["nav_search"] = None


def href(page: str = "", **params: str | None) -> str:
    """Relativer Link mit den aktiven Filtern. page="" bleibt auf der aktuellen Seite."""
    query = {
        param: value
        for param, value in st.session_state.get("_filters", {}).items()
        if value != FILTER_DEFAULTS[param]
    }
    for key, value in params.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = value
    target = page + ("?" + urlencode(query) if query else "")
    return escape(target or "?", quote=True)


def person_link(name: str) -> str:
    return href("gaeste", person=name, thema=None)


# ---------------------------------------------------------------------------
# Bausteine
# ---------------------------------------------------------------------------


def e(value: object) -> str:
    return escape(str(value), quote=True)


def html(markup: str) -> None:
    st.html(markup)


def card(key: str, tight: bool = False):
    return st.container(border=True, key=f"card-tight-{key}" if tight else f"card-{key}")


def card_head(title: str, right: str = "") -> str:
    return f'<div class="card-head"><span class="panel-title">{e(title)}</span>{right}</div>'


def hint(text: str) -> str:
    return f'<span class="card-hint">{e(text)}</span>'


def more_link(label: str, target: str) -> str:
    return f'<a class="more-link" href="{target}" target="_self">{e(label)}</a>'


def empty_state(message: str, min_h: int = 200) -> str:
    return f'<div class="empty-state" style="min-height:{min_h}px">{e(message)}</div>'


def bar_cell(width: str, color: str = "var(--blue)", height: str = "6px") -> str:
    return (
        f'<div class="bar-track" style="height:{height}">'
        f'<div class="bar-fill" style="width:{e(width)};background:{e(color)}"></div></div>'
    )


def sparkline_src(path: str, stroke: str) -> str:
    """st.html entfernt Inline-SVG; als Bild-Daten-URI bleibt die Sparkline erhalten."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1 -1 110 30" width="108" height="28">'
        f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="1.75" stroke-linejoin="round"/></svg>'
    )
    return "data:image/svg+xml," + quote(svg)


def page_header(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    sub = f'<p class="page-header-sub">{e(subtitle)}</p>' if subtitle else ""
    html(
        f'<div class="page-header"><span class="page-header-eyebrow">{e(eyebrow)}</span>'
        f'<h1 class="page-header-title">{e(title)}</h1>{sub}</div>'
    )


def kpi_strip(items: list[dict[str, str]]) -> None:
    cells = "".join(
        f'<div class="kpi-cell"><span class="stat-title">{e(i["label"])}</span>'
        f'<span class="kpi-value">{e(i["value"])}</span><span class="kpi-note">{e(i["note"])}</span></div>'
        for i in items
    )
    html(f'<div class="kpi-strip">{cells}</div>')


def footer() -> None:
    html(
        '<div class="page-footer">'
        f'<span class="footer-note">Datenstand {e(data_updated_at_label())} · Talkshow-Daten von fernsehserien.de</span>'
        '<span class="footer-note">NLP via BERTopic &amp; sentence-transformers · Bernstein Analytics</span>'
        "</div>"
    )


def person_search() -> None:
    names, labels = search_options()
    st.selectbox(
        "Person suchen",
        names,
        index=None,
        placeholder="Person suchen, z. B. Söder …",
        format_func=lambda name: labels.get(name, name),
        key="nav_search",
        on_change=_on_search,
    )
    target = st.session_state.pop("_goto_person", None)
    if target:
        st.switch_page(PAGES["gaeste"], query_params={"person": target})


def filter_bar(filters: Filters, with_category: bool = True) -> None:
    """Filterleiste wie im Reflex-Dashboard: aktive Filter hervorgehoben, Zurücksetzen bei Bedarf."""
    params = ["zeitraum", "sendung"] + (["kategorie"] if with_category else [])
    with st.container(border=True, key="fbar"):
        cols = st.columns([1] * len(params) + [1.5], vertical_alignment="bottom")
        for col, param in zip(cols, params):
            value = st.session_state["_filters"][param]
            active = value != FILTER_DEFAULTS[param]
            with col, st.container(key=f"fbar-active-{param}" if active else f"fbar-idle-{param}"):
                st.session_state[f"fbar_{param}"] = value
                st.selectbox(
                    FILTER_LABELS[param],
                    _filter_options(param),
                    key=f"fbar_{param}",
                    on_change=_on_filter_change,
                    args=(param,),
                )
        with cols[-1]:
            person_search()

        any_active = any(
            st.session_state["_filters"][p] != FILTER_DEFAULTS[p] for p in params
        )
        with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
            html(f'<span class="fbar-feedback">{e(filter_feedback(*filters))}</span>')
            if any_active:
                st.button("Zurücksetzen", on_click=_reset_filters, type="tertiary")


def scope_bar(text: str) -> None:
    with st.container(border=True, key="fbar"):
        left, right = st.columns([3, 1.5], vertical_alignment="bottom")
        with left:
            html(
                '<div class="scope-row"><span class="scope-badge">Gesamtdatensatz</span>'
                f'<span class="scope-text">{e(text)}</span></div>'
            )
        with right:
            person_search()


# ---------------------------------------------------------------------------
# Diagramme (Vega-Lite, Stil wie die Recharts-Diagramme in Reflex)
# ---------------------------------------------------------------------------

_AXIS = {"title": None, "domain": False, "ticks": False, "labelColor": AXIS_COLOR, "labelFontSize": 12}


def stacked_timeline_chart(rows: list[dict], series: list[dict]) -> None:
    records = []
    order = {s["show"]: i for i, s in enumerate(s for s in series if s["partial"] == "0")}
    for row in rows:
        for s in series:
            if s["key"] in row:
                records.append({
                    "Jahr": row["period"],
                    "Sendung": s["show"],
                    "Episoden": row[s["key"]],
                    "fill": s["fill"],
                    "order": order.get(s["show"], 0),
                })
    spec = {
        "height": 260,
        "mark": {"type": "bar"},
        "encoding": {
            "x": {"field": "Jahr", "type": "ordinal", "scale": {"paddingInner": 0.28},
                  "axis": {**_AXIS, "labelAngle": 0}},
            "y": {"field": "Episoden", "type": "quantitative", "stack": "zero",
                  "axis": {**_AXIS, "grid": True, "gridColor": GRID_COLOR, "tickMinStep": 1}},
            "color": {"field": "fill", "type": "nominal", "scale": None, "legend": None},
            "order": {"field": "order", "type": "quantitative"},
            "tooltip": [
                {"field": "Jahr", "type": "ordinal"},
                {"field": "Sendung", "type": "nominal"},
                {"field": "Episoden", "type": "quantitative"},
            ],
        },
        "background": "white",
        "config": {"view": {"stroke": None}},
    }
    st.vega_lite_chart(pd.DataFrame(records), spec, width="stretch", theme=None)


def year_bar_chart(rows: list[dict], label: str) -> None:
    spec = {
        "height": 180,
        "encoding": {
            "x": {"field": "period", "type": "ordinal", "sort": None, "scale": {"paddingInner": 0.2},
                  "axis": {**_AXIS, "labelAngle": 0}},
            "y": {"field": "episodes", "type": "quantitative", "axis": None},
        },
        "layer": [
            {"mark": {"type": "bar", "color": BLUE, "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
             "encoding": {"tooltip": [{"field": "period", "title": "Jahr"},
                                      {"field": "episodes", "title": label}]}},
            {"mark": {"type": "text", "dy": -7, "fontSize": 11, "color": "#334155"},
             "encoding": {"text": {"condition": {"test": "datum.episodes > 0", "field": "episodes"},
                                   "value": ""}}},
        ],
        "background": "white",
        "config": {"view": {"stroke": None}},
    }
    st.vega_lite_chart(pd.DataFrame(rows), spec, width="stretch", theme=None)


# ---------------------------------------------------------------------------
# Übersicht
# ---------------------------------------------------------------------------


def overview_page() -> None:
    filters = current_filters()
    filter_bar(filters)
    data = overview_data(*filters)

    page_header(
        "Politische Talkshow-Analyse",
        "Wer talkt mit wem?",
        "Daten aus sechs deutschen Talkshows — analysiert nach Gästen, Themen und Mustern. "
        "NLP-gestützt mit BERTopic-Modell.",
    )
    kpi_strip(data["kpis"])

    with card("timeline"):
        legend = "".join(
            f'<span class="legend-item"><span class="swatch" style="background:{e(s["color"])}"></span>{e(s["show"])}</span>'
            for s in data["timeline_series"] if s["partial"] == "0"
        )
        html(card_head("Erfasste Episoden pro Jahr", f'<div class="legend">{legend}</div>'))
        if data["timeline_rows"]:
            stacked_timeline_chart(data["timeline_rows"], data["timeline_series"])
        else:
            html(empty_state("Für diese Auswahl liegen keine Episodendaten vor.", 260))
        html(
            '<div class="chart-footnote"><span><span class="swatch swatch-partial"></span>Unvollständig erfasst</span>'
            "<span>Verläufe können aus der Erfassung stammen: Markus Lanz ist bis 2022 nur mit rund 10 Folgen "
            "pro Jahr enthalten.</span></div>"
        )

    left, right = st.columns(2)
    with left:
        with card("coverage"):
            selected = st.session_state["_filters"]["sendung"]
            rows = "".join(
                f'<a class="cov-row{" cov-row-active" if r["show"] == selected else ""}" target="_self" '
                f'href="{href(sendung=ALL_SHOWS if r["show"] == selected else r["show"])}" title="{e(r["range"])}">'
                f'<span class="cov-name"><span class="swatch swatch-sm" style="background:{e(r["color"])}"></span>{e(r["show"])}</span>'
                f'<span class="cov-track"><span class="cov-range" style="left:{e(r["left"])};width:{e(r["width"])};background:{e(r["color"])}"></span></span>'
                f'<span class="cov-count">{e(r["episodes"])}</span></a>'
                for r in data["coverage"]
            )
            axis = "".join(f"<span>{e(y)}</span>" for y in data["coverage_axis"])
            html(
                card_head("Datenabdeckung je Sendung", hint("Klick filtert"))
                + f'<div class="rows">{rows}<div class="cov-row"><span></span><div class="cov-axis">{axis}</div><span></span></div></div>'
            )
        with card("pairs"):
            if data["top_pairs"]:
                body = "".join(_pair_row(r) for r in data["top_pairs"])
            else:
                body = empty_state("Keine Paarungsdaten für diese Auswahl.", 160)
            html(card_head("Wer sitzt am häufigsten zusammen?", more_link("Netzwerk →", "netzwerke")) + body)
    with right, card("top-guests"):
        rows = "".join(
            f'<a class="guest-row-compact" target="_self" href="{person_link(r["name"])}">'
            f'<span class="rank">{e(r["rank"])}</span>'
            f'<span style="display:flex;flex-direction:column;min-width:0"><span class="table-name truncate">{e(r["name"])}</span>'
            f'<span class="row-meta">{e(r["party"])}</span></span>'
            f'{bar_cell(r["width"])}<span class="table-value">{e(r["value"])}</span></a>'
            for r in data["top_guests"]
        ) or empty_state("Keine Gäste für diese Auswahl.", 160)
        html(card_head("Häufigste Gäste", more_link("Alle Gäste →", href("gaeste"))) + rows)

    footer()


def _pair_row(r: dict[str, str]) -> str:
    return (
        '<div class="list-row"><span class="pair-names">'
        f'<a class="table-name" target="_self" href="{person_link(r["a"])}">{e(r["a"])}</a>'
        '<span class="pair-plus">+</span>'
        f'<a class="table-name" target="_self" href="{person_link(r["b"])}">{e(r["b"])}</a>'
        f'</span><span class="table-value">{e(r["value"])}×</span></div>'
    )


# ---------------------------------------------------------------------------
# Gäste
# ---------------------------------------------------------------------------


def gaeste_page() -> None:
    filters = current_filters()
    person = st.query_params.get("person")
    if person:
        person_page(filters, person)
        return

    filter_bar(filters)
    page_header(
        "Gäste-Analyse",
        "Gäste",
        "Wer tritt am häufigsten auf? Welche Kategorien dominieren? "
        "Filter nach Zeitraum, Sendung und Gastsegment.",
    )

    sort = st.session_state.get("guest_sort") or "appearances"
    data = guest_page_data(*filters, sort)

    with card("categories"):
        selected = filters.category
        segments = "".join(
            f'<div class="cat-seg" style="width:{e(r["width"])};background:{e(r["color"])};'
            f'opacity:{"0.25" if selected != ALL_CATEGORIES and selected != r["label"] else "1"}"></div>'
            for r in data["categories"]
        )
        chips = "".join(
            f'<a class="cat-chip{" cat-chip-active" if r["label"] == selected else ""}" target="_self" '
            f'href="{href(kategorie=ALL_CATEGORIES if r["label"] == selected else r["label"])}">'
            f'<span class="swatch" style="background:{e(r["color"])}"></span>'
            f'<span class="cat-label">{e(r["label"])}</span>'
            f'<span class="cat-count"><b>{e(r["count"])}</b> · {e(r["share"])}</span></a>'
            for r in data["categories"]
        )
        html(
            card_head(f"{format_count(data['total'])} Gäste nach Kategorie", hint("Klick filtert die Seite"))
            + f'<div class="cat-bar">{segments}</div><div class="cat-grid">{chips}</div>'
        )

    with card("ranking"):
        head, toggle = st.columns([1, 1], vertical_alignment="center")
        with head:
            html('<span class="panel-title">Rangliste</span>')
        with toggle, st.container(horizontal=True, horizontal_alignment="right"):
            st.segmented_control(
                "Sortierung",
                ["appearances", "topics"],
                format_func={"appearances": "Häufigste Auftritte", "topics": "Größte Themenbreite"}.get,
                default="appearances",
                key="guest_sort",
                label_visibility="collapsed",
            )
        value_header = "Themen" if sort == "topics" else "Auftritte"
        rows = "".join(
            f'<a class="rank-row" target="_self" href="{person_link(r["name"])}">'
            f'<span class="rank">{e(r["rank"])}</span><span class="table-name truncate">{e(r["name"])}</span>'
            f'<span class="table-cell">{e(r["party"])}</span><span class="table-cell truncate">{e(r["role"])}</span>'
            f'<span class="table-cell">{e(r["category"])}</span>'
            f'<span class="bar-with-value">{bar_cell(r["width"])}<span class="table-value">{e(r["value"])}</span></span></a>'
            for r in data["ranking"]
        ) or empty_state("Keine Gäste für diese Filterkombination.", 160)
        html(
            '<div class="rank-row rank-head"><span>#</span><span>Name</span><span>Partei</span><span>Rolle</span>'
            f'<span>Kategorie</span><span style="text-align:right">{value_header}</span></div>'
            + rows
            + hint("Partei = zuletzt erfasste Zugehörigkeit")
        )

    footer()


# ---------------------------------------------------------------------------
# Personenprofil
# ---------------------------------------------------------------------------


def _labelled_bar(label: str, value: str, width: str, color: str = "var(--blue)") -> str:
    return (
        f'<div class="labelled-bar"><span class="table-name truncate" title="{e(label)}">{e(label)}</span>'
        f'{bar_cell(width, color, "8px")}<span class="table-value table-value-dark">{e(value)}</span></div>'
    )


def _link_row(r: dict[str, str]) -> str:
    return (
        f'<a class="list-row" target="_self" href="{person_link(r["name"])}">'
        f'<span class="table-name">{e(r["name"])}</span><span class="table-value">{e(r["value"])}×</span></a>'
    )


def person_page(filters: Filters, name: str) -> None:
    filter_bar(filters, with_category=False)
    p = profile_data(filters.timeframe, filters.show, name)

    if not p:
        with card("not-found"):
            html(
                '<span class="panel-title">Person nicht gefunden</span>'
                '<p class="section-copy">Für diesen Namen gibt es im gewählten Zeitraum keine Auftritte.</p>'
                + more_link("Zur Gästeliste →", href("gaeste", person=None))
            )
        footer()
        return

    html(
        f'<div class="crumbs"><a href="{href("gaeste", person=None)}" target="_self">Gäste</a>'
        f'<span>/</span><span>{e(p["name"])}</span></div>'
        '<div class="page-header"><div class="profile-title-row">'
        f'<h1 class="page-header-title">{e(p["name"])}</h1>'
        f'<span class="badge-party">{e(p["party"])}</span><span class="badge-neutral">{e(p["category"])}</span></div>'
        f'<p class="page-header-sub">{e(p["roles"])}</p></div>'
    )
    kpi_strip([
        {"label": "Auftritte", "value": p["appearances"], "note": p["rank_note"]},
        {"label": "Themenbreite", "value": p["topic_breadth"], "note": "verschiedene Themen"},
        {"label": "Zeitraum", "value": p["span"], "note": p["last_seen"]},
        {"label": "Häufigste Sendung", "value": p["top_show"], "note": p["top_show_note"]},
    ])

    left, right = st.columns([1.4, 1])
    with left, card("profile-years"):
        html(card_head("Auftritte pro Jahr"))
        year_bar_chart(p["years"], "Auftritte")
    with right, card("profile-shows"):
        html(
            card_head("Nach Sendung")
            + "".join(_labelled_bar(r["show"], r["value"], r["width"], r["color"]) for r in p["shows"])
        )

    c1, c2, c3 = st.columns(3)
    with c1, card("profile-topics"):
        html(card_head("Themen") + "".join(_labelled_bar(r["label"], r["value"], r["width"]) for r in p["topics"]))
    with c2, card("profile-co"):
        html(card_head("Häufigste Mitgäste") + "".join(_link_row(r) for r in p["co_guests"]))
    with c3, card("profile-recent"):
        html(
            card_head("Letzte Auftritte")
            + "".join(
                f'<div class="recent-row"><span class="table-cell tabular">{e(r["date"])}</span>'
                f'<span class="recent-main"><span class="table-name">{e(r["show"])}</span>'
                f'<span class="table-cell truncate">{e(r["topic"])}</span></span></div>'
                for r in p["recent"]
            )
        )

    footer()


# ---------------------------------------------------------------------------
# Themen
# ---------------------------------------------------------------------------


def themen_page() -> None:
    filters = current_filters()
    filter_bar(filters, with_category=False)
    page_header("Themen-Analyse", "Themen", "Welche Themen dominierten die Diskussion?")

    data = topic_page_data(filters.timeframe, filters.show, st.query_params.get("thema"))
    active = data["active"]

    left, right = st.columns([1.15, 1])
    with left, card("topic-list", tight=True):
        rows = []
        for r in data["rows"]:
            is_active = r["id"] == active
            badge = '<span class="badge-neutral badge-xs">Format-Cluster</span>' if r["is_format"] else ""
            stroke = BLUE if is_active else AXIS_COLOR
            rows.append(
                f'<a class="topic-row{" topic-row-active" if is_active else ""}" target="_self" href="{href(thema=r["id"])}">'
                f'<span class="rank">{e(r["rank"])}</span>'
                f'<span class="topic-label"><span class="table-name truncate{" topic-name-active" if is_active else ""}">{e(r["label"])}</span>{badge}</span>'
                f'<img class="spark" alt="" width="108" height="28" src="{sparkline_src(r["spark"], stroke)}">'
                f'<span class="table-value table-value-dark">{e(r["episodes"])}</span></a>'
            )
        body = "".join(rows) or empty_state("Keine klassifizierten Themen für diese Auswahl.")
        html(
            '<div class="topic-row topic-head"><span>#</span><span>Thema</span><span>Verlauf</span>'
            '<span style="text-align:right">Ep.</span></div>'
            + body
            + f'<div class="topic-footnote">{e(data["note"])}</div>'
        )

    with right, card("topic-detail"):
        d = data["detail"]
        if not d:
            html(empty_state("Kein Thema ausgewählt."))
        else:
            badge = '<span class="badge-neutral">Format-Cluster</span>' if d.get("is_format") else ""
            html(
                '<div class="detail-head"><div class="detail-title-row">'
                f'<h2 class="detail-title">{e(d["label"])}</h2>{badge}</div>'
                f'<span class="mono-note">Topic {e(d["id"])} · {e(d["terms"])}</span></div>'
                '<div class="mini-stats">'
                + "".join(
                    f'<div class="mini-stat-cell"><span class="stat-title">{label}</span>'
                    f'<span class="mini-stat">{e(d[key])}</span></div>'
                    for label, key in (("Episoden", "episodes"), ("Anteil", "share"), ("Höhepunkt", "peak"))
                )
                + "</div>"
            )
            html('<span class="subhead">Episoden pro Jahr</span>')
            year_bar_chart(d["years"], "Episoden")
            html(
                '<span class="subhead">Häufigste Gäste zum Thema</span>'
                + "".join(_link_row(r) for r in d["guests"])
            )

    footer()


# ---------------------------------------------------------------------------
# Netzwerke
# ---------------------------------------------------------------------------


def _interactive_html(file_name: str, key: str, label: str, height: int) -> None:
    path = DATA_DIR / file_name
    if not path.exists():
        html(hint("Kein HTML-Export vorhanden."))
        return
    if st.toggle(label, key=key, help="Große HTML-Exporte werden nur bei Bedarf geladen."):
        content = load_html_content(str(path))
        if content is not None:
            components.html(content, height=height, scrolling=True)


def netzwerke_page() -> None:
    current_filters()  # hält die Filter für den nächsten Seitenwechsel fest
    scope_bar(
        "Netzwerk-Exporte werden in der Analyse-Pipeline erzeugt. "
        "Zeitraum- und Sendungsfilter gelten hier nicht."
    )

    head, tabs = st.columns([2, 1], vertical_alignment="bottom")
    with head:
        page_header("Netzwerke", "Ko-Auftritte und Themenbeziehungen")
    with tabs, st.container(horizontal=True, horizontal_alignment="right"):
        tab = st.segmented_control(
            "Netzwerk",
            ["guests", "topics"],
            format_func={"guests": "Gäste-Netzwerk", "topics": "Themen-Netzwerk"}.get,
            default="guests",
            key="network_tab",
            label_visibility="collapsed",
        ) or "guests"

    spec = NETWORK_ASSET_SPECS["guest_network" if tab == "guests" else "topic_network"]
    left, right = st.columns([2, 1])
    with left, card("network", tight=True):
        image = DATA_DIR / spec["image"]
        if image.exists():
            st.image(str(image), width="stretch")
        else:
            html(empty_state("Keine statische Vorschau verfügbar.", 480))
        html(hint("Knotengröße = Anzahl Verbindungen · Kantenstärke = gemeinsame Auftritte"))
        _interactive_html(spec["html"], f"load_{tab}_network_html", "Interaktive Ansicht laden", 880)
    with right:
        with card("network-text"):
            html(f'<p class="body-copy">{e(spec["description"])}</p>')
        if tab == "guests":
            with card("network-pairs"):
                html(
                    card_head("Stärkste Verbindungen", '<span class="scope-badge">Gesamter Zeitraum</span>')
                    + "".join(_pair_row(r) for r in network_pairs())
                )

    with card("diagnostics"):
        html(card_head("Modell-Diagnose (BERTopic)", hint("Schwere Visualisierungen — werden erst auf Anfrage geladen")))
        for key in ("topics_visualization", "topics_barchart"):
            viz = NETWORK_ASSET_SPECS[key]
            if not (DATA_DIR / viz["html"]).exists():
                continue
            html(
                f'<div class="lazy-head"><span class="table-name">{e(viz["title"])}</span>'
                f'<span class="table-cell">{e(viz["description"])}</span></div>'
            )
            _interactive_html(viz["html"], f"load_{key}_html", "Laden", 800)

    footer()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

PAGES = {
    "overview": st.Page(overview_page, title="Übersicht", default=True),
    "gaeste": st.Page(gaeste_page, title="Gäste", url_path="gaeste"),
    "themen": st.Page(themen_page, title="Themen", url_path="themen"),
    "netzwerke": st.Page(netzwerke_page, title="Netzwerke", url_path="netzwerke"),
}

st.html(CSS)

if get_bundle().error is not None:
    with card("error"):
        html(
            '<span class="panel-title">Daten nicht verfügbar</span>'
            f'<p class="section-copy">{e(get_bundle().error)}</p>'
        )
    st.stop()

st.navigation(list(PAGES.values()), position="top").run()
