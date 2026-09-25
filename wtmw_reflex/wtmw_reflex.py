"""
WTMW Reflex Dashboard — redesign
================================
Pages: Übersicht (/) · Gäste (/gaeste) · Personenprofil (/gaeste/[guest])
       Themen (/themen) · Netzwerke (/netzwerke)

Changes vs. the previous version (numbers refer to the design review):
  1  Stacked episodes-per-year chart, incomplete years dimmed, coverage note
  2  One coverage module replaces Sendungsmix chart + chips + coverage cards
  3  KPI strip with short values; "Mit Thema" replaces "Top-Thema"
  4  filter_bar() on every page; active filters are highlighted and removable
  5  Top guests and top pairings on the overview; every name links to a profile
  6  Datenstand in the footer; reload button removed from the public view
  7  CTA row removed
  9  Primary module per page; descriptions that repeat titles removed
 10  Personensuche → nav search + /gaeste/[guest] profile
 11  Themen and Netzwerke are separate pages; network scope is labelled
"""

from __future__ import annotations

import logging
from urllib.parse import unquote

import reflex as rx

from reflex_dashboard_data import (
    ALL_CATEGORIES,
    ALL_SHOWS,
    DEFAULT_TIMEFRAME,
    TIMEFRAME_LABELS,
    data_updated_at_label,
    ensure_network_assets,
    filter_guests,
    filter_shows,
    format_count,
    load_dashboard_bundle,
)
from reflex_dashboard_redesign import (
    build_axis_labels,
    build_category_share_rows,
    build_coverage_rows,
    build_guest_rank_rows,
    build_pair_rows,
    build_kpis,
    build_person_profile,
    build_stacked_timeline,
    build_topic_detail,
    build_topic_list_rows,
    default_topic_id,
    person_href,
    search_suggestions,
    topic_label_mismatches,
)

_log = logging.getLogger(__name__)

BUNDLE = load_dashboard_bundle()
NETWORK_ASSETS = ensure_network_assets()

for _m in topic_label_mismatches():
    _log.warning("topic_labels.json: id %s '%s' passt nicht zu Modell-Termen (%s)", _m["id"], _m["label"], _m["terms"])

_TOOLTIP_STYLE = {
    "background": "white",
    "border": "1px solid #e2e8f0",
    "borderRadius": "10px",
    "fontSize": "13px",
    "boxShadow": "0 4px 12px rgba(0,0,0,.1)",
}
_AXIS_TICK = {"fill": "#94a3b8", "fontSize": 12}

# Small per-render cache so the profile / topic detail builders run once per
# state change instead of once per computed var.
_CACHE: dict[tuple, object] = {}


def _cached(key: tuple, fn):
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = fn()
    return _CACHE[key]


# ─── State ──────────────────────────────────────────────────────────────────


class DashboardState(rx.State):
    selected_timeframe: str = DEFAULT_TIMEFRAME
    selected_show: str = ALL_SHOWS
    selected_guest_category: str = ALL_CATEGORIES

    bundle_refresh_nonce: int = 0
    data_updated_at: str = data_updated_at_label()

    guest_sort: str = "appearances"          # "appearances" | "topics"
    selected_topic_id: str = ""
    network_tab: str = "guests"              # "guests" | "topics"
    load_topics_visualization: bool = False
    load_topics_barchart: bool = False

    search_query: str = ""
    selected_guest_name: str = ""

    # ── filter actions ─────────────────────────────────────────
    def set_selected_timeframe(self, value: str) -> None:
        self.selected_timeframe = value

    def set_selected_show(self, value: str) -> None:
        self.selected_show = value

    def set_selected_guest_category(self, value: str) -> None:
        self.selected_guest_category = value

    def toggle_show(self, value: str) -> None:
        self.selected_show = ALL_SHOWS if self.selected_show == value else value

    def toggle_category(self, value: str) -> None:
        self.selected_guest_category = ALL_CATEGORIES if self.selected_guest_category == value else value

    def clear_show(self) -> None:
        self.selected_show = ALL_SHOWS

    def clear_category(self) -> None:
        self.selected_guest_category = ALL_CATEGORIES

    def reset_filters(self) -> None:
        self.selected_timeframe = DEFAULT_TIMEFRAME
        self.selected_show = ALL_SHOWS
        self.selected_guest_category = ALL_CATEGORIES

    # ── page actions ───────────────────────────────────────────
    def set_guest_sort(self, value: str) -> None:
        self.guest_sort = value

    def select_topic(self, topic_id: str) -> None:
        self.selected_topic_id = topic_id

    def set_network_tab(self, value: str) -> None:
        self.network_tab = value

    def enable_topics_visualization(self) -> None:
        self.load_topics_visualization = True

    def enable_topics_barchart(self) -> None:
        self.load_topics_barchart = True

    def set_search_query(self, value: str) -> None:
        self.search_query = value

    def clear_search(self) -> None:
        self.search_query = ""

    def load_person(self) -> None:
        self.selected_guest_name = unquote(self.router.page.params.get("guest", ""))
        self.search_query = ""

    def refresh_data(self) -> None:
        """Kept for admin use; no longer wired to a public button."""
        global BUNDLE, NETWORK_ASSETS
        BUNDLE = load_dashboard_bundle()
        NETWORK_ASSETS = ensure_network_assets()
        _CACHE.clear()
        self.bundle_refresh_nonce += 1
        self.data_updated_at = data_updated_at_label()

    # ── filtered frames (python-side helpers, not vars) ─────────
    def _shows(self):
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return fs

    def _guests(self, with_category: bool = True):
        cat = self.selected_guest_category if with_category else ALL_CATEGORIES
        fg, _, _ = filter_guests(BUNDLE.guests, self.selected_timeframe, self.selected_show, cat)
        return fg

    def _key(self, *extra) -> tuple:
        return (self.bundle_refresh_nonce, self.selected_timeframe, self.selected_show,
                self.selected_guest_category, *extra)

    # ── filter bar ─────────────────────────────────────────────
    @rx.var(cache=True)
    def show_filter_active(self) -> bool:
        return self.selected_show != ALL_SHOWS

    @rx.var(cache=True)
    def category_filter_active(self) -> bool:
        return self.selected_guest_category != ALL_CATEGORIES

    @rx.var(cache=True)
    def any_filter_active(self) -> bool:
        return (self.selected_timeframe != DEFAULT_TIMEFRAME or self.show_filter_active
                or self.category_filter_active)

    @rx.var(cache=True)
    def filter_feedback(self) -> str:
        _ = self.bundle_refresh_nonce
        fs = self._shows()
        fg = self._guests()
        n_eps = fs["uid"].nunique() if "uid" in fs.columns else len(fs)
        n_g = fg["name"].dropna().nunique() if "name" in fg.columns else 0
        return f"{format_count(n_eps)} Episoden · {format_count(n_g)} Gäste"

    # ── Übersicht ──────────────────────────────────────────────
    @rx.var(cache=True)
    def kpis(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return build_kpis(self._shows(), self._guests())

    @rx.var(cache=True)
    def timeline_rows(self) -> list[dict[str, int | str]]:
        _ = self.bundle_refresh_nonce
        return _cached(self._key("tl"), lambda: build_stacked_timeline(self._shows()))[0]

    @rx.var(cache=True)
    def timeline_series(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return _cached(self._key("tl"), lambda: build_stacked_timeline(self._shows()))[1]

    @rx.var(cache=True)
    def legend_series(self) -> list[dict[str, str]]:
        return [s for s in self.timeline_series if s["partial"] == "0"]

    @rx.var(cache=True)
    def has_timeline(self) -> bool:
        return len(self.timeline_rows) > 0

    @rx.var(cache=True)
    def coverage_rows(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        # Coverage ignores the show filter so every show stays clickable.
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, ALL_SHOWS)
        return build_coverage_rows(fs)

    @rx.var(cache=True)
    def coverage_axis(self) -> list[str]:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, ALL_SHOWS)
        return build_axis_labels(fs)

    @rx.var(cache=True)
    def top_guests(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return build_guest_rank_rows(self._guests(), BUNDLE.guest_metadata, "appearances", 10)

    @rx.var(cache=True)
    def top_pairs(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return build_pair_rows(self._guests(with_category=False), limit=6)

    @rx.var(cache=True)
    def network_pairs(self) -> list[dict[str, str]]:
        """Unfiltered: /netzwerke shows the whole data set, whatever filters are set elsewhere."""
        _ = self.bundle_refresh_nonce
        return build_pair_rows(BUNDLE.guests, limit=6)

    # ── Gäste ──────────────────────────────────────────────────
    @rx.var(cache=True)
    def category_rows(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return build_category_share_rows(self._guests(with_category=False))

    @rx.var(cache=True)
    def guest_rank_rows(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return build_guest_rank_rows(self._guests(), BUNDLE.guest_metadata, self.guest_sort, 25)

    @rx.var(cache=True)
    def guest_value_header(self) -> str:
        return "Themen" if self.guest_sort == "topics" else "Auftritte"

    @rx.var(cache=True)
    def total_guest_label(self) -> str:
        _ = self.bundle_refresh_nonce
        n = self._guests(with_category=False)["name"].dropna().nunique() if not BUNDLE.guests.empty else 0
        return f"{format_count(n)} Gäste nach Kategorie"

    # ── Themen ─────────────────────────────────────────────────
    @rx.var(cache=True)
    def topic_rows(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return build_topic_list_rows(self._shows())

    @rx.var(cache=True)
    def active_topic_id(self) -> str:
        ids = {r["id"] for r in self.topic_rows}
        if self.selected_topic_id in ids:
            return self.selected_topic_id
        return default_topic_id(self._shows())

    def _topic_detail(self) -> dict:
        tid = self.active_topic_id
        return _cached(self._key("topic", tid),
                       lambda: build_topic_detail(self._shows(), self._guests(with_category=False), tid)) or {}

    @rx.var(cache=True)
    def topic_header(self) -> dict[str, str]:
        _ = self.bundle_refresh_nonce
        d = self._topic_detail()
        return {k: str(d.get(k, "")) for k in ("id", "label", "terms", "episodes", "share", "peak")} | {
            "is_format": "1" if d.get("is_format") else ""}

    @rx.var(cache=True)
    def topic_years(self) -> list[dict[str, int | str]]:
        _ = self.bundle_refresh_nonce
        return self._topic_detail().get("years", [])

    @rx.var(cache=True)
    def topic_guests(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return self._topic_detail().get("guests", [])

    @rx.var(cache=True)
    def unclassified_note(self) -> str:
        _ = self.bundle_refresh_nonce
        fs = self._shows()
        if fs.empty or "topic" not in fs.columns:
            return ""
        eps = fs.drop_duplicates(subset=["uid"]) if "uid" in fs.columns else fs
        outliers = int((eps["topic"] == -1).sum())
        missing = int(eps["topic"].isna().sum())
        return f"Ohne Thema: {format_count(outliers + missing)} Episoden ({format_count(outliers)} Ausreißer, {format_count(missing)} ohne Beschreibung)"

    # ── Profil ─────────────────────────────────────────────────
    def _profile(self) -> dict:
        name = self.selected_guest_name
        return _cached(self._key("person", name),
                       lambda: build_person_profile(self._guests(with_category=False), BUNDLE.guest_metadata, name)) or {}

    @rx.var(cache=True)
    def has_profile(self) -> bool:
        _ = self.bundle_refresh_nonce
        return bool(self._profile())

    @rx.var(cache=True)
    def profile_header(self) -> dict[str, str]:
        _ = self.bundle_refresh_nonce
        p = self._profile()
        keys = ("name", "party", "category", "roles", "appearances", "rank_note", "topic_breadth",
                "span", "last_seen", "top_show", "top_show_note")
        return {k: str(p.get(k, "")) for k in keys}

    @rx.var(cache=True)
    def profile_years(self) -> list[dict[str, int | str]]:
        _ = self.bundle_refresh_nonce
        return self._profile().get("years", [])

    @rx.var(cache=True)
    def profile_shows(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return self._profile().get("shows", [])

    @rx.var(cache=True)
    def profile_topics(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return self._profile().get("topics", [])

    @rx.var(cache=True)
    def profile_co_guests(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return self._profile().get("co_guests", [])

    @rx.var(cache=True)
    def profile_recent(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return self._profile().get("recent", [])

    # ── Suche ──────────────────────────────────────────────────
    @rx.var(cache=True)
    def suggestions(self) -> list[dict[str, str]]:
        _ = self.bundle_refresh_nonce
        return search_suggestions(BUNDLE.guests, self.search_query)

    @rx.var(cache=True)
    def show_suggestions(self) -> bool:
        return self.search_query.strip() != ""


# ─── Shell ───────────────────────────────────────────────────────────────────


def _nav_link(label: str, href: str, key: str, active: str) -> rx.Component:
    cls = "nav-link nav-link-active" if active == key else "nav-link"
    return rx.link(label, href=href, class_name=cls)


def _suggestion(item: rx.Var) -> rx.Component:
    return rx.link(
        rx.flex(
            rx.flex(
                rx.text(item["name"], class_name="suggest-name"),
                rx.text(item["party"], class_name="suggest-meta"),
                direction="column",
            ),
            rx.text(item["value"], class_name="suggest-count"),
            justify="between",
            align="center",
            width="100%",
        ),
        href=item["href"],
        class_name="suggest-row",
    )


def nav_search() -> rx.Component:
    return rx.box(
        rx.input(
            placeholder="Person suchen, z. B. Söder …",
            value=DashboardState.search_query,
            on_change=DashboardState.set_search_query.debounce(200),
            class_name="nav-search",
            size="2",
        ),
        rx.cond(
            DashboardState.show_suggestions,
            rx.box(
                rx.cond(
                    DashboardState.suggestions.length() > 0,
                    rx.foreach(DashboardState.suggestions, _suggestion),
                    rx.text("Keine Treffer.", class_name="suggest-empty"),
                ),
                class_name="suggest-panel",
            ),
        ),
        class_name="nav-search-wrap",
    )


def navbar(active: str) -> rx.Component:
    return rx.box(
        rx.box(
            rx.flex(
                rx.link(
                    rx.flex(
                        rx.text("WTMW", class_name="nav-brand-title"),
                        rx.text("Wer talkt mit wem?", class_name="nav-brand-sub"),
                        align="baseline",
                        gap="6px",
                    ),
                    href="/",
                    class_name="nav-brand",
                ),
                rx.flex(
                    _nav_link("Übersicht", "/", "overview", active),
                    _nav_link("Gäste", "/gaeste", "gaeste", active),
                    _nav_link("Themen", "/themen", "themen", active),
                    _nav_link("Netzwerke", "/netzwerke", "netzwerke", active),
                    class_name="nav-links",
                ),
                align="center",
                gap="28px",
            ),
            nav_search(),
            class_name="nav-inner",
        ),
        class_name="navbar",
    )


def _filter_select(label: str, items, value, on_change) -> rx.Component:
    return rx.flex(
        rx.text(label, class_name="fbar-label"),
        rx.select(items=items, value=value, on_change=on_change, variant="ghost", size="2"),
        class_name="fbar-item",
        align="center",
    )


def _active_chip(label: str, value: rx.Var, on_clear) -> rx.Component:
    return rx.flex(
        rx.text(label, class_name="fbar-label fbar-label-active"),
        rx.text(value, class_name="fbar-value-active"),
        rx.button("×", on_click=on_clear, variant="ghost", size="1", class_name="fbar-clear"),
        class_name="fbar-item fbar-item-active",
        align="center",
    )


def filter_bar(with_category: bool = True, note: str | None = None) -> rx.Component:
    """Sticky filter bar shown on every filterable page (#4)."""
    show_ctrl = rx.cond(
        DashboardState.show_filter_active,
        _active_chip("Sendung", DashboardState.selected_show, DashboardState.clear_show),
        _filter_select("Sendung", BUNDLE.show_options, DashboardState.selected_show, DashboardState.set_selected_show),
    )
    controls = [
        _filter_select("Zeitraum", TIMEFRAME_LABELS, DashboardState.selected_timeframe, DashboardState.set_selected_timeframe),
        show_ctrl,
    ]
    if with_category:
        controls.append(
            rx.cond(
                DashboardState.category_filter_active,
                _active_chip("Gastkategorie", DashboardState.selected_guest_category, DashboardState.clear_category),
                _filter_select("Gastkategorie", BUNDLE.category_options, DashboardState.selected_guest_category,
                               DashboardState.set_selected_guest_category),
            )
        )
    return rx.box(
        rx.flex(
            rx.flex(*controls, gap="8px", wrap="wrap", align="center"),
            rx.flex(
                rx.text(note or DashboardState.filter_feedback, class_name="fbar-feedback"),
                rx.cond(
                    DashboardState.any_filter_active,
                    rx.button("Zurücksetzen", on_click=DashboardState.reset_filters, variant="ghost", size="1",
                              class_name="filter-reset"),
                ),
                gap="16px",
                align="center",
            ),
            class_name="fbar-inner",
        ),
        class_name="fbar",
    )


def scope_bar(text: str) -> rx.Component:
    return rx.box(
        rx.flex(
            rx.text("Gesamtdatensatz", class_name="scope-badge"),
            rx.text(text, class_name="scope-text"),
            class_name="fbar-inner",
            gap="10px",
            align="center",
            justify="start",
        ),
        class_name="fbar fbar-scope",
    )


def page_header(eyebrow: str, title: str | rx.Var, subtitle: str | None = None) -> rx.Component:
    children = [
        rx.text(eyebrow, class_name="page-header-eyebrow"),
        rx.heading(title, size="8", class_name="page-header-title"),
    ]
    if subtitle:
        children.append(rx.text(subtitle, class_name="page-header-sub"))
    return rx.flex(*children, direction="column", spacing="2", width="100%")


def card(*children, class_name: str = "", **props) -> rx.Component:
    return rx.box(*children, class_name=f"panel-card {class_name}".strip(), **props)


def card_head(title: str | rx.Var, right: rx.Component | None = None) -> rx.Component:
    return rx.flex(
        rx.heading(title, size="4", class_name="panel-title"),
        right or rx.fragment(),
        justify="between",
        align="baseline",
        width="100%",
    )


def more_link(label: str, href: str) -> rx.Component:
    return rx.link(label, href=href, class_name="more-link")


def footer() -> rx.Component:
    return rx.flex(
        rx.text("Datenstand ", DashboardState.data_updated_at, " · Talkshow-Daten von fernsehserien.de",
                class_name="footer-note"),
        rx.text("NLP via BERTopic & sentence-transformers · Bernstein Analytics", class_name="footer-note"),
        justify="between",
        wrap="wrap",
        gap="16px",
        class_name="page-footer",
    )


def shell(active: str, bar: rx.Component | None, *content, on_mount=None) -> rx.Component:
    return rx.box(
        navbar(active),
        bar or rx.fragment(),
        rx.box(*content, footer(), class_name="page-container page-stack"),
        class_name="app-shell",
    )


def kpi_strip(items: rx.Var) -> rx.Component:
    def _cell(item: rx.Var) -> rx.Component:
        return rx.flex(
            rx.text(item["label"], class_name="stat-title"),
            rx.text(item["value"], class_name="kpi-value"),
            rx.text(item["note"], class_name="kpi-note"),
            direction="column",
            gap="6px",
            class_name="kpi-cell",
        )
    return rx.box(rx.foreach(items, _cell), class_name="kpi-strip")


def empty_state(message: str, min_h: str = "200px") -> rx.Component:
    return rx.flex(rx.text(message, class_name="empty-copy"), align="center", justify="center",
                   min_height=min_h, width="100%", class_name="empty-state")


def bar_cell(width: rx.Var, color: str | rx.Var = "var(--blue)", height: str = "6px") -> rx.Component:
    return rx.box(
        rx.box(class_name="bar-fill", style={"width": width, "background": color}),
        class_name="bar-track",
        style={"height": height},
    )


# ─── Übersicht ───────────────────────────────────────────────────────────────


def timeline_chart() -> rx.Component:
    return rx.cond(
        DashboardState.has_timeline,
        rx.recharts.bar_chart(
            rx.recharts.cartesian_grid(stroke="#f1f5f9", vertical=False),
            rx.recharts.x_axis(data_key="period", tick_line=False, axis_line=False, tick=_AXIS_TICK),
            rx.recharts.y_axis(tick_line=False, axis_line=False, allow_decimals=False, tick=_AXIS_TICK, width=36),
            rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
            rx.foreach(
                DashboardState.timeline_series,
                lambda s: rx.recharts.bar(
                    data_key=s["key"],
                    name=s["show"],
                    stack_id="episodes",
                    fill=s["fill"],
                    is_animation_active=False,
                ),
            ),
            data=DashboardState.timeline_rows,
            width="100%",
            height=260,
            bar_category_gap="28%",
            margin={"top": 6, "right": 4, "left": 0, "bottom": 0},
        ),
        empty_state("Für diese Auswahl liegen keine Episodendaten vor.", "260px"),
    )


def _legend_item(s: rx.Var) -> rx.Component:
    return rx.flex(rx.box(class_name="swatch", style={"background": s["color"]}),
                   rx.text(s["show"], class_name="legend-label"), align="center", gap="6px")


def timeline_card() -> rx.Component:
    return card(
        card_head("Erfasste Episoden pro Jahr",
                  rx.flex(rx.foreach(DashboardState.legend_series, _legend_item), gap="14px", wrap="wrap")),
        timeline_chart(),
        rx.flex(
            rx.flex(rx.box(class_name="swatch swatch-partial"), rx.text("Unvollständig erfasst"),
                    align="center", gap="6px"),
            rx.text("Verläufe können aus der Erfassung stammen: Markus Lanz ist bis 2022 nur mit rund 10 Folgen "
                    "pro Jahr enthalten, 2025 hat Lücken bei Lanz und Maischberger."),
            class_name="chart-footnote",
        ),
        class_name="stack-16",
    )


def _coverage_row(row: rx.Var) -> rx.Component:
    return rx.box(
        rx.flex(rx.box(class_name="swatch swatch-sm", style={"background": row["color"]}),
                rx.text(row["show"]), class_name="cov-name", align="center", gap="8px"),
        rx.tooltip(
            rx.box(rx.box(class_name="cov-range", style={"left": row["left"], "width": row["width"],
                                                          "background": row["color"]}),
                   class_name="cov-track"),
            content=row["range"],
        ),
        rx.text(row["episodes"], class_name="cov-count"),
        on_click=DashboardState.toggle_show(row["show"]),
        class_name="cov-row",
    )


def coverage_card() -> rx.Component:
    return card(
        card_head("Datenabdeckung je Sendung", rx.text("Klick filtert", class_name="card-hint")),
        rx.flex(
            rx.foreach(DashboardState.coverage_rows, _coverage_row),
            rx.box(
                rx.box(),
                rx.flex(rx.foreach(DashboardState.coverage_axis, lambda y: rx.text(y)), justify="between",
                        class_name="cov-axis"),
                rx.box(),
                class_name="cov-row cov-row-axis",
            ),
            direction="column",
            gap="2px",
        ),
        class_name="stack-14",
    )


def _guest_row_compact(row: rx.Var) -> rx.Component:
    return rx.link(
        rx.text(row["rank"], class_name="rank"),
        rx.flex(rx.text(row["name"], class_name="table-name"), rx.text(row["party"], class_name="row-meta"),
                direction="column"),
        bar_cell(row["width"]),
        rx.text(row["value"], class_name="table-value"),
        href=row["href"],
        class_name="guest-row-compact",
    )


def top_guests_card() -> rx.Component:
    return card(
        card_head("Häufigste Gäste", more_link("Alle Gäste →", "/gaeste")),
        rx.foreach(DashboardState.top_guests, _guest_row_compact),
        class_name="stack-12",
    )


def _pair_row(row: rx.Var) -> rx.Component:
    return rx.flex(
        rx.flex(
            rx.link(row["a"], href=row["a_href"], class_name="table-name"),
            rx.text("+", class_name="pair-plus"),
            rx.link(row["b"], href=row["b_href"], class_name="table-name"),
            gap="8px", wrap="wrap", align="center",
        ),
        rx.text(row["value"], "×", class_name="table-value"),
        class_name="list-row",
    )


def pairs_card() -> rx.Component:
    return card(
        card_head("Wer sitzt am häufigsten zusammen?", more_link("Netzwerk →", "/netzwerke")),
        rx.cond(DashboardState.top_pairs.length() > 0,
                rx.box(rx.foreach(DashboardState.top_pairs, _pair_row)),
                empty_state("Keine Paarungsdaten für diese Auswahl.", "160px")),
        class_name="stack-12",
    )


def overview_page() -> rx.Component:
    if BUNDLE.error is not None:
        return shell("overview", None, card(rx.heading("Daten nicht verfügbar", size="6"),
                                            rx.text(BUNDLE.error, class_name="section-copy")))
    return shell(
        "overview",
        filter_bar(),
        page_header("Politische Talkshow-Analyse", "Wer talkt mit wem?",
                    "Daten aus sechs deutschen Talkshows — analysiert nach Gästen, Themen und Mustern. "
                    "NLP-gestützt mit BERTopic-Modell."),
        kpi_strip(DashboardState.kpis),
        timeline_card(),
        rx.grid(
            rx.flex(coverage_card(), pairs_card(), direction="column", gap="16px"),
            top_guests_card(),
            class_name="grid-2",
        ),
    )


# ─── Gäste ───────────────────────────────────────────────────────────────────


def _category_segment(row: rx.Var) -> rx.Component:
    return rx.box(
        class_name="cat-seg",
        style={
            "width": row["width"],
            "background": row["color"],
            "opacity": rx.cond(
                DashboardState.category_filter_active & (DashboardState.selected_guest_category != row["label"]),
                "0.25", "1"),
        },
    )


def _category_chip(row: rx.Var) -> rx.Component:
    return rx.box(
        rx.box(class_name="swatch", style={"background": row["color"]}),
        rx.text(row["label"], class_name="cat-label"),
        rx.text(rx.el.b(row["count"]), " · ", row["share"], class_name="cat-count"),
        on_click=DashboardState.toggle_category(row["label"]),
        class_name=rx.cond(DashboardState.selected_guest_category == row["label"], "cat-chip cat-chip-active",
                           "cat-chip"),
    )


def _guest_rank_row(row: rx.Var) -> rx.Component:
    return rx.link(
        rx.text(row["rank"], class_name="rank"),
        rx.text(row["name"], class_name="table-name"),
        rx.text(row["party"], class_name="table-cell"),
        rx.text(row["role"], class_name="table-cell truncate"),
        rx.text(row["category"], class_name="table-cell"),
        rx.flex(bar_cell(row["width"]), rx.text(row["value"], class_name="table-value"), class_name="bar-with-value"),
        href=row["href"],
        class_name="rank-row",
    )


def _seg_button(label: str, value: str) -> rx.Component:
    return rx.box(
        label,
        on_click=DashboardState.set_guest_sort(value),
        class_name=rx.cond(DashboardState.guest_sort == value, "seg-btn seg-btn-active", "seg-btn"),
    )


def gaeste_page() -> rx.Component:
    return shell(
        "gaeste",
        filter_bar(),
        page_header("Gäste-Analyse", "Gäste",
                    "Wer tritt am häufigsten auf? Welche Kategorien dominieren? "
                    "Filter nach Zeitraum, Sendung und Gastsegment."),
        card(
            card_head(DashboardState.total_guest_label, rx.text("Klick filtert die Seite", class_name="card-hint")),
            rx.flex(rx.foreach(DashboardState.category_rows, _category_segment), class_name="cat-bar"),
            rx.box(rx.foreach(DashboardState.category_rows, _category_chip), class_name="cat-grid"),
            class_name="stack-16",
        ),
        card(
            card_head("Rangliste", rx.flex(_seg_button("Häufigste Auftritte", "appearances"),
                                           _seg_button("Größte Themenbreite", "topics"), class_name="seg")),
            rx.box(
                rx.text("#"), rx.text("Name"), rx.text("Partei"), rx.text("Rolle"), rx.text("Kategorie"),
                rx.text(DashboardState.guest_value_header, text_align="right"),
                class_name="rank-row rank-head",
            ),
            rx.cond(
                DashboardState.guest_rank_rows.length() > 0,
                rx.box(rx.foreach(DashboardState.guest_rank_rows, _guest_rank_row)),
                empty_state("Keine Gäste für diese Filterkombination.", "160px"),
            ),
            rx.text("Partei = zuletzt erfasste Zugehörigkeit", class_name="card-hint"),
            class_name="stack-12",
        ),
    )


# ─── Personenprofil ─────────────────────────────────────────────────────────


def _profile_stat(label: str, value: rx.Var, note: rx.Var) -> rx.Component:
    return rx.flex(rx.text(label, class_name="stat-title"), rx.text(value, class_name="kpi-value"),
                   rx.text(note, class_name="kpi-note"), direction="column", gap="6px", class_name="kpi-cell")


def _labelled_bar(row: rx.Var, label_key: str, color_key: str | None = None) -> rx.Component:
    return rx.box(
        rx.text(row[label_key], class_name="table-name"),
        bar_cell(row["width"], row[color_key] if color_key else "var(--blue)", "8px"),
        rx.text(row["value"], class_name="table-value table-value-dark"),
        class_name="labelled-bar",
    )


def _link_row(row: rx.Var) -> rx.Component:
    return rx.link(rx.text(row["name"], class_name="table-name"), rx.text(row["value"], "×", class_name="table-value"),
                   href=row["href"], class_name="list-row list-row-link")


def _recent_row(row: rx.Var) -> rx.Component:
    return rx.box(
        rx.text(row["date"], class_name="table-cell tabular"),
        rx.flex(rx.text(row["show"], class_name="table-name"), rx.text(row["topic"], class_name="table-cell truncate"),
                direction="column", min_width="0"),
        class_name="recent-row",
    )


def person_page() -> rx.Component:
    h = DashboardState.profile_header
    return shell(
        "gaeste",
        filter_bar(with_category=False),
        rx.cond(
            DashboardState.has_profile,
            rx.flex(
                rx.flex(
                    rx.flex(rx.link("Gäste", href="/gaeste"), rx.text("/"), rx.text(h["name"]), class_name="crumbs"),
                    rx.flex(
                        rx.heading(h["name"], size="8", class_name="page-header-title"),
                        rx.text(h["party"], class_name="guest-detail-badge"),
                        rx.text(h["category"], class_name="badge-neutral"),
                        align="center", gap="12px", wrap="wrap",
                    ),
                    rx.text(h["roles"], class_name="page-header-sub"),
                    direction="column", gap="14px",
                ),
                rx.box(
                    _profile_stat("Auftritte", h["appearances"], h["rank_note"]),
                    _profile_stat("Themenbreite", h["topic_breadth"], "verschiedene Themen"),
                    _profile_stat("Zeitraum", h["span"], h["last_seen"]),
                    _profile_stat("Häufigste Sendung", h["top_show"], h["top_show_note"]),
                    class_name="kpi-strip",
                ),
                rx.grid(
                    card(card_head("Auftritte pro Jahr"),
                         rx.recharts.bar_chart(
                             rx.recharts.x_axis(data_key="period", tick_line=False, axis_line=False, tick=_AXIS_TICK),
                             rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
                             rx.recharts.bar(data_key="episodes", name="Auftritte", fill="#2563eb", radius=[3, 3, 0, 0],
                                             label={"position": "top", "fill": "#334155", "fontSize": 11}),
                             data=DashboardState.profile_years, width="100%", height=180,
                             margin={"top": 16, "right": 0, "left": 0, "bottom": 0}),
                         class_name="stack-14"),
                    card(card_head("Nach Sendung"),
                         rx.foreach(DashboardState.profile_shows, lambda r: _labelled_bar(r, "show", "color")),
                         class_name="stack-12"),
                    class_name="grid-2 grid-2-wide",
                ),
                rx.grid(
                    card(card_head("Themen"), rx.foreach(DashboardState.profile_topics, lambda r: _labelled_bar(r, "label")),
                         class_name="stack-12"),
                    card(card_head("Häufigste Mitgäste"), rx.foreach(DashboardState.profile_co_guests, _link_row),
                         class_name="stack-8"),
                    card(card_head("Letzte Auftritte"), rx.foreach(DashboardState.profile_recent, _recent_row),
                         class_name="stack-8"),
                    class_name="grid-3",
                ),
                direction="column", gap="24px", width="100%",
            ),
            card(rx.heading("Person nicht gefunden", size="5"),
                 rx.text("Für diesen Namen gibt es im gewählten Zeitraum keine Auftritte.", class_name="section-copy"),
                 more_link("Zur Gästeliste →", "/gaeste"), class_name="stack-12"),
        ),
    )


# ─── Themen ──────────────────────────────────────────────────────────────────


def _topic_row(row: rx.Var) -> rx.Component:
    active = DashboardState.active_topic_id == row["id"]
    return rx.box(
        rx.text(row["rank"], class_name="rank"),
        rx.flex(
            rx.text(row["label"], class_name=rx.cond(active, "table-name topic-name-active truncate", "table-name truncate")),
            rx.cond(row["is_format"] != "", rx.text("Format-Cluster", class_name="badge-neutral badge-xs")),
            align="center", gap="8px", min_width="0",
        ),
        rx.el.svg(
            rx.el.svg.path(d=row["spark"], fill="none", stroke=rx.cond(active, "#2563eb", "#94a3b8"),
                           stroke_width="1.75", stroke_linejoin="round"),
            view_box="0 0 108 28", width="108", height="28", overflow="visible",
        ),
        rx.text(row["episodes"], class_name="table-value table-value-dark"),
        on_click=DashboardState.select_topic(row["id"]),
        class_name=rx.cond(active, "topic-row topic-row-active", "topic-row"),
    )


def _topic_stat(label: str, value: rx.Var) -> rx.Component:
    return rx.flex(rx.text(label, class_name="stat-title"), rx.text(value, class_name="mini-stat"),
                   direction="column", gap="4px", class_name="mini-stat-cell")


def topic_detail_card() -> rx.Component:
    h = DashboardState.topic_header
    return card(
        rx.flex(
            rx.flex(rx.heading(h["label"], size="5", class_name="detail-title"),
                    rx.cond(h["is_format"] != "", rx.text("Format-Cluster", class_name="badge-neutral")),
                    align="center", gap="10px", wrap="wrap"),
            rx.text("Topic ", h["id"], " · ", h["terms"], class_name="mono-note"),
            direction="column", gap="8px",
        ),
        rx.box(_topic_stat("Episoden", h["episodes"]), _topic_stat("Anteil", h["share"]),
               _topic_stat("Höhepunkt", h["peak"]), class_name="mini-stats"),
        rx.flex(
            rx.text("Episoden pro Jahr", class_name="subhead"),
            rx.recharts.bar_chart(
                rx.recharts.x_axis(data_key="period", tick_line=False, axis_line=False, tick=_AXIS_TICK),
                rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
                rx.recharts.bar(data_key="episodes", name="Episoden", fill="#2563eb", radius=[3, 3, 0, 0],
                                label={"position": "top", "fill": "#334155", "fontSize": 11}),
                data=DashboardState.topic_years, width="100%", height=180,
                margin={"top": 16, "right": 0, "left": 0, "bottom": 0},
            ),
            direction="column", gap="6px",
        ),
        rx.flex(
            rx.text("Häufigste Gäste zum Thema", class_name="subhead"),
            rx.foreach(DashboardState.topic_guests, _link_row),
            direction="column",
        ),
        class_name="stack-18 sticky-card",
    )


def themen_page() -> rx.Component:
    return shell(
        "themen",
        filter_bar(with_category=False),
        page_header("Themen-Analyse", "Themen", "Welche Themen dominierten die Diskussion?"),
        rx.grid(
            card(
                rx.box(rx.text("#"), rx.text("Thema"), rx.text("Verlauf"), rx.text("Ep.", text_align="right"),
                       class_name="topic-row topic-head"),
                rx.cond(DashboardState.topic_rows.length() > 0,
                        rx.box(rx.foreach(DashboardState.topic_rows, _topic_row)),
                        empty_state("Keine klassifizierten Themen für diese Auswahl.")),
                rx.text(DashboardState.unclassified_note, class_name="topic-footnote"),
                class_name="card-tight",
            ),
            topic_detail_card(),
            class_name="grid-2 grid-2-topics",
        ),
    )


# ─── Netzwerke ───────────────────────────────────────────────────────────────


def _network_view(asset: dict, alt: str) -> rx.Component:
    img = (rx.image(src=asset.get("image_src"), alt=alt, class_name="network-image")
           if asset.get("has_image") else empty_state("Keine statische Vorschau verfügbar.", "480px"))
    link = (rx.el.a("Interaktive Ansicht öffnen →", href=asset.get("html_src"), target="_blank",
                    rel="noopener noreferrer", class_name="more-link")
            if asset.get("has_html") else rx.text("Kein HTML-Export vorhanden.", class_name="card-hint"))
    return card(
        img,
        rx.flex(rx.text("Knotengröße = Anzahl Verbindungen · Kantenstärke = gemeinsame Auftritte",
                        class_name="card-hint"), link, justify="between", align="center", wrap="wrap", gap="8px"),
        class_name="stack-12 card-tight",
    )


def _network_tab(label: str, value: str) -> rx.Component:
    return rx.box(label, on_click=DashboardState.set_network_tab(value),
                  class_name=rx.cond(DashboardState.network_tab == value, "pill-btn pill-btn-active", "pill-btn"))


def _lazy_row(title: str, description: str, html_src: str | None, loaded: rx.Var, handler) -> rx.Component:
    if not html_src:
        return rx.fragment()
    return rx.box(
        rx.flex(
            rx.flex(rx.text(title, class_name="table-name"), rx.text(description, class_name="table-cell"),
                    direction="column", gap="2px"),
            rx.cond(loaded, rx.fragment(), rx.button("Laden", on_click=handler, variant="soft", size="2")),
            justify="between", align="center",
        ),
        rx.cond(loaded, rx.el.iframe(src=html_src, width="100%", loading="lazy", class_name="network-iframe")),
        class_name="lazy-row",
    )


def netzwerke_page() -> rx.Component:
    guest_network = NETWORK_ASSETS.get("guest_network", {})
    topic_network = NETWORK_ASSETS.get("topic_network", {})
    tv = NETWORK_ASSETS.get("topics_visualization", {})
    tb = NETWORK_ASSETS.get("topics_barchart", {})
    return shell(
        "netzwerke",
        scope_bar("Netzwerk-Exporte werden in der Analyse-Pipeline erzeugt. Zeitraum- und Sendungsfilter gelten hier nicht."),
        rx.flex(
            page_header("Netzwerke", "Ko-Auftritte und Themenbeziehungen"),
            rx.flex(_network_tab("Gäste-Netzwerk", "guests"), _network_tab("Themen-Netzwerk", "topics"), class_name="pill"),
            justify="between", align="end", gap="24px", width="100%",
        ),
        rx.grid(
            rx.cond(DashboardState.network_tab == "guests",
                    _network_view(guest_network, "Gäste-Netzwerk"),
                    _network_view(topic_network, "Themen-Netzwerk")),
            rx.flex(
                card(rx.text(rx.cond(DashboardState.network_tab == "guests",
                                     str(guest_network.get("description", "")),
                                     str(topic_network.get("description", ""))), class_name="body-copy")),
                rx.cond(
                    DashboardState.network_tab == "guests",
                    card(card_head("Stärkste Verbindungen", rx.text("Gesamter Zeitraum", class_name="scope-badge")),
                         rx.foreach(DashboardState.network_pairs, _pair_row), class_name="stack-8"),
                ),
                direction="column", gap="16px",
            ),
            class_name="grid-2 grid-2-network",
        ),
        card(
            card_head("Modell-Diagnose (BERTopic)", rx.text("Schwere Visualisierungen — werden erst auf Anfrage geladen",
                                                           class_name="card-hint")),
            _lazy_row(str(tv.get("title", "BERTopic-Cluster")), str(tv.get("description", "")),
                      tv.get("html_src") if tv.get("has_html") else None,
                      DashboardState.load_topics_visualization, DashboardState.enable_topics_visualization),
            _lazy_row(str(tb.get("title", "Themen-Häufigkeit")), str(tb.get("description", "")),
                      tb.get("html_src") if tb.get("has_html") else None,
                      DashboardState.load_topics_barchart, DashboardState.enable_topics_barchart),
            class_name="stack-8",
        ),
    )


# ─── App ─────────────────────────────────────────────────────────────────────

app = rx.App(
    theme=rx.theme(appearance="light", accent_color="blue", gray_color="slate", radius="medium", scaling="100%"),
    stylesheets=["/reflex_dashboard.css", "/reflex_dashboard_redesign.css"],
)

app.add_page(overview_page, route="/", title="Wer talkt mit wem? – Übersicht")
app.add_page(gaeste_page, route="/gaeste", title="WTMW – Gäste")
app.add_page(person_page, route="/gaeste/[guest]", title="WTMW – Personenprofil", on_load=DashboardState.load_person)
app.add_page(themen_page, route="/themen", title="WTMW – Themen")
app.add_page(netzwerke_page, route="/netzwerke", title="WTMW – Netzwerke")
# Old route keeps working for bookmarks.
app.add_page(gaeste_page, route="/personensuche", title="WTMW – Gäste")
