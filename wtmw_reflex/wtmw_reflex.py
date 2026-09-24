"""
WTMW Reflex Dashboard
=====================
Modern & Clean design | Blue accent | Multi-page navigation
Pages: Übersicht (/) · Gäste (/gaeste) · Themen (/themen) · Personensuche (/personensuche)
"""

from __future__ import annotations

import pandas as pd
import reflex as rx

from reflex_dashboard_data import (
    ALL_CATEGORIES,
    ALL_SHOWS,
    DEFAULT_TIMEFRAME,
    TIMEFRAME_LABELS,
    build_appearance_rows,
    build_category_rows,
    build_episode_count,
    build_show_cards,
    build_show_mix_rows,
    build_timeline_rows,
    build_top_diversity_label,
    build_top_guest_label,
    build_top_topic_label,
    build_topic_diversity_rows,
    build_topic_rows,
    build_topics_over_time_data,
    build_treemap_data,
    build_treemap_legend,
    build_top_guest_pairs,
    build_unique_guest_count,
    compact_label,
    current_timestamp_label,
    data_updated_at_label,
    ensure_network_assets,
    filter_guests,
    filter_shows,
    format_count,
    format_date_range,
    format_topic_text,
    load_dashboard_bundle,
)

# ─── Globals ────────────────────────────────────────────────────────────────

BUNDLE = load_dashboard_bundle()
NETWORK_ASSETS = ensure_network_assets()
INITIAL_LOADED_AT = current_timestamp_label()
INITIAL_DATA_UPDATED_AT = data_updated_at_label()

# Chart colours aligned with the blue design system
PALETTE = {
    "blue":           "#2563eb",
    "blue_mid":       "#3b82f6",
    "blue_light":     "#93c5fd",
    "amber":          "#d97706",
    "slate":          "#475569",
    "surface_border": "1px solid #e2e8f0",
}

# Six visually distinct colours for multi-series charts (Tailwind-based)
CHART_COLORS = [
    "#2563eb",  # blue-600
    "#7c3aed",  # violet-600
    "#0e7490",  # cyan-700
    "#059669",  # emerald-600
    "#d97706",  # amber-600
    "#e11d48",  # rose-600
]

_TOOLTIP_STYLE = {
    "background": "white",
    "border": "1px solid #e2e8f0",
    "borderRadius": "10px",
    "fontSize": "13px",
    "boxShadow": "0 4px 12px rgba(0,0,0,.1)",
}


# ─── State ──────────────────────────────────────────────────────────────────

class DashboardState(rx.State):
    # ── Global filters ────────────────────────────────────────
    selected_timeframe: str = DEFAULT_TIMEFRAME
    selected_show: str = ALL_SHOWS
    selected_guest_category: str = ALL_CATEGORIES

    # ── Meta / refresh ────────────────────────────────────────
    bundle_refresh_nonce: int = 0
    bundle_loaded_at: str = INITIAL_LOADED_AT
    data_updated_at: str = INITIAL_DATA_UPDATED_AT

    # ── Lazy-load flags for heavy BERTopic iframes ─────────────
    load_topics_visualization: bool = False
    load_topics_over_time: bool = False
    load_topics_hierarchy: bool = False
    load_topics_barchart: bool = False

    # ── Person search ──────────────────────────────────────────
    person_search_query: str = ""
    selected_guest_name: str = ""

    # ── Filter actions ─────────────────────────────────────────
    def set_selected_timeframe(self, value: str) -> None:
        self.selected_timeframe = value

    def set_selected_show(self, value: str) -> None:
        self.selected_show = value

    def set_selected_guest_category(self, value: str) -> None:
        self.selected_guest_category = value

    def reset_filters(self) -> None:
        self.selected_timeframe = DEFAULT_TIMEFRAME
        self.selected_show = ALL_SHOWS
        self.selected_guest_category = ALL_CATEGORIES

    def show_all_shows(self) -> None:
        self.selected_show = ALL_SHOWS

    def show_all_categories(self) -> None:
        self.selected_guest_category = ALL_CATEGORIES

    def choose_show(self, value: str) -> None:
        self.selected_show = value

    def choose_category(self, value: str) -> None:
        self.selected_guest_category = value

    # ── Lazy-load toggles ──────────────────────────────────────
    def enable_topics_visualization(self) -> None:
        self.load_topics_visualization = True

    def enable_topics_over_time(self) -> None:
        self.load_topics_over_time = True

    def enable_topics_hierarchy(self) -> None:
        self.load_topics_hierarchy = True

    def enable_topics_barchart(self) -> None:
        self.load_topics_barchart = True

    # ── Data refresh ───────────────────────────────────────────
    def refresh_data(self) -> None:
        global BUNDLE, NETWORK_ASSETS
        BUNDLE = load_dashboard_bundle()
        NETWORK_ASSETS = ensure_network_assets()
        self.bundle_refresh_nonce += 1
        self.bundle_loaded_at = current_timestamp_label()
        self.data_updated_at = data_updated_at_label()

    # ── Person search actions ──────────────────────────────────
    def set_person_search_query(self, value: str) -> None:
        self.person_search_query = value
        self.selected_guest_name = ""

    def select_guest(self, name: str) -> None:
        self.selected_guest_name = name

    def clear_guest_selection(self) -> None:
        self.selected_guest_name = ""

    # ── Computed: filter state ─────────────────────────────────

    @rx.var(cache=True)
    def filtered_shows_count(self) -> str:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return build_episode_count(fs)

    @rx.var(cache=True)
    def filtered_guests_count(self) -> str:
        _ = self.bundle_refresh_nonce
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, self.selected_guest_category
        )
        return build_unique_guest_count(fg)

    @rx.var(cache=True)
    def range_label(self) -> str:
        _ = self.bundle_refresh_nonce
        fs, start, end = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        if fs.empty and start is None:
            return "–"
        return format_date_range(start, end)

    @rx.var(cache=True)
    def top_topic_label(self) -> str:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return compact_label(build_top_topic_label(fs), 56)

    @rx.var(cache=True)
    def top_guest_label(self) -> str:
        _ = self.bundle_refresh_nonce
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, self.selected_guest_category
        )
        return compact_label(build_top_guest_label(fg), 56)

    @rx.var(cache=True)
    def top_diversity_label(self) -> str:
        _ = self.bundle_refresh_nonce
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, self.selected_guest_category
        )
        return compact_label(build_top_diversity_label(fg), 56)

    @rx.var(cache=True)
    def scope_description(self) -> str:
        _ = self.bundle_refresh_nonce
        return " / ".join(p for p in [self.selected_show, self.selected_timeframe] if p)

    @rx.var(cache=True)
    def guest_scope_description(self) -> str:
        _ = self.bundle_refresh_nonce
        return " / ".join(
            p for p in [self.selected_show, self.selected_timeframe, self.selected_guest_category] if p
        )

    @rx.var(cache=True)
    def filter_feedback(self) -> str:
        _ = self.bundle_refresh_nonce
        return (
            f"Zeige {self.filtered_shows_count} von {self.total_episode_count} Episoden "
            f"und {self.filtered_guests_count} von {self.total_guest_count} Gaesten."
        )

    # ── Computed: chart data ───────────────────────────────────

    @rx.var(cache=True)
    def timeline_rows(self) -> list[dict[str, int | str]]:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return build_timeline_rows(fs)

    @rx.var(cache=True)
    def has_timeline(self) -> bool:
        return len(self.timeline_rows) > 0

    @rx.var(cache=True)
    def show_mix_rows(self) -> list[dict[str, int | str]]:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return build_show_mix_rows(fs)

    @rx.var(cache=True)
    def topic_rows(self) -> list[dict[str, int | str]]:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return build_topic_rows(fs)

    @rx.var(cache=True)
    def has_topics(self) -> bool:
        return len(self.topic_rows) > 0

    @rx.var(cache=True)
    def topics_over_time_rows(self) -> list[dict[str, int | str]]:
        """Wide-format rows: period + t0..t5 — for multi-line chart."""
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        rows, _ = build_topics_over_time_data(fs, n_topics=6)
        return rows

    @rx.var(cache=True)
    def topics_over_time_labels(self) -> list[str]:
        """Exactly 6 display labels (empty string = unused slot)."""
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        _, labels = build_topics_over_time_data(fs, n_topics=6)
        return labels

    @rx.var(cache=True)
    def has_topics_over_time(self) -> bool:
        return len(self.topics_over_time_rows) > 0

    @rx.var(cache=True)
    def treemap_data(self) -> list[dict[str, str | int]]:
        """Flat topic list for Recharts Treemap — keys: name (str), size (int)."""
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        raw = build_treemap_data(fs)
        # Drop 'fill' so the dict type stays str | int — fills live in treemap_fills
        return [{"name": str(item["name"]), "size": int(item["size"])} for item in raw]

    @rx.var(cache=True)
    def treemap_fills(self) -> list[str]:
        """Per-cell fill colours in the same order as treemap_data.
        Kept as list[str] so Cell.fill gets the correct inferred type."""
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return [str(item["fill"]) for item in build_treemap_data(fs)]

    @rx.var(cache=True)
    def has_treemap_data(self) -> bool:
        return len(self.treemap_data) > 0

    @rx.var(cache=True)
    def treemap_legend(self) -> list[dict[str, str]]:
        """Thematic group colour legend — [{"color": str, "label": str}, ...]."""
        return build_treemap_legend()

    @rx.var(cache=True)
    def top_guest_pairs(self) -> list[dict[str, str | int]]:
        """Top 15 guest pairs by shared-episode count — filter-reactive."""
        _ = self.bundle_refresh_nonce
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, ALL_CATEGORIES
        )
        return build_top_guest_pairs(fg, n=15)

    @rx.var(cache=True)
    def has_top_guest_pairs(self) -> bool:
        return len(self.top_guest_pairs) > 0

    @rx.var(cache=True)
    def category_rows(self) -> list[dict[str, int | str]]:
        _ = self.bundle_refresh_nonce
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, self.selected_guest_category
        )
        return build_category_rows(fg)

    @rx.var(cache=True)
    def has_categories(self) -> bool:
        return len(self.category_rows) > 0

    @rx.var(cache=True)
    def appearance_rows(self) -> list[list[str]]:
        _ = self.bundle_refresh_nonce
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, self.selected_guest_category
        )
        return build_appearance_rows(fg, BUNDLE.guest_metadata)

    @rx.var(cache=True)
    def diversity_rows(self) -> list[list[str]]:
        _ = self.bundle_refresh_nonce
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, self.selected_guest_category
        )
        return build_topic_diversity_rows(fg, BUNDLE.guest_metadata)

    @rx.var(cache=True)
    def show_cards(self) -> list[list[str]]:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        return build_show_cards(fs)

    @rx.var(cache=True)
    def has_guest_rows(self) -> bool:
        return len(self.appearance_rows) > 0

    @rx.var(cache=True)
    def has_any_data(self) -> bool:
        _ = self.bundle_refresh_nonce
        fs, _, _ = filter_shows(BUNDLE.shows, self.selected_timeframe, self.selected_show)
        fg, _, _ = filter_guests(
            BUNDLE.guests, self.selected_timeframe, self.selected_show, self.selected_guest_category
        )
        return not fs.empty or not fg.empty

    @rx.var(cache=True)
    def total_episode_count(self) -> str:
        _ = self.bundle_refresh_nonce
        return build_episode_count(BUNDLE.shows)

    @rx.var(cache=True)
    def total_guest_count(self) -> str:
        _ = self.bundle_refresh_nonce
        return build_unique_guest_count(BUNDLE.guests)

    # ── Computed: person search ────────────────────────────────

    @rx.var(cache=True)
    def person_search_rows(self) -> list[list[str]]:
        _ = self.bundle_refresh_nonce
        q = self.person_search_query.strip()
        if not q:
            return []
        guests = BUNDLE.guests
        if guests.empty or "name" not in guests.columns:
            return []
        mask = guests["name"].str.lower().str.contains(q.lower(), na=False)
        matching = guests[mask].dropna(subset=["name"])
        if matching.empty:
            return []
        appearances = (
            matching.groupby("name")
            .size()
            .reset_index(name="appearances")
            .sort_values("appearances", ascending=False)
        )
        appearances = appearances.merge(BUNDLE.guest_metadata, on="name", how="left")
        rows: list[list[str]] = []
        for row in appearances.head(25).itertuples(index=False):
            rows.append([
                str(row.name),
                compact_label(getattr(row, "primary_party", None) or "–", 20),
                compact_label(getattr(row, "known_roles", None) or "–", 42),
                format_count(getattr(row, "appearances", 0)),
            ])
        return rows

    @rx.var(cache=True)
    def has_search_results(self) -> bool:
        return len(self.person_search_rows) > 0

    @rx.var(cache=True)
    def search_is_empty(self) -> bool:
        return self.person_search_query.strip() == ""

    @rx.var(cache=True)
    def guest_detail_header(self) -> list[str]:
        """[name, party, roles, total_appearances, date_range]"""
        _ = self.bundle_refresh_nonce
        name = self.selected_guest_name
        if not name:
            return []
        guests = BUNDLE.guests
        if guests.empty or "name" not in guests.columns:
            return []
        rows = guests[guests["name"] == name]
        if rows.empty:
            return []
        meta = BUNDLE.guest_metadata
        meta_row = meta[meta["name"] == name]
        party = (
            str(meta_row.iloc[0]["primary_party"])
            if not meta_row.empty and pd.notna(meta_row.iloc[0]["primary_party"])
            else "–"
        )
        roles = (
            str(meta_row.iloc[0]["known_roles"])
            if not meta_row.empty and pd.notna(meta_row.iloc[0]["known_roles"])
            else "–"
        )
        total = format_count(len(rows))
        date_range = "–"
        if "date" in rows.columns:
            dates = rows["date"].dropna()
            if not dates.empty:
                date_range = format_date_range(dates.min(), dates.max())
        return [name, party, roles, total, date_range]

    @rx.var(cache=True)
    def guest_detail_appearances(self) -> list[list[str]]:
        """List of [date, show, topic] rows for the selected guest."""
        _ = self.bundle_refresh_nonce
        name = self.selected_guest_name
        if not name:
            return []
        guests = BUNDLE.guests
        if guests.empty or "name" not in guests.columns:
            return []
        rows = guests[guests["name"] == name].copy()
        if rows.empty:
            return []
        if "date" in rows.columns:
            rows = rows.sort_values("date", ascending=False)
        result: list[list[str]] = []
        for row in rows.head(30).itertuples(index=False):
            date_val = getattr(row, "date", None)
            date_str = (
                pd.Timestamp(date_val).strftime("%d.%m.%Y")
                if date_val is not None and pd.notna(date_val)
                else "–"
            )
            show = compact_label(getattr(row, "show_display", None) or "–", 22)
            topic = compact_label(format_topic_text(getattr(row, "topic_label", None)), 44)
            result.append([date_str, show, topic])
        return result

    @rx.var(cache=True)
    def has_guest_detail(self) -> bool:
        return len(self.guest_detail_header) > 0


# ─── Shared components ───────────────────────────────────────────────────────


def navbar(active: str = "") -> rx.Component:
    """Sticky top navigation bar with four section links."""
    def _link(label: str, href: str, key: str) -> rx.Component:
        cls = "nav-link nav-link-active" if active == key else "nav-link"
        return rx.link(label, href=href, class_name=cls)

    return rx.box(
        rx.box(
            # Brand
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
            # Links
            rx.flex(
                _link("Übersicht", "/", "overview"),
                _link("Gäste", "/gaeste", "gaeste"),
                _link("Themen", "/themen", "themen"),
                _link("Personensuche", "/personensuche", "personensuche"),
                class_name="nav-links",
            ),
            class_name="nav-inner",
        ),
        class_name="navbar",
    )


def section_intro(eyebrow: str, title: str, description: str) -> rx.Component:
    return rx.flex(
        rx.text(eyebrow, class_name="section-eyebrow"),
        rx.heading(title, size="6", class_name="section-title"),
        rx.text(description, class_name="section-copy"),
        direction="column",
        spacing="2",
        width="100%",
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> rx.Component:
    return rx.flex(
        rx.text(eyebrow, class_name="page-header-eyebrow"),
        rx.heading(title, size="8", class_name="page-header-title"),
        rx.text(subtitle, class_name="page-header-sub"),
        direction="column",
        spacing="2",
        width="100%",
    )


def stat_card(title: str, value: rx.Var | str, note: str) -> rx.Component:
    return rx.card(
        rx.flex(
            rx.text(title, class_name="stat-title"),
            rx.text(value, class_name="stat-value"),
            rx.text(note, class_name="stat-note"),
            direction="column",
            spacing="2",
            width="100%",
        ),
        class_name="stat-card",
        size="3",
        variant="surface",
    )


def chart_card(title: str, description: str, chart: rx.Component) -> rx.Component:
    return rx.card(
        rx.flex(
            rx.heading(title, size="4", class_name="panel-title"),
            rx.text(description, class_name="panel-copy"),
            rx.box(chart, width="100%", min_height="280px"),
            direction="column",
            spacing="3",
            width="100%",
        ),
        class_name="panel-card",
        size="3",
        variant="surface",
    )


def empty_state(message: str, min_h: str = "280px") -> rx.Component:
    return rx.flex(
        rx.text(message, class_name="empty-copy"),
        align="center",
        justify="center",
        width="100%",
        min_height=min_h,
        class_name="empty-state",
    )


def filter_panel() -> rx.Component:
    return rx.card(
        rx.flex(
            rx.flex(
                rx.flex(
                    rx.text("Zeitraum", class_name="filter-label"),
                    rx.select(
                        items=TIMEFRAME_LABELS,
                        value=DashboardState.selected_timeframe,
                        on_change=DashboardState.set_selected_timeframe,
                        variant="surface",
                        width="100%",
                    ),
                    direction="column",
                    spacing="1",
                    width="100%",
                ),
                rx.flex(
                    rx.text("Sendung", class_name="filter-label"),
                    rx.select(
                        items=BUNDLE.show_options,
                        value=DashboardState.selected_show,
                        on_change=DashboardState.set_selected_show,
                        variant="surface",
                        width="100%",
                    ),
                    direction="column",
                    spacing="1",
                    width="100%",
                ),
                rx.flex(
                    rx.text("Gastkategorie", class_name="filter-label"),
                    rx.select(
                        items=BUNDLE.category_options,
                        value=DashboardState.selected_guest_category,
                        on_change=DashboardState.set_selected_guest_category,
                        variant="surface",
                        width="100%",
                    ),
                    direction="column",
                    spacing="1",
                    width="100%",
                ),
                class_name="filter-grid",
            ),
            rx.flex(
                rx.text(DashboardState.filter_feedback, class_name="filter-feedback"),
                rx.button(
                    "Filter zurücksetzen",
                    on_click=DashboardState.reset_filters,
                    variant="ghost",
                    class_name="filter-reset",
                    size="2",
                ),
                justify="between",
                align="center",
                wrap="wrap",
                gap="3",
                width="100%",
            ),
            direction="column",
            spacing="4",
            width="100%",
        ),
        class_name="filter-panel",
        size="3",
        variant="surface",
    )


def show_filter_panel() -> rx.Component:
    """Filter panel with only timeframe + show (for Themen page)."""
    return rx.card(
        rx.flex(
            rx.flex(
                rx.flex(
                    rx.text("Zeitraum", class_name="filter-label"),
                    rx.select(
                        items=TIMEFRAME_LABELS,
                        value=DashboardState.selected_timeframe,
                        on_change=DashboardState.set_selected_timeframe,
                        variant="surface",
                        width="100%",
                    ),
                    direction="column",
                    spacing="1",
                    width="100%",
                ),
                rx.flex(
                    rx.text("Sendung", class_name="filter-label"),
                    rx.select(
                        items=BUNDLE.show_options,
                        value=DashboardState.selected_show,
                        on_change=DashboardState.set_selected_show,
                        variant="surface",
                        width="100%",
                    ),
                    direction="column",
                    spacing="1",
                    width="100%",
                ),
                class_name="filter-grid",
            ),
            rx.flex(
                rx.text(DashboardState.filter_feedback, class_name="filter-feedback"),
                rx.button(
                    "Zurücksetzen",
                    on_click=DashboardState.reset_filters,
                    variant="ghost",
                    class_name="filter-reset",
                    size="2",
                ),
                justify="between",
                align="center",
                wrap="wrap",
                gap="3",
                width="100%",
            ),
            direction="column",
            spacing="4",
            width="100%",
        ),
        class_name="filter-panel",
        size="3",
        variant="surface",
    )


def ranking_row(row: list[str]) -> rx.Component:
    return rx.table.row(
        rx.table.cell(row[0], class_name="table-name"),
        rx.table.cell(row[1], class_name="table-cell"),
        rx.table.cell(row[2], class_name="table-cell"),
        rx.table.cell(row[3], class_name="table-value", align="right"),
    )


def ranking_table(
    title: str,
    description: str,
    headers: tuple[str, str, str, str],
    rows: rx.Var,
) -> rx.Component:
    return rx.card(
        rx.flex(
            rx.heading(title, size="4", class_name="panel-title"),
            rx.text(description, class_name="panel-copy"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell(headers[0]),
                        rx.table.column_header_cell(headers[1]),
                        rx.table.column_header_cell(headers[2]),
                        rx.table.column_header_cell(headers[3], align="right"),
                    )
                ),
                rx.table.body(rx.foreach(rows, ranking_row)),
                variant="surface",
                size="2",
                width="100%",
            ),
            direction="column",
            spacing="3",
            width="100%",
        ),
        class_name="panel-card",
        size="3",
        variant="surface",
    )


def show_card(card: list[str]) -> rx.Component:
    return rx.card(
        rx.flex(
            rx.text(card[0], class_name="show-card-title"),
            rx.text(card[1], class_name="show-card-value"),
            rx.text(card[2], class_name="show-card-note"),
            direction="column",
            spacing="2",
        ),
        class_name="show-card",
        size="3",
        variant="surface",
    )


def quick_filter_show_button(row: dict[str, int | str]) -> rx.Component:
    return rx.button(
        rx.text(row["show_short"]),
        rx.text("(", row["episodes"], ")", class_name="filter-chip-count"),
        on_click=DashboardState.choose_show(row["show_display"]),
        variant="soft",
        class_name="filter-chip",
        size="2",
    )


def quick_filter_category_button(row: dict[str, int | str]) -> rx.Component:
    return rx.button(
        rx.text(row["category_short"]),
        rx.text("(", row["guests"], ")", class_name="filter-chip-count"),
        on_click=DashboardState.choose_category(row["category_display"]),
        variant="soft",
        class_name="filter-chip",
        size="2",
    )


def quick_filter_row(title: str, rows: rx.Var, all_label: str, reset_handler, renderer) -> rx.Component:
    return rx.flex(
        rx.text(title, class_name="quick-filter-title"),
        rx.flex(
            rx.button(
                all_label,
                on_click=reset_handler,
                variant="outline",
                class_name="filter-chip",
                size="2",
            ),
            rx.foreach(rows, renderer),
            wrap="wrap",
            spacing="2",
            width="100%",
        ),
        direction="column",
        spacing="2",
        width="100%",
    )


# ─── Charts ──────────────────────────────────────────────────────────────────


def timeline_chart() -> rx.Component:
    return rx.cond(
        DashboardState.has_timeline,
        rx.recharts.line_chart(
            rx.recharts.cartesian_grid(
                stroke_dasharray="4 4",
                stroke="rgba(0,0,0,.06)",
                vertical=False,
            ),
            rx.recharts.x_axis(
                data_key="period",
                tick_line=False,
                axis_line=False,
                min_tick_gap=28,
                tick={"fill": "#94a3b8", "fontSize": 12},
            ),
            rx.recharts.y_axis(
                tick_line=False,
                axis_line=False,
                allow_decimals=False,
                tick={"fill": "#94a3b8", "fontSize": 12},
            ),
            rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
            rx.recharts.line(
                data_key="episodes",
                type_="monotone",
                stroke=PALETTE["blue"],
                stroke_width=2.5,
                dot=False,
                active_dot={"r": 5, "fill": PALETTE["blue"]},
                name="Episoden",
            ),
            data=DashboardState.timeline_rows,
            width="100%",
            height=280,
            margin={"top": 8, "right": 20, "left": 0, "bottom": 0},
        ),
        empty_state("Für diese Auswahl liegen keine Episodendaten vor."),
    )


def show_mix_chart() -> rx.Component:
    return rx.cond(
        DashboardState.has_timeline,
        rx.recharts.bar_chart(
            rx.recharts.cartesian_grid(
                stroke_dasharray="4 4",
                stroke="rgba(0,0,0,.06)",
                vertical=False,
            ),
            rx.recharts.x_axis(
                data_key="show_short",
                tick_line=False,
                axis_line=False,
                tick={"fill": "#94a3b8", "fontSize": 12},
            ),
            rx.recharts.y_axis(
                tick_line=False,
                axis_line=False,
                allow_decimals=False,
                tick={"fill": "#94a3b8", "fontSize": 12},
            ),
            rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
            rx.recharts.bar(
                data_key="episodes",
                fill=PALETTE["blue"],
                radius=[6, 6, 0, 0],
                name="Episoden",
                fill_opacity=0.85,
            ),
            data=DashboardState.show_mix_rows,
            width="100%",
            height=280,
            margin={"top": 8, "right": 20, "left": 0, "bottom": 0},
        ),
        empty_state("Keine Sendungsdaten im gewählten Zeitraum."),
    )


def topic_chart() -> rx.Component:
    """Horizontal bar chart for topics — avoids the angled-label problem."""
    return rx.cond(
        DashboardState.has_topics,
        rx.recharts.bar_chart(
            rx.recharts.cartesian_grid(
                stroke_dasharray="4 4",
                stroke="rgba(0,0,0,.06)",
                horizontal=False,
            ),
            rx.recharts.x_axis(
                type_="number",
                tick_line=False,
                axis_line=False,
                allow_decimals=False,
                tick={"fill": "#94a3b8", "fontSize": 12},
            ),
            rx.recharts.y_axis(
                type_="category",
                data_key="topic_short",
                tick_line=False,
                axis_line=False,
                width=180,
                tick={"fill": "#334155", "fontSize": 12},
            ),
            rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
            rx.recharts.bar(
                data_key="episodes",
                fill=PALETTE["blue"],
                radius=[0, 6, 6, 0],
                name="Episoden",
                fill_opacity=0.9,
            ),
            data=DashboardState.topic_rows,
            layout="vertical",
            width="100%",
            height=320,
            margin={"top": 4, "right": 20, "left": 0, "bottom": 4},
        ),
        empty_state("Keine klassifizierten Themen für diese Auswahl."),
    )


def category_chart() -> rx.Component:
    """Horizontal bar chart for guest categories."""
    return rx.cond(
        DashboardState.has_categories,
        rx.recharts.bar_chart(
            rx.recharts.cartesian_grid(
                stroke_dasharray="4 4",
                stroke="rgba(0,0,0,.06)",
                horizontal=False,
            ),
            rx.recharts.x_axis(
                type_="number",
                tick_line=False,
                axis_line=False,
                allow_decimals=False,
                tick={"fill": "#94a3b8", "fontSize": 12},
            ),
            rx.recharts.y_axis(
                type_="category",
                data_key="category_short",
                tick_line=False,
                axis_line=False,
                width=160,
                tick={"fill": "#334155", "fontSize": 12},
            ),
            rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
            rx.recharts.bar(
                data_key="guests",
                fill=PALETTE["slate"],
                radius=[0, 6, 6, 0],
                name="Gäste",
                fill_opacity=0.8,
            ),
            data=DashboardState.category_rows,
            layout="vertical",
            width="100%",
            height=320,
            margin={"top": 4, "right": 20, "left": 0, "bottom": 4},
        ),
        empty_state("Der aktuelle Gastfilter liefert keine Kategorien."),
    )


# ─── Treemap (Themen-Hierarchie) ─────────────────────────────────────────────


def _treemap_legend_item(entry: dict[str, str]) -> rx.Component:
    """One colour swatch + group label for the treemap legend."""
    return rx.flex(
        rx.box(
            width="12px",
            height="12px",
            border_radius="3px",
            background_color=entry["color"],
            flex_shrink="0",
        ),
        rx.text(
            entry["label"],
            font_size="12px",
            color="#475569",
            white_space="nowrap",
        ),
        align="center",
        gap="6px",
    )


def topic_treemap() -> rx.Component:
    """
    Recharts Treemap showing all topics sized by episode count.
    Topics are colour-coded by thematic group, with a legend below.
    """
    return rx.cond(
        DashboardState.has_treemap_data,
        rx.box(
            # The chart itself
            rx.recharts.treemap(
                rx.foreach(
                    DashboardState.treemap_fills,
                    lambda fill_color: rx.recharts.cell(fill=fill_color),
                ),
                rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
                data=DashboardState.treemap_data,
                data_key="size",
                name_key="name",
                width="100%",
                height=400,
                aspect_ratio=2,
                is_animation_active=True,
                animation_duration=500,
            ),
            # Group legend below the chart
            rx.flex(
                rx.foreach(
                    DashboardState.treemap_legend,
                    _treemap_legend_item,
                ),
                flex_wrap="wrap",
                gap="12px",
                padding_top="12px",
                padding_x="4px",
                justify="center",
            ),
            width="100%",
        ),
        empty_state("Keine Themen-Daten für diese Auswahl.", "260px"),
    )


# ─── Topics over time chart (custom, replaces BERTopic iframe) ───────────────


def _legend_item(color: str, label: rx.Var) -> rx.Component:
    """One coloured dot + label; hidden when label is empty string."""
    return rx.cond(
        label != "",
        rx.flex(
            rx.box(
                width="10px",
                height="10px",
                border_radius="3px",
                background=color,
                flex_shrink="0",
            ),
            rx.text(label, font_size="0.75rem", color="var(--text-lo)", line_height="1.3"),
            align="center",
            gap="6px",
        ),
        rx.box(),  # invisible placeholder keeps layout stable
    )


def top_guest_pairs_chart() -> rx.Component:
    """
    Horizontal bar chart showing the top 15 guest pairs by shared-episode
    count — the strongest bilateral relationships in the dataset.
    Filter-reactive (responds to timeframe + show).
    """
    return rx.cond(
        DashboardState.has_top_guest_pairs,
        rx.recharts.bar_chart(
            rx.recharts.bar(
                data_key="episodes",
                fill=CHART_COLORS[0],
                radius=4,
                label=False,
                is_animation_active=True,
                animation_duration=500,
            ),
            rx.recharts.x_axis(
                type_="number",
                tick_line=False,
                axis_line=False,
                tick={"fill": "#94a3b8", "fontSize": 11},
            ),
            rx.recharts.y_axis(
                type_="category",
                data_key="pair",
                width=260,
                tick_line=False,
                axis_line=False,
                tick={"fill": "#475569", "fontSize": 11},
            ),
            rx.recharts.cartesian_grid(
                stroke_dasharray="3 3",
                vertical=True,
                horizontal=False,
                stroke="#f1f5f9",
            ),
            rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
            data=DashboardState.top_guest_pairs,
            layout="vertical",
            width="100%",
            height=500,
            margin={"left": 0, "right": 24, "top": 8, "bottom": 8},
        ),
        empty_state("Keine Paarungsdaten für diese Auswahl.", "260px"),
    )


def topics_over_time_chart() -> rx.Component:
    """
    Smooth multi-line Recharts chart — top 6 topics over time.
    Replaces the cluttered BERTopic Plotly iframe.
    """
    labels = DashboardState.topics_over_time_labels

    return rx.cond(
        DashboardState.has_topics_over_time,
        rx.flex(
            # Chart
            rx.recharts.line_chart(
                rx.recharts.cartesian_grid(
                    stroke_dasharray="4 4",
                    stroke="rgba(0,0,0,.06)",
                    vertical=False,
                ),
                rx.recharts.x_axis(
                    data_key="period",
                    tick_line=False,
                    axis_line=False,
                    min_tick_gap=28,
                    tick={"fill": "#94a3b8", "fontSize": 12},
                ),
                rx.recharts.y_axis(
                    tick_line=False,
                    axis_line=False,
                    allow_decimals=False,
                    tick={"fill": "#94a3b8", "fontSize": 12},
                ),
                rx.recharts.graphing_tooltip(content_style=_TOOLTIP_STYLE),
                # Six smooth lines — name drives the tooltip label
                rx.recharts.line(
                    data_key="t0", type_="monotone",
                    stroke=CHART_COLORS[0], stroke_width=2,
                    dot=False, active_dot={"r": 4},
                    name=DashboardState.topics_over_time_labels[0],
                ),
                rx.recharts.line(
                    data_key="t1", type_="monotone",
                    stroke=CHART_COLORS[1], stroke_width=2,
                    dot=False, active_dot={"r": 4},
                    name=DashboardState.topics_over_time_labels[1],
                ),
                rx.recharts.line(
                    data_key="t2", type_="monotone",
                    stroke=CHART_COLORS[2], stroke_width=2,
                    dot=False, active_dot={"r": 4},
                    name=DashboardState.topics_over_time_labels[2],
                ),
                rx.recharts.line(
                    data_key="t3", type_="monotone",
                    stroke=CHART_COLORS[3], stroke_width=2,
                    dot=False, active_dot={"r": 4},
                    name=DashboardState.topics_over_time_labels[3],
                ),
                rx.recharts.line(
                    data_key="t4", type_="monotone",
                    stroke=CHART_COLORS[4], stroke_width=2,
                    dot=False, active_dot={"r": 4},
                    name=DashboardState.topics_over_time_labels[4],
                ),
                rx.recharts.line(
                    data_key="t5", type_="monotone",
                    stroke=CHART_COLORS[5], stroke_width=2,
                    dot=False, active_dot={"r": 4},
                    name=DashboardState.topics_over_time_labels[5],
                ),
                data=DashboardState.topics_over_time_rows,
                width="100%",
                height=320,
                margin={"top": 8, "right": 16, "left": 0, "bottom": 0},
            ),
            # Custom legend below chart
            rx.flex(
                _legend_item(CHART_COLORS[0], labels[0]),
                _legend_item(CHART_COLORS[1], labels[1]),
                _legend_item(CHART_COLORS[2], labels[2]),
                _legend_item(CHART_COLORS[3], labels[3]),
                _legend_item(CHART_COLORS[4], labels[4]),
                _legend_item(CHART_COLORS[5], labels[5]),
                wrap="wrap",
                gap="16px",
                width="100%",
                padding_top="8px",
            ),
            direction="column",
            spacing="3",
            width="100%",
        ),
        empty_state("Keine Themen-Zeitverlauf-Daten für diese Auswahl."),
    )


# ─── Network cards ────────────────────────────────────────────────────────────


def network_metric_tile(metric: dict[str, str]) -> rx.Component:
    return rx.box(
        rx.text(metric["label"], class_name="network-metric-label"),
        rx.text(metric["value"], class_name="network-metric-value"),
        class_name="network-metric-tile",
    )


def network_highlight_row(item: dict[str, str]) -> rx.Component:
    return rx.flex(
        rx.text(item["label"], class_name="network-highlight-label"),
        rx.text(item["value"], class_name="network-highlight-value"),
        direction="column",
        spacing="1",
        class_name="network-highlight-row",
    )


def network_preview_card(
    title: str,
    description: str,
    image_src: str | None,
    html_src: str | None,
    insight_badges: list[str] | None = None,
    metrics: list[dict[str, str]] | None = None,
    highlights: list[dict[str, str]] | None = None,
    scope_note: str | None = None,
    exported_at: str | None = None,
) -> rx.Component:
    badges = insight_badges or []
    metric_rows = metrics or []
    highlight_rows = highlights or []
    img = (
        rx.box(
            rx.image(src=image_src, alt=title, width="100%", object_fit="contain", class_name="network-image"),
            class_name="network-preview-surface",
        )
        if image_src
        else empty_state("Keine statische Vorschau verfügbar.", "360px")
    )
    iframe = (
        rx.box(
            rx.el.iframe(src=html_src, width="100%", loading="lazy", class_name="network-iframe"),
            class_name="network-preview-surface network-preview-surface-iframe",
        )
        if html_src
        else empty_state("Keine interaktive Ansicht gefunden.", "480px")
    )
    open_link = (
        rx.el.a(
            "Interaktive Ansicht separat öffnen →",
            href=html_src,
            target="_blank",
            rel="noopener noreferrer",
            class_name="network-link",
        )
        if html_src
        else rx.text("Kein HTML-Export vorhanden.", class_name="network-note")
    )
    card_children: list[rx.Component] = []
    if badges:
        card_children.append(
            rx.flex(
                *[rx.text(badge, class_name="network-badge") for badge in badges],
                wrap="wrap",
                spacing="2",
                width="100%",
            )
        )
    card_children.extend(
        [
            rx.heading(title, size="4", class_name="panel-title"),
            rx.text(description, class_name="panel-copy"),
        ]
    )
    if metric_rows:
        card_children.append(
            rx.grid(
                *[network_metric_tile(metric) for metric in metric_rows],
                columns="repeat(3, minmax(0, 1fr))",
                gap="3",
                width="100%",
                class_name="network-metrics-grid",
            )
        )
    if highlight_rows:
        card_children.append(
            rx.grid(
                *[network_highlight_row(item) for item in highlight_rows],
                columns="repeat(auto-fit, minmax(220px, 1fr))",
                gap="3",
                width="100%",
                class_name="network-highlights-grid",
            )
        )
    if scope_note or exported_at:
        meta_children: list[rx.Component] = []
        if scope_note:
            meta_children.append(rx.text(scope_note, class_name="network-scope-note"))
        if exported_at:
            meta_children.append(rx.text(f"Export: {exported_at}", class_name="network-export-note"))
        card_children.append(
            rx.flex(
                *meta_children,
                justify="between",
                align="center",
                wrap="wrap",
                gap="2",
                width="100%",
            )
        )
    card_children.append(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Statisch", value="static"),
                rx.tabs.trigger("Interaktiv", value="interactive"),
                class_name="network-tabs-list",
            ),
            rx.tabs.content(
                rx.flex(
                    img,
                    rx.text(
                        "PNG-Export fuer schnellen Strukturueberblick und lesbare Cluster.",
                        class_name="network-note",
                    ),
                    direction="column",
                    spacing="3",
                    width="100%",
                ),
                value="static",
            ),
            rx.tabs.content(
                rx.flex(
                    iframe,
                    rx.text(
                        "HTML-Ansicht fuer Hover, Zoom und Detailexploration im Vollgraphen.",
                        class_name="network-note",
                    ),
                    open_link,
                    direction="column",
                    spacing="3",
                    width="100%",
                ),
                value="interactive",
            ),
            default_value="static",
            width="100%",
        )
    )
    return rx.card(
        rx.flex(
            *card_children,
            direction="column",
            spacing="4",
            width="100%",
        ),
        class_name="panel-card network-card",
        size="3",
        variant="surface",
    )


def bertopic_embed_card(
    title: str,
    description: str,
    html_src: str | None,
    is_loaded: rx.Var,
    load_handler,
) -> rx.Component:
    content = (
        rx.cond(
            is_loaded,
            rx.flex(
                rx.el.iframe(src=html_src, width="100%", loading="lazy", class_name="network-iframe"),
                rx.el.a(
                    "Visualisierung separat öffnen →",
                    href=html_src,
                    target="_blank",
                    rel="noopener noreferrer",
                    class_name="network-link",
                ),
                direction="column",
                spacing="3",
                width="100%",
            ),
            rx.flex(
                rx.text(
                    "Schwere Visualisierung — wird erst auf Anfrage geladen.",
                    class_name="network-note",
                ),
                rx.button(
                    "Visualisierung laden",
                    on_click=load_handler,
                    variant="soft",
                    size="2",
                ),
                direction="column",
                spacing="3",
                align="start",
                width="100%",
            ),
        )
        if html_src
        else empty_state("HTML-Export nicht gefunden.")
    )
    return rx.card(
        rx.flex(
            rx.heading(title, size="4", class_name="panel-title"),
            rx.text(description, class_name="panel-copy"),
            content,
            direction="column",
            spacing="3",
            width="100%",
        ),
        class_name="panel-card",
        size="3",
        variant="surface",
    )


# ─── Page: Übersicht (/)) ─────────────────────────────────────────────────────


def overview_page() -> rx.Component:
    if BUNDLE.error is not None:
        return rx.box(
            navbar("overview"),
            rx.container(
                rx.card(
                    rx.flex(
                        rx.heading("Daten nicht verfügbar", size="7"),
                        rx.text(BUNDLE.error, class_name="section-copy"),
                        direction="column",
                        spacing="3",
                    ),
                    class_name="panel-card",
                    size="4",
                    variant="surface",
                ),
                max_width="900px",
                padding="48px 24px",
            ),
            class_name="app-shell",
            min_height="100vh",
        )

    return rx.box(
        navbar("overview"),
        rx.box(
            # ── Hero ──────────────────────────────────────────
            rx.flex(
                rx.flex(
                    rx.text("Politische Talkshow-Analyse", class_name="hero-kicker"),
                    rx.heading("Wer talkt mit wem?", size="9", class_name="hero-title"),
                    rx.text(
                        "Daten aus sechs deutschen Talkshows — analysiert nach Gästen, "
                        "Themen und Mustern. NLP-gestützt mit BERTopic-Modell.",
                        class_name="hero-copy",
                    ),
                    rx.flex(
                        rx.flex(
                            rx.text("Datenstand:", class_name="hero-meta"),
                            rx.text(DashboardState.data_updated_at, class_name="hero-meta"),
                            gap="6px",
                            align="center",
                        ),
                        rx.flex(
                            rx.text("Geladen:", class_name="hero-meta"),
                            rx.text(DashboardState.bundle_loaded_at, class_name="hero-meta"),
                            gap="6px",
                            align="center",
                        ),
                        rx.button(
                            "Neu laden",
                            on_click=DashboardState.refresh_data,
                            class_name="hero-refresh",
                            variant="ghost",
                            size="2",
                        ),
                        wrap="wrap",
                        gap="12px",
                        align="center",
                        width="100%",
                    ),
                    direction="column",
                    spacing="4",
                    width="100%",
                    max_width="620px",
                ),
                # ── Stat cards ────────────────────────────────
                rx.grid(
                    stat_card("Episoden", DashboardState.filtered_shows_count, "im Fokus"),
                    stat_card("Gäste", DashboardState.filtered_guests_count, "unique Personen"),
                    stat_card("Zeitraum", DashboardState.range_label, "abgedeckt"),
                    stat_card("Top-Thema", DashboardState.top_topic_label, "dominiert"),
                    columns="repeat(2, 1fr)",
                    spacing="3",
                    width="100%",
                    max_width="480px",
                ),
                class_name="hero-grid",
                width="100%",
            ),

            # ── Timeline ───────────────────────────────────────
            chart_card(
                "Episoden im Zeitverlauf",
                "Sendehäufigkeit nach Monat oder Jahr — je nach gewähltem Zeitrahmen.",
                timeline_chart(),
            ),

            # ── Show mix + Show cards ─────────────────────────
            rx.grid(
                chart_card(
                    "Sendungsmix",
                    "Verteilung der Episoden über alle Talkshows im gewählten Ausschnitt.",
                    rx.flex(
                        show_mix_chart(),
                        quick_filter_row(
                            "Schnell filtern nach Sendung",
                            DashboardState.show_mix_rows,
                            "Alle Sendungen",
                            DashboardState.show_all_shows,
                            quick_filter_show_button,
                        ),
                        direction="column",
                        spacing="3",
                        width="100%",
                    ),
                ),
                rx.card(
                    rx.flex(
                        rx.heading("Sendungs-Coverage", size="4", class_name="panel-title"),
                        rx.text(
                            "Episodenanzahl und erfasster Zeitraum je Talkshow.",
                            class_name="panel-copy",
                        ),
                        rx.grid(
                            rx.foreach(DashboardState.show_cards, show_card),
                            columns="repeat(auto-fit, minmax(160px, 1fr))",
                            gap="3",
                            width="100%",
                        ),
                        direction="column",
                        spacing="3",
                        width="100%",
                    ),
                    class_name="panel-card",
                    size="3",
                    variant="surface",
                ),
                columns="repeat(auto-fit, minmax(420px, 1fr))",
                gap="4",
                width="100%",
            ),

            # ── CTA links ──────────────────────────────────────
            rx.flex(
                rx.link(
                    rx.button("Gäste-Übersicht →", variant="soft", size="3"),
                    href="/gaeste",
                ),
                rx.link(
                    rx.button("Themen & Netzwerke →", variant="soft", size="3"),
                    href="/themen",
                ),
                rx.link(
                    rx.button("Personensuche →", variant="soft", size="3"),
                    href="/personensuche",
                ),
                wrap="wrap",
                gap="3",
                width="100%",
            ),

            # ── Footer ────────────────────────────────────────
            rx.text(
                "Wer talkt mit wem? — Talkshow-Daten von fernsehserien.de · "
                "NLP via BERTopic & sentence-transformers · Bernstein Analytics",
                class_name="footer-note",
                width="100%",
            ),

            class_name="page-container",
            display="flex",
            flex_direction="column",
            gap="32px",
            width="100%",
        ),
        class_name="app-shell",
        min_height="100vh",
        width="100%",
    )


# ─── Page: Gäste (/gaeste) ───────────────────────────────────────────────────


def gaeste_page() -> rx.Component:
    return rx.box(
        navbar("gaeste"),
        rx.box(
            page_header(
                "Gäste-Analyse",
                "Gäste-Übersicht",
                "Wer tritt am häufigsten auf? Welche Kategorien dominieren? "
                "Filter nach Zeitraum, Sendung und Gastsegment.",
            ),
            filter_panel(),
            # ── Highlights ────────────────────────────────────
            rx.grid(
                stat_card("Häufigster Gast", DashboardState.top_guest_label, "meiste Auftritte"),
                stat_card("Größtes Profil", DashboardState.top_diversity_label, "meiste Themen"),
                stat_card("Gäste gesamt", DashboardState.filtered_guests_count, "im Fokus"),
                stat_card("Episoden", DashboardState.filtered_shows_count, "im Zeitraum"),
                columns="repeat(auto-fit, minmax(200px, 1fr))",
                gap="4",
                width="100%",
            ),
            # ── Ranking tables ─────────────────────────────────
            rx.cond(
                DashboardState.has_guest_rows,
                rx.grid(
                    ranking_table(
                        "Häufigste Auftritte",
                        "Wiederkehrende Gesichter im gewählten Filterraum.",
                        ("Name", "Partei", "Rolle", "Auftritte"),
                        DashboardState.appearance_rows,
                    ),
                    ranking_table(
                        "Größte Themenbreite",
                        "Gäste mit der höchsten Zahl verschiedener Themencluster.",
                        ("Name", "Partei", "Rolle", "Themen"),
                        DashboardState.diversity_rows,
                    ),
                    columns="repeat(auto-fit, minmax(400px, 1fr))",
                    gap="4",
                    width="100%",
                ),
                empty_state("Keine Gäste für diese Filterkombination.", "180px"),
            ),
            # ── Category chart ─────────────────────────────────
            chart_card(
                "Gäste nach Kategorie",
                "Einzigartige Gäste nach ihrer primären Zuordnung im Datensatz.",
                rx.flex(
                    category_chart(),
                    quick_filter_row(
                        "Schnell filtern nach Kategorie",
                        DashboardState.category_rows,
                        "Alle Kategorien",
                        DashboardState.show_all_categories,
                        quick_filter_category_button,
                    ),
                    direction="column",
                    spacing="3",
                    width="100%",
                ),
            ),
            class_name="page-container",
            display="flex",
            flex_direction="column",
            gap="32px",
            width="100%",
        ),
        class_name="app-shell",
        min_height="100vh",
        width="100%",
    )


# ─── Page: Themen (/themen) ───────────────────────────────────────────────────


def themen_page() -> rx.Component:
    guest_network = NETWORK_ASSETS.get("guest_network", {})
    topic_network = NETWORK_ASSETS.get("topic_network", {})
    topics_visualization = NETWORK_ASSETS.get("topics_visualization", {})
    topics_over_time = NETWORK_ASSETS.get("topics_over_time", {})
    topics_hierarchy = NETWORK_ASSETS.get("topics_hierarchy", {})
    topics_barchart = NETWORK_ASSETS.get("topics_barchart", {})

    return rx.box(
        navbar("themen"),
        rx.box(
            page_header(
                "Themen-Analyse",
                "Themen & Netzwerke",
                "Welche Themen dominierten die Diskussion? Wie hängen Gäste und Themen zusammen?",
            ),
            show_filter_panel(),
            # ── Top-Themen Chart ───────────────────────────────
            chart_card(
                "Top-Themen",
                "Die stärksten Themencluster im gewählten Datensatz — identifiziert via BERTopic.",
                topic_chart(),
            ),
            # ── Intensivste Gäste-Beziehungen (filter-reactive) ────────────
            chart_card(
                "Intensivste Gäste-Beziehungen",
                "Top 15 Gäste-Paare nach Anzahl gemeinsamer Auftritte in derselben Episode — "
                "reagiert auf Zeitraum- und Sendungsfilter.",
                top_guest_pairs_chart(),
            ),
            # ── Networks ──────────────────────────────────────
            section_intro(
                "Netzwerke",
                "Ko-Auftritte und Themenbeziehungen",
                "Vollständige Netzwerk-Exporte des Gesamtdatensatzes — "
                "Klicken oder Hover für Details, Suche im Suchfeld.",
            ),
            rx.flex(
                network_preview_card(
                    str(guest_network.get("title", "Gäste-Netzwerk")),
                    str(guest_network.get("description", "")),
                    guest_network.get("image_src") if guest_network.get("has_image") else None,
                    guest_network.get("html_src") if guest_network.get("has_html") else None,
                    guest_network.get("insight_badges"),
                    None,
                    None,
                    guest_network.get("scope_note"),
                    guest_network.get("exported_at"),
                ),
                network_preview_card(
                    str(topic_network.get("title", "Themen-Netzwerk")),
                    str(topic_network.get("description", "")),
                    topic_network.get("image_src") if topic_network.get("has_image") else None,
                    topic_network.get("html_src") if topic_network.get("has_html") else None,
                    topic_network.get("insight_badges"),
                    None,
                    None,
                    topic_network.get("scope_note"),
                    topic_network.get("exported_at"),
                ),
                direction="column",
                gap="4",
                width="100%",
            ),
            # ── Themen im Zeitverlauf (custom chart) ────────────
            chart_card(
                "Themen im Zeitverlauf",
                "Die sechs meistdiskutierten Themen — Episodenhäufigkeit nach Zeitperiode. "
                "Kurven geglättet (monotone Interpolation).",
                topics_over_time_chart(),
            ),

            # ── BERTopic deep-dives (lazy iframes) ────────────
            section_intro(
                "BERTopic-Exploration",
                "Modell-Visualisierungen",
                "Direkt eingebettete Plotly-Exporte des trainierten Modells — "
                "werden erst auf Anfrage geladen.",
            ),
            # ── Treemap (ersetzt BERTopic-Hierarchie-iframe) ──────
            chart_card(
                "Themen-Hierarchie",
                "Alle Topics nach Episodenzahl — größere Fläche = mehr Episoden. "
                "Farbe zeigt den Rang: Dunkelblau = meistdiskutiert, Hellblau = seltener.",
                topic_treemap(),
            ),

            rx.grid(
                bertopic_embed_card(
                    str(topics_visualization.get("title", "BERTopic-Cluster")),
                    str(topics_visualization.get("description", "")),
                    topics_visualization.get("html_src") if topics_visualization.get("has_html") else None,
                    DashboardState.load_topics_visualization,
                    DashboardState.enable_topics_visualization,
                ),
                bertopic_embed_card(
                    str(topics_barchart.get("title", "Themen-Häufigkeit")),
                    str(topics_barchart.get("description", "")),
                    topics_barchart.get("html_src") if topics_barchart.get("has_html") else None,
                    DashboardState.load_topics_barchart,
                    DashboardState.enable_topics_barchart,
                ),
                columns="1",
                gap="4",
                width="100%",
            ),
            class_name="page-container",
            display="flex",
            flex_direction="column",
            gap="32px",
            width="100%",
        ),
        class_name="app-shell",
        min_height="100vh",
        width="100%",
    )


# ─── Page: Personensuche (/personensuche) ─────────────────────────────────────


def search_result_row(row: list[str]) -> rx.Component:
    return rx.table.row(
        rx.table.cell(
            rx.button(
                row[0],
                on_click=DashboardState.select_guest(row[0]),
                variant="ghost",
                size="2",
                style={"fontWeight": "600", "color": "var(--blue)", "cursor": "pointer"},
            )
        ),
        rx.table.cell(row[1], class_name="table-cell"),
        rx.table.cell(row[2], class_name="table-cell"),
        rx.table.cell(row[3], class_name="table-value", align="right"),
    )


def appearance_detail_row(row: list[str]) -> rx.Component:
    return rx.table.row(
        rx.table.cell(row[0], class_name="table-cell"),
        rx.table.cell(row[1], class_name="table-cell"),
        rx.table.cell(row[2], class_name="table-cell"),
    )


def guest_detail_panel() -> rx.Component:
    return rx.cond(
        DashboardState.has_guest_detail,
        rx.card(
            rx.flex(
                # Header row
                rx.flex(
                    rx.flex(
                        rx.heading(
                            DashboardState.guest_detail_header[0],
                            size="6",
                            class_name="guest-detail-name",
                        ),
                        rx.text(
                            DashboardState.guest_detail_header[1],
                            class_name="guest-detail-badge",
                        ),
                        align="center",
                        gap="12px",
                        wrap="wrap",
                    ),
                    rx.button(
                        "×",
                        on_click=DashboardState.clear_guest_selection,
                        variant="ghost",
                        size="2",
                        color_scheme="gray",
                    ),
                    justify="between",
                    align="start",
                    width="100%",
                ),
                # Role
                rx.text(
                    DashboardState.guest_detail_header[2],
                    class_name="guest-detail-meta",
                ),
                # Stats row
                rx.flex(
                    rx.flex(
                        rx.text("Auftritte", class_name="guest-detail-stat-label"),
                        rx.text(DashboardState.guest_detail_header[3], class_name="guest-detail-stat-val"),
                        direction="column",
                        spacing="1",
                    ),
                    rx.flex(
                        rx.text("Zeitraum", class_name="guest-detail-stat-label"),
                        rx.text(DashboardState.guest_detail_header[4], class_name="guest-detail-stat-val"),
                        direction="column",
                        spacing="1",
                    ),
                    gap="32px",
                    wrap="wrap",
                ),
                # Appearances table
                rx.heading("Auftrittsverlauf (neueste zuerst)", size="3", class_name="panel-title"),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Datum"),
                            rx.table.column_header_cell("Sendung"),
                            rx.table.column_header_cell("Thema"),
                        )
                    ),
                    rx.table.body(
                        rx.foreach(DashboardState.guest_detail_appearances, appearance_detail_row)
                    ),
                    variant="surface",
                    size="2",
                    width="100%",
                ),
                direction="column",
                spacing="4",
                width="100%",
            ),
            class_name="guest-detail-card",
            size="4",
            variant="surface",
        ),
        rx.box(),  # empty placeholder when no guest selected
    )


def personensuche_page() -> rx.Component:
    return rx.box(
        navbar("personensuche"),
        rx.box(
            page_header(
                "Personensuche",
                "Person nachschlagen",
                "Suche nach Gästen aus dem Datensatz — sieh Auftritte, Partei, Rolle und Themen.",
            ),
            # ── Search input ───────────────────────────────────
            rx.card(
                rx.flex(
                    rx.heading("Gastsuche", size="4", class_name="panel-title"),
                    rx.text(
                        "Tippe einen Namen ein — alle passenden Gäste werden angezeigt.",
                        class_name="panel-copy",
                    ),
                    rx.input(
                        placeholder="z. B. Söder, Baerbock, Lauterbach …",
                        value=DashboardState.person_search_query,
                        on_change=DashboardState.set_person_search_query,
                        size="3",
                        width="100%",
                        max_width="560px",
                    ),
                    # Search results table
                    rx.cond(
                        DashboardState.has_search_results,
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    rx.table.column_header_cell("Name"),
                                    rx.table.column_header_cell("Partei"),
                                    rx.table.column_header_cell("Rolle"),
                                    rx.table.column_header_cell("Auftritte", align="right"),
                                )
                            ),
                            rx.table.body(
                                rx.foreach(DashboardState.person_search_rows, search_result_row)
                            ),
                            variant="surface",
                            size="2",
                            width="100%",
                        ),
                        rx.cond(
                            DashboardState.search_is_empty,
                            rx.text(
                                "Suche beginnt ab dem ersten eingegebenen Zeichen.",
                                class_name="search-hint",
                            ),
                            rx.text(
                                "Keine Treffer für diese Suchanfrage.",
                                class_name="search-hint",
                            ),
                        ),
                    ),
                    direction="column",
                    spacing="4",
                    width="100%",
                ),
                class_name="panel-card",
                size="3",
                variant="surface",
            ),
            # ── Guest detail ────────────────────────────────────
            guest_detail_panel(),
            class_name="page-container",
            display="flex",
            flex_direction="column",
            gap="32px",
            width="100%",
        ),
        class_name="app-shell",
        min_height="100vh",
        width="100%",
    )


# ─── App ─────────────────────────────────────────────────────────────────────

app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue",
        gray_color="slate",
        radius="medium",
        scaling="100%",
    ),
    stylesheets=["/reflex_dashboard.css"],
)

app.add_page(overview_page, route="/", title="Wer talkt mit wem? – Übersicht")
app.add_page(gaeste_page, route="/gaeste", title="WTMW – Gäste-Übersicht")
app.add_page(themen_page, route="/themen", title="WTMW – Themen & Netzwerke")
app.add_page(personensuche_page, route="/personensuche", title="WTMW – Personensuche")
