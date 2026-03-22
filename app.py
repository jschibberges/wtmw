import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import streamlit.components.v1 as components
from app_helpers import (
    TIMEFRAME_OPTIONS,
    filter_by_timeframe,
    format_topic_label,
    prepare_guest_metadata,
    summarize_show_coverage,
    summarize_topic_counts,
)

matplotlib.use("Agg")

st.set_page_config(
    page_title="Wer talkt mit wem?",
    page_icon="🎙️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    """Loads all necessary data files."""
    data_path = DATA_DIR / "all_data_with_topics.xlsx"
    guest_range_path = DATA_DIR / "guest_topic_range.xlsx"
    guests_with_topics_path = DATA_DIR / "guests_with_topics.xlsx"

    if not all([data_path.exists(), guest_range_path.exists(), guests_with_topics_path.exists()]):
        st.error(
            "Daten nicht gefunden. Bitte führe zuerst 'analyze_talkshows.py' aus, "
            "um die Analysedateien zu generieren."
        )
        return None, None, None, None, None

    try:
        df_shows = pd.read_excel(data_path)
        df_guest_range = pd.read_excel(guest_range_path)
        df_guests_with_topics = pd.read_excel(guests_with_topics_path)

        if "date" in df_shows.columns:
            df_shows["date"] = pd.to_datetime(df_shows["date"], errors="coerce", dayfirst=True)

        show_info_cols = [
            col for col in ["uid", "date", "show", "title", "station"] if col in df_shows.columns
        ]
        if show_info_cols:
            df_guests_with_topics = df_guests_with_topics.merge(
                df_shows[show_info_cols], on="uid", how="left"
            )

        if "date" in df_guests_with_topics.columns:
            df_guests_with_topics["date"] = pd.to_datetime(
                df_guests_with_topics["date"], errors="coerce"
            )

        if "topic_label" in df_guest_range.columns:
            df_guest_range.rename(columns={"topic_label": "unique_topics"}, inplace=True)
        if "unique_topics" in df_guest_range.columns:
            df_guest_range["unique_topics"] = pd.to_numeric(
                df_guest_range["unique_topics"], errors="coerce"
            )
            df_guest_range.sort_values("unique_topics", ascending=False, inplace=True)

        guest_metadata = prepare_guest_metadata(df_guests_with_topics)

        appearance_counts = (
            df_guests_with_topics.groupby("name", as_index=False)
            .size()
            .rename(columns={"size": "appearances"})
            .sort_values("appearances", ascending=False)
        )
        appearance_counts = appearance_counts.merge(guest_metadata, on="name", how="left")
        df_guest_range = df_guest_range.merge(guest_metadata, on="name", how="left")

        return df_shows, df_guest_range, df_guests_with_topics, appearance_counts, guest_metadata

    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return None, None, None, None, None


@st.cache_data(ttl=3600)
def load_html_content(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def display_html_file(path, *, height: int = 800, load_key: str | None = None):
    """Display an HTML file on demand (toggle) to avoid slowing down reruns."""
    if not path.exists():
        st.warning(f"Visualisierung nicht gefunden: {path.name}. Bitte 'analyze_talkshows.py' ausführen.")
        return

    toggle_key = load_key or f"load_{path.stem}"
    should_load = st.toggle(
        "Interaktive Visualisierung laden",
        value=False,
        key=toggle_key,
        help="Große HTML-Netzwerke werden nur bei Bedarf geladen.",
    )
    if not should_load:
        st.caption("Nicht geladen. Aktivieren, um die interaktive Ansicht einzublenden.")
        return

    html_content = load_html_content(str(path))
    if html_content is None:
        st.warning(f"Visualisierung nicht gefunden: {path.name}.")
        return
    components.html(html_content, height=height, scrolling=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

st.title("Wer talkt mit wem? – Talkshow Analyse")
st.markdown("Eine interaktive Analyse deutscher Polit-Talkshows, ihrer Gäste und Themen.")

df_shows, df_guest_range, df_guests_with_topics, appearance_counts, guest_metadata = load_data()

if df_shows is not None:
    TAB_LABELS = [
        "📊 Übersicht",
        "🕸️ Netzwerke",
        "🏆 Rankings",
        "🔍 Personen",
        "🗂️ Themen",
        "ℹ️ Über das Projekt",
    ]
    selected_tab = st.radio(
        "Hauptnavigation",
        TAB_LABELS,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )

    # -----------------------------------------------------------------------
    # TAB 1: Übersicht
    # -----------------------------------------------------------------------
    if selected_tab == "📊 Übersicht":
        st.header("Übersicht")

        # --- Global metrics ---
        episode_identifier = "uid" if "uid" in df_shows.columns else None
        episodes_analyzed = (
            int(df_shows[episode_identifier].nunique())
            if episode_identifier is not None
            else int(len(df_shows))
        )
        unique_guests = (
            int(df_guests_with_topics["name"].dropna().nunique())
            if df_guests_with_topics is not None and not df_guests_with_topics.empty
            and "name" in df_guests_with_topics.columns
            else 0
        )

        if "date" in df_shows.columns:
            overall_dates = pd.to_datetime(df_shows["date"], errors="coerce").dropna()
            overall_start = overall_dates.min() if not overall_dates.empty else None
            overall_end = overall_dates.max() if not overall_dates.empty else None
        else:
            overall_start = overall_end = None

        if pd.notna(overall_start) and pd.notna(overall_end):
            timeframe_value = (
                f"{overall_start.strftime('%d.%m.%Y')} – {overall_end.strftime('%d.%m.%Y')}"
            )
        elif pd.notna(overall_end):
            timeframe_value = overall_end.strftime("%d.%m.%Y")
        else:
            timeframe_value = "–"

        metric_cols = st.columns(3)
        metric_cols[0].metric("Analysierte Episoden", f"{episodes_analyzed:,}".replace(",", "."))
        metric_cols[1].metric("Identifizierte Gäste", f"{unique_guests:,}".replace(",", "."))
        metric_cols[2].metric("Zeitraum der Daten", timeframe_value)

        # --- Show coverage table ---
        coverage_df = summarize_show_coverage(df_shows)
        if not coverage_df.empty:
            st.subheader("Zeiträume der Shows")
            coverage_display = coverage_df.rename(columns={"show": "Sendung", "episodes": "Episoden"}).copy()
            for date_col in ["first_date", "last_date"]:
                coverage_display[date_col] = pd.to_datetime(coverage_display[date_col], errors="coerce")
            coverage_display["Erste Episode"] = coverage_display["first_date"].dt.strftime("%d.%m.%Y").fillna("–")
            coverage_display["Letzte Episode"] = coverage_display["last_date"].dt.strftime("%d.%m.%Y").fillna("–")
            coverage_display["Episoden"] = pd.to_numeric(
                coverage_display["Episoden"], errors="coerce"
            ).astype("Int64")
            coverage_display.sort_values("Sendung", inplace=True)
            coverage_display = coverage_display[["Sendung", "Erste Episode", "Letzte Episode", "Episoden"]]
            st.dataframe(coverage_display, width="stretch", hide_index=True)

        # --- Guest category distribution ---
        cat_col = None
        for candidate in ["CategoryPrimary", "category", "Category"]:
            if df_guests_with_topics is not None and candidate in df_guests_with_topics.columns:
                cat_col = candidate
                break

        if cat_col is not None:
            st.subheader("Gästekategorien")
            cat_counts = (
                df_guests_with_topics.dropna(subset=[cat_col])
                .drop_duplicates(subset=["name"])[[cat_col]]
                .groupby(cat_col)
                .size()
                .sort_values(ascending=False)
                .rename("Anzahl Gäste")
                .reset_index()
                .rename(columns={cat_col: "Kategorie"})
            )
            if not cat_counts.empty:
                cat_chart_col, cat_table_col = st.columns([2, 1])
                with cat_chart_col:
                    fig, ax = plt.subplots(figsize=(7, max(3, len(cat_counts) * 0.45)))
                    bars = ax.barh(
                        cat_counts["Kategorie"][::-1],
                        cat_counts["Anzahl Gäste"][::-1],
                        color="#4C72B0",
                    )
                    ax.bar_label(bars, padding=3, fontsize=9)
                    ax.set_xlabel("Einzigartige Gäste")
                    ax.set_title("Gäste nach Kategorie (eindeutig)")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                with cat_table_col:
                    st.dataframe(cat_counts, width="stretch", hide_index=True)

        # --- Topic distribution ---
        topic_counts_df = summarize_topic_counts(df_shows)
        if not topic_counts_df.empty:
            st.subheader("Episoden pro Thema")
            topic_counts_df["Episoden"] = (
                pd.to_numeric(topic_counts_df["Episoden"], errors="coerce").fillna(0).astype(int)
            )
            classified = topic_counts_df[
                ~topic_counts_df["Thema"].isin({"Kein Thema zugeordnet", "Sonstige / kein Thema"})
            ]
            chart_data = classified.head(20).set_index("Thema")
            st.bar_chart(chart_data)
            unclassified_count = int(
                topic_counts_df.loc[
                    topic_counts_df["Thema"].isin({"Kein Thema zugeordnet", "Sonstige / kein Thema"}),
                    "Episoden",
                ].sum()
            )
            st.caption(
                f"Nicht klassifizierte Episoden: {unclassified_count:,}".replace(",", ".")
            )
            with st.expander("Alle Themen anzeigen"):
                st.dataframe(topic_counts_df, width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # TAB 2: Netzwerke
    # -----------------------------------------------------------------------
    elif selected_tab == "🕸️ Netzwerke":
        st.header("Netzwerke")

        # Reuse pre-computed topic counts for metrics
        topic_counts_df = summarize_topic_counts(df_shows)

        network_section = st.radio(
            "Netzwerk wählen",
            ["Gäste-Netzwerk", "Themen-Netzwerk", "BERTopic-Visualisierungen"],
            horizontal=True,
            key="network_section",
        )

        if network_section == "Gäste-Netzwerk":
            st.subheader("Gäste-Netzwerk")
            st.markdown(
                "Wer tritt häufig mit wem auf? Die kuratierte Ansicht zeigt lesbare Cluster; "
                "die interaktive Ansicht eignet sich für Exploration im Detail."
            )

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Gäste im Datensatz", f"{len(appearance_counts):,}".replace(",", "."))
            mc2.metric(
                "Gäste mit mind. 3 Auftritten",
                f"{int((appearance_counts['appearances'] >= 3).sum()):,}".replace(",", "."),
            )
            top_guest = appearance_counts.iloc[0]["name"] if not appearance_counts.empty else "–"
            mc3.metric("Häufigster Gast", str(top_guest))

            view_static, view_interactive = st.tabs(["Kuratiert (statisch)", "Interaktiv"])
            with view_static:
                st.caption("Statische Schnellansicht: fokussiert auf lesbare Cluster statt Vollgraph.")
                guest_network_path = DATA_DIR / "cooccurrence_network.png"
                if guest_network_path.exists():
                    st.image(str(guest_network_path), width="stretch")
                else:
                    st.warning("Grafik 'cooccurrence_network.png' nicht gefunden.")
            with view_interactive:
                st.caption("Zoomen, verschieben und Knotendetails per Hover anzeigen.")
                display_html_file(
                    DATA_DIR / "cooccurrence_network.html",
                    height=880,
                    load_key="load_guest_network_html",
                )

        elif network_section == "Themen-Netzwerk":
            st.subheader("Themen-Netzwerk")
            st.markdown(
                "Welche Themen sind miteinander verbunden? Knotengröße zeigt die Zahl der zugeordneten "
                "Gäste, Kantendicke die Zahl geteilter Gäste."
            )

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Themen mit Episoden", f"{len(topic_counts_df):,}".replace(",", "."))
            if not topic_counts_df.empty:
                top_topic = str(topic_counts_df.iloc[0]["Thema"])
                top_topic_count = int(topic_counts_df.iloc[0]["Episoden"])
            else:
                top_topic, top_topic_count = "–", 0
            mc2.metric("Größtes Thema", top_topic)
            mc3.metric("Episoden im größten Thema", f"{top_topic_count:,}".replace(",", "."))

            view_static, view_interactive = st.tabs(["Kuratiert (statisch)", "Interaktiv"])
            with view_static:
                topic_network_path = DATA_DIR / "topic_cooccurrence_network.png"
                if topic_network_path.exists():
                    st.image(str(topic_network_path), width="stretch")
                else:
                    st.warning("Grafik 'topic_cooccurrence_network.png' nicht gefunden.")
            with view_interactive:
                display_html_file(
                    DATA_DIR / "topic_cooccurrence_network.html",
                    height=880,
                    load_key="load_topic_network_html",
                )

        elif network_section == "BERTopic-Visualisierungen":
            st.subheader("BERTopic-Visualisierungen")
            st.markdown(
                "Interaktive Visualisierungen des trainierten Themenmodells. "
                "Die Inhalte werden erst bei Bedarf geladen."
            )

            with st.expander("Themen-Cluster (Interaktiv)", expanded=False):
                st.markdown("Jede Blase entspricht einem Thema; Nähe = semantische Ähnlichkeit.")
                display_html_file(
                    DATA_DIR / "topics_visualization.html",
                    load_key="load_topics_visualization_html",
                )

            with st.expander("Themen im Zeitverlauf (Interaktiv)", expanded=False):
                st.markdown("Wie haben sich die Top-Themen über die Jahre entwickelt?")
                display_html_file(
                    DATA_DIR / "topics_over_time.html",
                    load_key="load_topics_over_time_html",
                )

            with st.expander("Themen-Hierarchie (Interaktiv)", expanded=False):
                st.markdown("Hierarchische Struktur der Themen.")
                display_html_file(
                    DATA_DIR / "topics_hierarchy.html",
                    load_key="load_topics_hierarchy_html",
                )

            with st.expander("Themen-Häufigkeit (Balkendiagramm, Interaktiv)", expanded=False):
                st.markdown("Die häufigsten Themen als interaktives Balkendiagramm.")
                display_html_file(
                    DATA_DIR / "topics_barchart.html",
                    load_key="load_topics_barchart_html",
                )

    # -----------------------------------------------------------------------
    # TAB 3: Rankings
    # -----------------------------------------------------------------------
    elif selected_tab == "🏆 Rankings":
        st.header("Rankings")

        timeframe_options = list(TIMEFRAME_OPTIONS.keys())
        default_index = timeframe_options.index("Gesamter Zeitraum")
        selected_timeframe = st.selectbox(
            "Zeitraum wählen",
            options=timeframe_options,
            index=default_index,
            key="rankings_timeframe",
        )

        filtered_guests, start_date, end_date = filter_by_timeframe(
            df_guests_with_topics, selected_timeframe
        )

        if filtered_guests.empty:
            st.info("Keine Daten im ausgewählten Zeitraum verfügbar.")
        else:
            if start_date is not None and end_date is not None:
                st.caption(
                    f"Zeitraum: {start_date.strftime('%d.%m.%Y')} – {end_date.strftime('%d.%m.%Y')}"
                )

            # Appearance counts for filtered period
            appearance_counts_filtered = (
                filtered_guests.groupby("name", as_index=False)
                .size()
                .rename(columns={"size": "appearances"})
                .sort_values("appearances", ascending=False)
            )
            appearance_counts_filtered = appearance_counts_filtered.merge(
                guest_metadata, on="name", how="left"
            )

            appearance_table = (
                appearance_counts_filtered[["name", "primary_party", "known_roles", "appearances"]]
                .rename(
                    columns={
                        "name": "Name",
                        "primary_party": "Partei",
                        "known_roles": "Rollen",
                        "appearances": "Auftritte",
                    }
                )
                .copy()
            )
            appearance_table["Partei"] = appearance_table.get("Partei", pd.Series()).fillna("–")
            appearance_table["Rollen"] = appearance_table.get("Rollen", pd.Series()).fillna("–")
            appearance_table["Auftritte"] = pd.to_numeric(
                appearance_table["Auftritte"], errors="coerce"
            ).astype("Int64")

            # Topic diversity for filtered period
            topic_source = filtered_guests.dropna(subset=["name"]).copy()
            if "topic" in topic_source.columns:
                topic_source = topic_source[topic_source["topic"] != -1]
            if "topic_label" in topic_source.columns:
                topic_source = topic_source[topic_source["topic_label"].notna()]
                topic_counts_filtered = (
                    topic_source.groupby("name", as_index=False)["topic_label"]
                    .nunique()
                    .rename(columns={"topic_label": "unique_topics"})
                    .sort_values("unique_topics", ascending=False)
                )
            else:
                topic_counts_filtered = pd.DataFrame(columns=["name", "unique_topics"])

            topic_counts_filtered = topic_counts_filtered.merge(guest_metadata, on="name", how="left")
            topic_range_table = (
                topic_counts_filtered[["name", "primary_party", "known_roles", "unique_topics"]]
                .rename(
                    columns={
                        "name": "Name",
                        "primary_party": "Partei",
                        "known_roles": "Rollen",
                        "unique_topics": "Einzigartige Themen",
                    }
                )
                .copy()
            )
            topic_range_table["Partei"] = topic_range_table.get("Partei", pd.Series()).fillna("–")
            topic_range_table["Rollen"] = topic_range_table.get("Rollen", pd.Series()).fillna("–")
            if "Einzigartige Themen" in topic_range_table.columns:
                topic_range_table["Einzigartige Themen"] = pd.to_numeric(
                    topic_range_table["Einzigartige Themen"], errors="coerce"
                ).astype("Int64")

            # Top-10 bar charts
            chart_col1, chart_col2 = st.columns(2)
            top_n = 10

            with chart_col1:
                st.subheader(f"Top {top_n} nach Auftritten")
                top_appearances = appearance_table.head(top_n).copy()
                if not top_appearances.empty:
                    fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.45)))
                    ax.barh(
                        top_appearances["Name"][::-1],
                        top_appearances["Auftritte"][::-1],
                        color="#4C72B0",
                    )
                    ax.set_xlabel("Auftritte")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

            with chart_col2:
                st.subheader(f"Top {top_n} nach Themenvielfalt")
                top_topics_r = topic_range_table.head(top_n).copy()
                if not top_topics_r.empty:
                    fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.45)))
                    ax.barh(
                        top_topics_r["Name"][::-1],
                        pd.to_numeric(top_topics_r["Einzigartige Themen"], errors="coerce")[::-1],
                        color="#DD8452",
                    )
                    ax.set_xlabel("Einzigartige Themen")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

            # Full tables
            table_col1, table_col2 = st.columns(2)
            with table_col1:
                st.subheader("Top 25 Gäste nach Auftritten")
                st.dataframe(appearance_table.head(25), width="stretch", hide_index=True)
            with table_col2:
                st.subheader("Top 25 Gäste nach Themenvielfalt")
                st.dataframe(topic_range_table.head(25), width="stretch", hide_index=True)

            with st.expander("Gesamte Ranglisten anzeigen"):
                st.subheader("Alle Gäste nach Auftritten")
                st.dataframe(appearance_table, width="stretch", hide_index=True)
                st.subheader("Alle Gäste nach Themenvielfalt")
                st.dataframe(topic_range_table, width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # TAB 4: Personen
    # -----------------------------------------------------------------------
    elif selected_tab == "🔍 Personen":
        st.header("Personensuche")

        # Timeframe filter
        timeframe_options = list(TIMEFRAME_OPTIONS.keys())
        default_index = timeframe_options.index("Gesamter Zeitraum")
        selected_timeframe = st.selectbox(
            "Zeitraum wählen",
            options=timeframe_options,
            index=default_index,
            key="person_timeframe",
        )

        filtered_for_person, p_start, p_end = filter_by_timeframe(
            df_guests_with_topics, selected_timeframe
        )

        if p_start is not None and p_end is not None:
            st.caption(f"Zeitraum: {p_start.strftime('%d.%m.%Y')} – {p_end.strftime('%d.%m.%Y')}")

        # Person selector (built from filtered data so only active guests appear)
        guest_names = sorted(filtered_for_person["name"].dropna().unique()) if not filtered_for_person.empty else []
        search_term = st.selectbox(
            "Wähle eine Person aus der Liste:",
            options=guest_names,
            index=None,
            placeholder="Namen auswählen...",
            key="person_search",
        )

        if search_term:
            person_data = filtered_for_person[filtered_for_person["name"] == search_term].copy()
            st.subheader(f"Analyse für: {search_term}")

            # Metrics
            appearance_count = len(person_data)

            topic_range_count = 0
            if "topic_label" in person_data.columns and "topic" in person_data.columns:
                topic_range_count = int(
                    person_data[person_data["topic"] != -1]["topic_label"].dropna().nunique()
                )

            meta_row = guest_metadata[guest_metadata["name"] == search_term]
            primary_party = meta_row["primary_party"].iloc[0] if not meta_row.empty else None
            known_roles = meta_row["known_roles"].iloc[0] if not meta_row.empty else None

            col1, col2 = st.columns(2)
            col1.metric("Auftritte im Zeitraum", appearance_count)
            col2.metric("Verschiedene Themen", topic_range_count)

            info_col1, info_col2 = st.columns(2)
            info_col1.markdown(f"**Partei:** {primary_party if primary_party else '–'}")
            info_col2.markdown(f"**Rollen:** {known_roles if known_roles else '–'}")

            if "date" in person_data.columns and person_data["date"].notna().any():
                last_date = person_data["date"].dropna().max()
                st.caption(f"Letzter Auftritt im Zeitraum: {last_date.strftime('%d.%m.%Y')}")

            # Appearances list
            st.markdown("---")
            st.subheader("Auftritte im Zeitraum")

            sendung_col = "show" if "show" in person_data.columns else None
            titel_col = "title" if "title" in person_data.columns else sendung_col

            display_cols: dict = {}
            if "date" in person_data.columns:
                display_cols["Datum"] = person_data["date"]
            if sendung_col:
                display_cols["Sendung"] = person_data[sendung_col]
            if titel_col and titel_col != sendung_col:
                display_cols["Titel"] = person_data[titel_col]
            if "topic_label" in person_data.columns:
                display_cols["Thema"] = person_data["topic_label"].apply(format_topic_label)

            display_df = pd.DataFrame(display_cols)
            if "Datum" in display_df.columns:
                display_df["Datum"] = pd.to_datetime(display_df["Datum"], errors="coerce")
                display_df.sort_values("Datum", ascending=False, inplace=True, na_position="last")
                display_df["Datum"] = display_df["Datum"].dt.strftime("%d.%m.%Y").fillna("–")
            for text_col in ["Sendung", "Titel", "Thema"]:
                if text_col in display_df.columns:
                    display_df[text_col] = display_df[text_col].fillna("–")

            st.dataframe(display_df, width="stretch", hide_index=True)

            # Topic summary
            if "topic_label" in person_data.columns:
                topics = person_data["topic_label"].dropna()
                if not topics.empty:
                    topic_summary = (
                        topics.apply(format_topic_label)
                        .rename("topic_label")
                        .value_counts()
                        .reset_index(name="Auftritte")
                        .rename(columns={"topic_label": "Thema"})
                    )
                    st.subheader("Themenschwerpunkte")
                    st.dataframe(topic_summary, width="stretch", hide_index=True)

            # Co-guest analysis
            st.markdown("---")
            st.subheader("Häufigste Mitgäste im Zeitraum")

            if "uid" in person_data.columns and "uid" in filtered_for_person.columns:
                person_uids = set(person_data["uid"].dropna().unique())
                co_guests_df = filtered_for_person[
                    (filtered_for_person["uid"].isin(person_uids))
                    & (filtered_for_person["name"] != search_term)
                ].copy()

                if not co_guests_df.empty:
                    co_guest_counts = (
                        co_guests_df.groupby("name", as_index=False)
                        .size()
                        .rename(columns={"size": "Gemeinsame Sendungen"})
                        .sort_values("Gemeinsame Sendungen", ascending=False)
                    )
                    co_guest_counts = co_guest_counts.merge(guest_metadata, on="name", how="left")
                    co_guest_counts.rename(
                        columns={
                            "name": "Name",
                            "primary_party": "Partei",
                            "known_roles": "Rollen",
                        },
                        inplace=True,
                    )
                    co_guest_counts["Partei"] = co_guest_counts["Partei"].fillna("–")
                    co_guest_counts["Rollen"] = co_guest_counts["Rollen"].fillna("–")
                    co_guest_counts["Gemeinsame Sendungen"] = pd.to_numeric(
                        co_guest_counts["Gemeinsame Sendungen"], errors="coerce"
                    ).astype("Int64")

                    top_co = co_guest_counts.head(10)
                    co_chart_col, co_table_col = st.columns([1, 2])
                    with co_chart_col:
                        fig, ax = plt.subplots(figsize=(5, max(3, len(top_co) * 0.45)))
                        ax.barh(
                            top_co["Name"][::-1],
                            pd.to_numeric(top_co["Gemeinsame Sendungen"], errors="coerce")[::-1],
                            color="#55A868",
                        )
                        ax.set_xlabel("Gemeinsame Sendungen")
                        ax.set_title("Top 10 Mitgäste")
                        ax.spines[["top", "right"]].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig, width="stretch")
                        plt.close(fig)
                    with co_table_col:
                        st.dataframe(
                            co_guest_counts.head(25), width="stretch", hide_index=True
                        )
                else:
                    st.info("Keine Mitgäste gefunden.")
            else:
                st.info("Mitgast-Analyse benötigt eine UID-Spalte in den Daten.")

    # -----------------------------------------------------------------------
    # TAB 5: Themen
    # -----------------------------------------------------------------------
    elif selected_tab == "🗂️ Themen":
        st.header("Themensuche")

        # Timeframe filter
        timeframe_options = list(TIMEFRAME_OPTIONS.keys())
        default_index = timeframe_options.index("Gesamter Zeitraum")
        selected_timeframe = st.selectbox(
            "Zeitraum wählen",
            options=timeframe_options,
            index=default_index,
            key="topic_timeframe",
        )

        filtered_shows, t_start, t_end = filter_by_timeframe(df_shows, selected_timeframe)
        filtered_guests_topic, _, _ = filter_by_timeframe(df_guests_with_topics, selected_timeframe)

        if t_start is not None and t_end is not None:
            st.caption(f"Zeitraum: {t_start.strftime('%d.%m.%Y')} – {t_end.strftime('%d.%m.%Y')}")

        topic_df = pd.DataFrame()
        if not filtered_shows.empty and "topic" in filtered_shows.columns and "topic_label" in filtered_shows.columns:
            topic_df = filtered_shows[filtered_shows["topic"].notna()].copy()
            topic_df = topic_df[topic_df["topic"] != -1]

        if topic_df.empty:
            st.info("Keine Themeninformationen verfügbar.")
        else:
            topic_df["topic_display"] = topic_df["topic_label"].apply(format_topic_label)
            options_df = (
                topic_df[["topic_label", "topic_display"]]
                .dropna(subset=["topic_label"])
                .drop_duplicates()
                .sort_values("topic_display")
            )
            topic_options = options_df["topic_label"].tolist()
            topic_display_map = dict(zip(topic_options, options_df["topic_display"]))

            selected_topic = st.selectbox(
                "Wähle ein Thema:",
                options=topic_options,
                index=None,
                placeholder="Thema auswählen...",
                format_func=lambda x: topic_display_map.get(x, "–"),
                key="topic_search",
            )

            if selected_topic:
                topic_label_display = topic_display_map.get(selected_topic, selected_topic)
                st.subheader(f"Analyse für: {topic_label_display}")

                topic_episodes = topic_df[topic_df["topic_label"] == selected_topic].copy()
                topic_guests = filtered_guests_topic[
                    filtered_guests_topic["topic_label"] == selected_topic
                ].copy() if not filtered_guests_topic.empty and "topic_label" in filtered_guests_topic.columns else pd.DataFrame()

                episode_count = len(topic_episodes)
                unique_g = topic_guests["name"].nunique() if not topic_guests.empty else 0

                date_series = pd.to_datetime(
                    topic_episodes["date"] if "date" in topic_episodes.columns else pd.Series(dtype="datetime64[ns]"),
                    errors="coerce",
                )
                first_date = date_series.min() if not date_series.empty else None
                last_date = date_series.max() if not date_series.empty else None
                if pd.notna(first_date) and pd.notna(last_date):
                    timespan = f"{first_date.strftime('%d.%m.%Y')} – {last_date.strftime('%d.%m.%Y')}"
                elif pd.notna(last_date):
                    timespan = last_date.strftime("%d.%m.%Y")
                else:
                    timespan = "–"

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Sendungen", episode_count)
                mc2.metric("Zeitraum", timespan)
                mc3.metric("Einzigartige Gäste", int(unique_g) if pd.notna(unique_g) else 0)

                # Temporal trend chart
                st.markdown("---")
                st.subheader("Zeitverlauf")

                if "date" in topic_episodes.columns and topic_episodes["date"].notna().any():
                    trend_df = topic_episodes.copy()
                    trend_df["month"] = pd.to_datetime(trend_df["date"], errors="coerce").dt.to_period("M")
                    trend_df = trend_df.dropna(subset=["month"])
                    if not trend_df.empty:
                        monthly_counts = (
                            trend_df.groupby("month").size().rename("Sendungen").reset_index()
                        )
                        monthly_counts["month_dt"] = monthly_counts["month"].dt.to_timestamp()
                        monthly_counts.sort_values("month_dt", inplace=True)

                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.bar(
                            monthly_counts["month_dt"],
                            monthly_counts["Sendungen"],
                            width=20,
                            color="#4C72B0",
                            alpha=0.85,
                        )
                        ax.set_ylabel("Sendungen")
                        ax.set_title(f"Sendungen pro Monat: {topic_label_display}")
                        ax.spines[["top", "right"]].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig, width="stretch")
                        plt.close(fig)
                else:
                    st.info("Keine Datumsinformationen für Zeitverlauf verfügbar.")

                # Episode list
                st.markdown("---")
                st.subheader("Sendungen mit diesem Thema")

                column_mapping = {
                    "date": "Datum",
                    "show": "Sendung",
                    "title": "Titel der Sendung",
                    "station": "Sender",
                }
                available_cols = [col for col in column_mapping if col in topic_episodes.columns]
                episode_display = topic_episodes[available_cols].rename(
                    columns={k: column_mapping[k] for k in available_cols}
                ).copy()

                if "Datum" in episode_display.columns:
                    episode_display["Datum"] = pd.to_datetime(
                        episode_display["Datum"], errors="coerce"
                    )
                    episode_display.sort_values(
                        "Datum", ascending=False, inplace=True, na_position="last"
                    )
                    episode_display["Datum"] = (
                        episode_display["Datum"].dt.strftime("%d.%m.%Y").fillna("–")
                    )
                for text_col in ["Sendung", "Titel der Sendung", "Sender"]:
                    if text_col in episode_display.columns:
                        episode_display[text_col] = episode_display[text_col].fillna("–")

                st.dataframe(episode_display, width="stretch", hide_index=True)

                # Guest list
                if not topic_guests.empty:
                    st.subheader("Top Gäste zum Thema")
                    guest_summary = (
                        topic_guests.groupby("name", as_index=False)
                        .size()
                        .rename(columns={"size": "Auftritte"})
                        .sort_values("Auftritte", ascending=False)
                    )
                    guest_summary = guest_summary.merge(guest_metadata, on="name", how="left")
                    guest_summary.rename(
                        columns={
                            "name": "Name",
                            "primary_party": "Partei",
                            "known_roles": "Rollen",
                        },
                        inplace=True,
                    )
                    guest_summary["Partei"] = guest_summary["Partei"].fillna("–")
                    guest_summary["Rollen"] = guest_summary["Rollen"].fillna("–")
                    guest_summary["Auftritte"] = pd.to_numeric(
                        guest_summary["Auftritte"], errors="coerce"
                    ).astype("Int64")
                    st.dataframe(guest_summary.head(25), width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # TAB 6: Über das Projekt
    # -----------------------------------------------------------------------
    elif selected_tab == "ℹ️ Über das Projekt":
        st.header("Über das Projekt")
        st.markdown(
            """
**Wer talkt mit wem?** ist eine Datenplattform zur Analyse von Gästen, Themen und Netzwerken
in deutschen politischen Talkshows. Ausgewertet werden Anne Will, Caren Miosga, Hart aber Fair,
Markus Lanz, Maischberger und Maybrit Illner.

---

### Technologie-Stack

- **Datensammlung:** Python mit `requests` und `BeautifulSoup` (Quelle: fernsehserien.de)
- **Datenanalyse:** `pandas` für die Datenverarbeitung, `spaCy` für deutsche NLP-Vorverarbeitung
- **Topic Modeling:** `BERTopic` mit `intfloat/multilingual-e5-base`-Embeddings, `UMAP` und `HDBSCAN`
- **Gastklassifizierung:** Regelbasierte Klassifizierung (Regex auf Rollenbezeichnungen) + Embedding-Fallback
- **Netzwerkanalyse:** `networkx` und `pyvis` für Gäste- und Themen-Netzwerke
- **Dashboard:** `Streamlit`

---

### Analyseschritte

1. **Scraping** (`scrape_talkshows.py`): Sendungsdaten (Titel, Datum, Beschreibung, Gäste)
2. **Analyse** (`analyze_talkshows.py`): Gästebereinigung, Klassifizierung, Topic Modeling, Netzwerke
3. **Dashboard** (`streamlit run app.py`): Interaktive Exploration

---

### Gastklassifizierung

Gäste werden in 9 Kategorien eingeteilt: Politics & Government · Media & Communication ·
Academia & Expertise · Business & Economy · Civil Society & Advocacy · Arts & Culture ·
Sports · Religion · Citizens & Everyday Voices.

Die Klassifizierung läuft zweistufig: zuerst regelbasiert (Regex auf Rollenbezeichnungen
und Parteizugehörigkeit), dann per Embedding-Ähnlichkeit zu Kategorie-Prototypen für
unerkannte Gäste.

---

### Themenmodellierung

BERTopic mit deutschen Stopwörtern, Lemmatisierung und POS-Filterung (nur Nomen, Eigennamen,
Adjektive). Topic-Namen können in `data/topic_labels.json` ohne Retraining angepasst werden.
"""
        )

else:
    st.info(
        "Lade Daten... Falls dies länger dauert, stelle sicher, dass die Analysedateien "
        "im 'data'-Ordner vorhanden sind."
    )
