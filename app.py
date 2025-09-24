import streamlit as st
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Wer talkt mit wem?",
    page_icon="🎙️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def _most_common_value(series: pd.Series) -> str | None:
    """Return the most frequent non-empty value in a series."""
    if series is None:
        return None
    valid = series.dropna().astype(str).str.strip()
    valid = valid[valid != ""]
    if valid.empty:
        return None
    counts = valid.value_counts()
    return counts.index[0] if not counts.empty else None


def _summarize_roles(series: pd.Series) -> str | None:
    """Build a short, unique role summary for a guest."""
    if series is None:
        return None
    roles = [role.strip() for role in series.dropna().astype(str) if role and role.strip()]
    if not roles:
        return None
    unique_roles = list(dict.fromkeys(roles))
    summary = ", ".join(unique_roles[:3])
    if len(unique_roles) > 3:
        summary += ", …"
    return summary


def format_topic_label(label: str | float | None) -> str:
    """Make topic labels from BERTopic more readable for the UI."""
    if not isinstance(label, str):
        return "Kein Thema zugeordnet"
    text = label.strip()
    if not text:
        return "Kein Thema zugeordnet"
    parts = text.split(" ", 1)
    prefix = parts[0]
    if prefix.lstrip("-").isdigit():
        if prefix == "-1":
            return "Sonstige / kein Thema"
        remainder = parts[1].strip() if len(parts) > 1 else ""
        return f"Topic {prefix}: {remainder}" if remainder else f"Topic {prefix}"
    return text


def prepare_guest_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Create a helper table with party and role summaries per guest."""
    if df.empty:
        return pd.DataFrame(columns=["name", "primary_party", "known_roles"])

    metadata = (
        df.groupby("name", as_index=False)
        .agg(
            primary_party=("party", _most_common_value),
            known_roles=("role", _summarize_roles),
        )
    )
    return metadata

TIMEFRAME_OPTIONS: dict[str, pd.DateOffset | None] = {
    "6 Monate": pd.DateOffset(months=6),
    "Letzte 12 Monate": pd.DateOffset(months=12),
    "2 Jahre": pd.DateOffset(years=2),
    "Gesamter Zeitraum": None,
}


def filter_by_timeframe(
    df: pd.DataFrame | None, timeframe_label: str
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """Return a copy of the data filtered to the chosen timeframe."""
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(), None, None

    date_series = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = date_series.dropna()
    if valid_dates.empty:
        return pd.DataFrame(), None, None

    latest_date = valid_dates.max()
    offset = TIMEFRAME_OPTIONS.get(timeframe_label)

    if offset is None or timeframe_label == "Gesamter Zeitraum":
        filtered = df.copy()
        start_date = valid_dates.min()
    else:
        start_date = latest_date - offset
        filtered = df[date_series >= start_date].copy()

    return filtered, start_date, latest_date


def summarize_show_coverage(df: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize date coverage and episode count per show."""
    if (
        df is None
        or df.empty
        or "show" not in df.columns
        or "date" not in df.columns
    ):
        return pd.DataFrame(columns=["show", "first_date", "last_date", "episodes"])

    working = df.dropna(subset=["show"]).copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working[working["date"].notna()]
    if working.empty:
        return pd.DataFrame(columns=["show", "first_date", "last_date", "episodes"])

    date_summary = (
        working.groupby("show")["date"]
        .agg(["min", "max"])
        .rename(columns={"min": "first_date", "max": "last_date"})
    )

    if "uid" in working.columns:
        episode_counts = working.groupby("show")["uid"].nunique()
    else:
        episode_counts = working.groupby("show").size()

    summary = date_summary.join(episode_counts.rename("episodes"))
    return summary.reset_index()


def summarize_topic_counts(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return episode counts per topic, including unclassified episodes."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Thema", "Episoden"])

    working = df.copy()
    if "topic_label" in working.columns:
        working["topic_display"] = working["topic_label"].apply(format_topic_label)
    else:
        working["topic_display"] = "Kein Thema zugeordnet"

    if "topic" in working.columns:
        working.loc[working["topic"].isna(), "topic_display"] = "Kein Thema zugeordnet"
        working.loc[working["topic"] == -1, "topic_display"] = "Sonstige / kein Thema"

    topic_counts = (
        working["topic_display"].fillna("Kein Thema zugeordnet").value_counts()
    )

    return (
        topic_counts.rename_axis("Thema")
        .reset_index(name="Episoden")
        .sort_values("Episoden", ascending=False)
    )

# --- Data Loading Functions ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data():
    """Loads all necessary data files."""
    data_path = DATA_DIR / "all_data_with_topics.xlsx"
    guest_range_path = DATA_DIR / "guest_topic_range.xlsx"
    guests_with_topics_path = DATA_DIR / "guests_with_topics.xlsx"
    
    if not all([data_path.exists(), guest_range_path.exists(), guests_with_topics_path.exists()]):
        st.error("Daten nicht gefunden. Bitte führe zuerst 'analyze_talkshows.py' aus, um die Analysedateien zu generieren.")
        return None, None, None, None, None

    try:
        df_shows = pd.read_excel(data_path)
        df_guest_range = pd.read_excel(guest_range_path)
        df_guests_with_topics = pd.read_excel(guests_with_topics_path)

        if 'date' in df_shows.columns:
            df_shows['date'] = pd.to_datetime(df_shows['date'], errors='coerce', dayfirst=True)

        show_info_cols = [col for col in ['uid', 'date', 'show', 'title', 'station'] if col in df_shows.columns]
        if show_info_cols:
            df_guests_with_topics = df_guests_with_topics.merge(
                df_shows[show_info_cols],
                on='uid',
                how='left'
            )

        if 'date' in df_guests_with_topics.columns:
            df_guests_with_topics['date'] = pd.to_datetime(df_guests_with_topics['date'], errors='coerce')

        if 'topic_label' in df_guest_range.columns:
            df_guest_range.rename(columns={'topic_label': 'unique_topics'}, inplace=True)
        if 'unique_topics' in df_guest_range.columns:
            df_guest_range['unique_topics'] = pd.to_numeric(df_guest_range['unique_topics'], errors='coerce')
            df_guest_range.sort_values('unique_topics', ascending=False, inplace=True)

        guest_metadata = prepare_guest_metadata(df_guests_with_topics)

        appearance_counts = (
            df_guests_with_topics.groupby('name', as_index=False)
            .size()
            .rename(columns={'size': 'appearances'})
            .sort_values('appearances', ascending=False)
        )
        appearance_counts = appearance_counts.merge(guest_metadata, on='name', how='left')

        df_guest_range = df_guest_range.merge(guest_metadata, on='name', how='left')

        return df_shows, df_guest_range, df_guests_with_topics, appearance_counts, guest_metadata
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return None, None, None, None, None

def display_html_file(path):
    """Reads and displays an HTML file in Streamlit."""
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.warning(f"Visualisierung nicht gefunden: {path.name}. Bitte 'analyze_talkshows.py' ausführen.")

# --- Main App ---
st.title('Wer talkt mit wem? - Talkshow Analyse')
st.markdown("Eine interaktive Analyse deutscher Polit-Talkshows, ihrer Gäste und Themen.")

# Load data and handle potential errors
df_shows, df_guest_range, df_guests_with_topics, appearance_counts, guest_metadata = load_data()

if df_shows is not None:
    tab_labels = [
        "📊 Dashboards & Netzwerke",
        "🏆 Rankings",
        "🔍 Personensuche",
        "🗂️ Themensuche",
        "ℹ️ Über das Projekt",
    ]
    selected_tab = st.radio(
        "Hauptnavigation",
        tab_labels,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )

    if selected_tab == "📊 Dashboards & Netzwerke":
        st.header("Dashboards & Netzwerke")
        st.markdown("Visualisierungen der Gast- und Themen-Netzwerke sowie der Themenanalyse.")

        episode_identifier = "uid" if "uid" in df_shows.columns else None
        if episode_identifier is not None:
            episodes_analyzed = int(df_shows[episode_identifier].nunique())
        else:
            episodes_analyzed = int(len(df_shows))

        if (
            df_guests_with_topics is not None
            and not df_guests_with_topics.empty
            and "name" in df_guests_with_topics.columns
        ):
            unique_guests = int(df_guests_with_topics["name"].dropna().nunique())
        else:
            unique_guests = 0

        if "date" in df_shows.columns:
            overall_dates = pd.to_datetime(df_shows["date"], errors="coerce")
            overall_dates = overall_dates.dropna()
            overall_start = overall_dates.min() if not overall_dates.empty else None
            overall_end = overall_dates.max() if not overall_dates.empty else None
        else:
            overall_start = overall_end = None

        if pd.notna(overall_start) and pd.notna(overall_end):
            timeframe_value = f"{overall_start.strftime('%d.%m.%Y')} – {overall_end.strftime('%d.%m.%Y')}"
        elif pd.notna(overall_end):
            timeframe_value = overall_end.strftime("%d.%m.%Y")
        else:
            timeframe_value = "–"

        metric_cols = st.columns(3)
        metric_cols[0].metric("Analysierte Episoden", f"{episodes_analyzed:,}".replace(",", "."))
        metric_cols[1].metric("Identifizierte Gäste", f"{unique_guests:,}".replace(",", "."))
        metric_cols[2].metric("Zeitraum der Daten", timeframe_value)

        coverage_df = summarize_show_coverage(df_shows)
        if not coverage_df.empty:
            st.subheader("Zeiträume der Shows")
            coverage_display = coverage_df.rename(
                columns={"show": "Sendung", "episodes": "Episoden"}
            ).copy()
            for date_col in ["first_date", "last_date"]:
                coverage_display[date_col] = pd.to_datetime(coverage_display[date_col], errors="coerce")
            coverage_display["Erste Episode"] = coverage_display["first_date"].dt.strftime("%d.%m.%Y")
            coverage_display["Letzte Episode"] = coverage_display["last_date"].dt.strftime("%d.%m.%Y")
            coverage_display["Erste Episode"].fillna("–", inplace=True)
            coverage_display["Letzte Episode"].fillna("–", inplace=True)
            coverage_display["Episoden"] = pd.to_numeric(
                coverage_display["Episoden"], errors="coerce"
            ).astype("Int64")
            coverage_display.sort_values("Sendung", inplace=True)
            coverage_display = coverage_display[["Sendung", "Erste Episode", "Letzte Episode", "Episoden"]]
            st.dataframe(coverage_display, use_container_width=True, hide_index=True)

        topic_counts_df = summarize_topic_counts(df_shows)
        if not topic_counts_df.empty:
            st.subheader("Episoden pro Thema")
            topic_counts_df["Episoden"] = pd.to_numeric(
                topic_counts_df["Episoden"], errors="coerce"
            ).fillna(0).astype(int)
            chart_data = topic_counts_df.head(20).set_index("Thema")
            st.bar_chart(chart_data)
            unclassified_labels = {"Kein Thema zugeordnet", "Sonstige / kein Thema"}
            unclassified_count = int(
                topic_counts_df.loc[
                    topic_counts_df["Thema"].isin(unclassified_labels), "Episoden"
                ].sum()
            )
            st.caption(
                "Nicht klassifizierte Episoden: "
                + (f"{unclassified_count:,}".replace(",", ".") if unclassified_count else "0")
            )
            st.dataframe(topic_counts_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gäste-Netzwerk")
            st.markdown("Wer tritt häufig mit wem auf? Die Größe der Knoten repräsentiert die Auftrittshäufigkeit, die Farbe die Themenvielfalt (heller = mehr Themen).")
            guest_network_path = DATA_DIR / "cooccurrence_network.png"
            if guest_network_path.exists():
                st.image(str(guest_network_path), use_column_width=True)
            else:
                st.warning("Grafik 'cooccurrence_network.png' nicht gefunden.")

        with col2:
            st.subheader("Themen-Netzwerk")
            st.markdown("Welche Themen sind miteinander verbunden? Die Größe der Knoten repräsentiert die Anzahl der Gäste, die zu diesem Thema gesprochen haben. Die Dicke der Verbindungslinien zeigt, wie viele Gäste sie teilen.")
            topic_network_path = DATA_DIR / "topic_cooccurrence_network.png"
            if topic_network_path.exists():
                st.image(str(topic_network_path), use_column_width=True)
            else:
                st.warning("Grafik 'topic_cooccurrence_network.png' nicht gefunden.")

        st.subheader("Interaktive Themen-Visualisierungen")

        with st.expander("Themen-Cluster (Interaktiv)"):
            st.markdown("Interaktive Visualisierung der Themencluster. Jede Blase ist ein Thema.")
            display_html_file(DATA_DIR / "topics_visualization.html")

        with st.expander("Themen im Zeitverlauf (Interaktiv)"):
            st.markdown("Wie haben sich die Top-Themen über die Jahre entwickelt?")
            display_html_file(DATA_DIR / "topics_over_time.html")

        with st.expander("Themen-Hierarchie (Interaktiv)"):
            st.markdown("Hierarchische Struktur der Themen.")
            display_html_file(DATA_DIR / "topics_hierarchy.html")

        with st.expander("Themen-Häufigkeit (Balkendiagramm, Interaktiv)"):
            st.markdown("Die häufigsten Themen als Balkendiagramm.")
            display_html_file(DATA_DIR / "topics_barchart.html")

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

        filtered_guests, start_date, end_date = filter_by_timeframe(df_guests_with_topics, selected_timeframe)

        if filtered_guests.empty:
            st.info("Keine Daten im ausgewählten Zeitraum verfügbar.")
        else:
            if start_date is not None and end_date is not None:
                st.caption(f"Zeitraum: {start_date.strftime('%d.%m.%Y')} – {end_date.strftime('%d.%m.%Y')}")

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
            )
            if "Partei" in appearance_table.columns:
                appearance_table["Partei"] = appearance_table["Partei"].fillna("–")
            if "Rollen" in appearance_table.columns:
                appearance_table["Rollen"] = appearance_table["Rollen"].fillna("–")
            appearance_table["Auftritte"] = (
                pd.to_numeric(appearance_table["Auftritte"], errors="coerce").astype("Int64")
            )

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

            topic_counts_filtered = topic_counts_filtered.merge(
                guest_metadata, on="name", how="left"
            )
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
            )
            if "Partei" in topic_range_table.columns:
                topic_range_table["Partei"] = topic_range_table["Partei"].fillna("–")
            if "Rollen" in topic_range_table.columns:
                topic_range_table["Rollen"] = topic_range_table["Rollen"].fillna("–")
            if "Einzigartige Themen" in topic_range_table.columns:
                topic_range_table["Einzigartige Themen"] = pd.to_numeric(
                    topic_range_table["Einzigartige Themen"], errors="coerce"
                ).astype("Int64")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top 25 Gäste nach Auftritten")
                st.dataframe(appearance_table.head(25), width="stretch", hide_index=True)

            with col2:
                st.subheader("Top 25 Gäste nach Themenvielfalt")
                st.dataframe(topic_range_table.head(25), width="stretch", hide_index=True)

            with st.expander("Gesamte Ranglisten anzeigen"):
                st.subheader("Alle Gäste nach Auftritten")
                st.dataframe(appearance_table, width="stretch", hide_index=True)
                st.subheader("Alle Gäste nach Themenvielfalt")
                st.dataframe(topic_range_table, width="stretch", hide_index=True)

    elif selected_tab == "🔍 Personensuche":
        st.header("Personensuche")

        guest_names = sorted(df_guests_with_topics["name"].dropna().unique())
        search_term = st.selectbox(
            "Wähle eine Person aus der Liste:",
            options=guest_names,
            index=None,
            placeholder="Namen auswählen...",
        )

        if search_term:
            person_data = df_guests_with_topics[df_guests_with_topics["name"] == search_term].copy()
            st.subheader(f"Analyse für: {search_term}")

            appearance_val = appearance_counts.loc[
                appearance_counts["name"] == search_term, "appearances"
            ]
            appearance_count = int(appearance_val.iloc[0]) if not appearance_val.empty else 0

            topic_range_val = df_guest_range.loc[
                df_guest_range["name"] == search_term, "unique_topics"
            ]
            topic_range_count = int(topic_range_val.iloc[0]) if not topic_range_val.empty else 0

            meta_row = guest_metadata[guest_metadata["name"] == search_term]
            primary_party = meta_row["primary_party"].iloc[0] if not meta_row.empty else None
            known_roles = meta_row["known_roles"].iloc[0] if not meta_row.empty else None

            col1, col2 = st.columns(2)
            col1.metric("Anzahl Auftritte", appearance_count)
            col2.metric("Themenvielfalt (Anzahl einzigartiger Themen)", topic_range_count)

            info_col1, info_col2 = st.columns(2)
            info_col1.markdown(f"**Partei:** {primary_party if primary_party else '–'}")
            info_col2.markdown(f"**Rollen:** {known_roles if known_roles else '–'}")

            if "date" in person_data.columns and person_data["date"].notna().any():
                last_date = person_data["date"].dropna().max()
                st.caption(f"Letzter Auftritt am {last_date.strftime('%d.%m.%Y')}")

            st.markdown("---")
            st.subheader("Alle Auftritte")

            sendung_col = "show" if "show" in person_data.columns else "Talkshow"
            titel_col = "title" if "title" in person_data.columns else sendung_col

            display_df = pd.DataFrame(
                {
                    "Datum": person_data["date"] if "date" in person_data.columns else pd.NaT,
                    "Sendung": person_data[sendung_col],
                    "Titel der Sendung": person_data[titel_col],
                    "Thema (Topic Model)": person_data["topic_label"].apply(format_topic_label),
                }
            )
            display_df["Datum"] = pd.to_datetime(display_df["Datum"], errors="coerce")
            display_df.sort_values(by="Datum", ascending=False, inplace=True, na_position="last")
            display_df["Datum"] = display_df["Datum"].dt.strftime("%d.%m.%Y")
            display_df["Datum"] = display_df["Datum"].fillna("–")
            display_df["Sendung"] = display_df["Sendung"].fillna("–")
            display_df["Titel der Sendung"] = display_df["Titel der Sendung"].fillna("–")
            st.dataframe(display_df, width="stretch", hide_index=True)

            topics = person_data["topic_label"].dropna()
            if not topics.empty:
                topic_summary = (
                    topics.apply(format_topic_label)
                    .value_counts()
                    .reset_index(name="Auftritte")
                    .rename(columns={"index": "Thema (Topic Model)"})
                )
                st.subheader("Themenschwerpunkte (Topic Model)")
                st.dataframe(topic_summary, width="stretch", hide_index=True)

    elif selected_tab == "🗂️ Themensuche":
        st.header("Themensuche")

        topic_df = pd.DataFrame()
        if "topic" in df_shows.columns and "topic_label" in df_shows.columns:
            topic_df = df_shows[df_shows["topic"].notna()].copy()
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
            )

            if selected_topic:
                topic_label_display = topic_display_map.get(selected_topic, selected_topic)
                st.subheader(f"Analyse für: {topic_label_display}")

                topic_episodes = topic_df[topic_df["topic_label"] == selected_topic].copy()
                topic_guests = df_guests_with_topics[
                    df_guests_with_topics["topic_label"] == selected_topic
                ].copy()

                episode_count = len(topic_episodes)
                unique_guests = topic_guests["name"].nunique()

                date_series = (
                    topic_episodes["date"]
                    if "date" in topic_episodes.columns
                    else pd.Series(dtype="datetime64[ns]")
                )
                date_series = pd.to_datetime(date_series, errors="coerce") if not date_series.empty else date_series
                first_date = date_series.min() if not date_series.empty else None
                last_date = date_series.max() if not date_series.empty else None
                if pd.notna(first_date) and pd.notna(last_date):
                    timespan = f"{first_date.strftime('%d.%m.%Y')} – {last_date.strftime('%d.%m.%Y')}"
                elif pd.notna(last_date):
                    timespan = last_date.strftime("%d.%m.%Y")
                else:
                    timespan = "–"

                col1, col2, col3 = st.columns(3)
                col1.metric("Sendungen", episode_count)
                col2.metric("Zeitraum", timespan)
                col3.metric(
                    "Einzigartige Gäste",
                    int(unique_guests) if pd.notna(unique_guests) else 0,
                )

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
                )
                if "Datum" in episode_display.columns:
                    episode_display["Datum"] = pd.to_datetime(
                        episode_display["Datum"], errors="coerce"
                    )
                    episode_display.sort_values(
                        by="Datum", ascending=False, inplace=True, na_position="last"
                    )
                    episode_display["Datum"] = episode_display["Datum"].dt.strftime("%d.%m.%Y")
                    episode_display["Datum"] = episode_display["Datum"].fillna("–")
                if "Sendung" in episode_display.columns:
                    episode_display["Sendung"] = episode_display["Sendung"].fillna("–")
                if "Titel der Sendung" in episode_display.columns:
                    episode_display["Titel der Sendung"] = episode_display["Titel der Sendung"].fillna("–")
                if "Sender" in episode_display.columns:
                    episode_display["Sender"] = episode_display["Sender"].fillna("–")
                st.dataframe(episode_display, width="stretch", hide_index=True)

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

    elif selected_tab == "ℹ️ Über das Projekt":
        st.header("Über das Projekt")
        st.markdown(
            """
        Dieses Dashboard bietet eine interaktive Analyse von deutschen Polit-Talkshows. 
        Es nutzt Daten, die von den Webseiten der Sender und anderen öffentlichen Quellen gesammelt wurden.
        
        **Technologie-Stack:**
        - **Datensammlung:** Python mit `requests` und `BeautifulSoup`.
        - **Datenanalyse:** `pandas` für die Datenverarbeitung.
        - **Topic Modeling:** `BERTopic` mit `sentence-transformers`, `UMAP` und `HDBSCAN` zur Identifizierung und Visualisierung von Themen aus den Sendungsbeschreibungen.
        - **Netzwerkanalyse:** `networkx` zur Erstellung und Visualisierung von Gast- und Themen-Netzwerken.
        - **Dashboard:** `Streamlit` für die interaktive Webanwendung.
        
        **Analyseschritte:**
        1.  **Scraping:** Sammeln von Sendungsdaten (Titel, Datum, Beschreibung, Gäste).
        2.  **Datenbereinigung:** Konsolidierung von Gästenamen und Aufbereitung der Daten.
        3.  **Topic Modeling:** Analyse der Sendungsbeschreibungen, um die Kernthemen jeder Episode zu identifizieren.
        4.  **Netzwerkanalyse:**
            - **Gäste-Netzwerk:** Zeigt, welche Gäste häufig zusammen auftreten.
            - **Themen-Netzwerk:** Zeigt, welche Themen durch gemeinsame Gäste miteinander verbunden sind.
        5.  **Visualisierung:** Erstellung von interaktiven Grafiken und Tabellen zur einfachen Exploration der Ergebnisse.
        
        Das Projekt zielt darauf ab, Einblicke in die deutsche Medien- und Politiklandschaft zu geben, Trends aufzuzeigen und die Verbindungen zwischen Akteuren und Themen transparent zu machen.
        """
        )
else:
    st.info("Lade Daten... Falls dies länger dauert, stelle sicher, dass die Analysedateien im 'data'-Ordner vorhanden sind.")
