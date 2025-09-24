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

# --- Data Loading Functions ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data():
    """Loads all necessary data files."""
    data_path = DATA_DIR / "all_data_with_topics.xlsx"
    guest_range_path = DATA_DIR / "guest_topic_range.xlsx"
    guests_with_topics_path = DATA_DIR / "guests_with_topics.xlsx"
    
    if not all([data_path.exists(), guest_range_path.exists(), guests_with_topics_path.exists()]):
        st.error("Daten nicht gefunden. Bitte führe zuerst 'analyze_talkshows.py' aus, um die Analysedateien zu generieren.")
        return None, None, None, None

    try:
        df_shows = pd.read_excel(data_path)
        df_guest_range = pd.read_excel(guest_range_path)
        df_guests_with_topics = pd.read_excel(guests_with_topics_path)
        
        # Calculate appearance counts from the detailed guest list
        appearance_counts = df_guests_with_topics['name'].value_counts().reset_index()
        appearance_counts.columns = ['name', 'appearances']
        
        return df_shows, df_guest_range, df_guests_with_topics, appearance_counts
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return None, None, None, None

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
df_shows, df_guest_range, df_guests_with_topics, appearance_counts = load_data()

if df_shows is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboards & Netzwerke", "🏆 Rankings", "🔍 Personensuche", "ℹ️ Über das Projekt"])

    with tab1:
        st.header("Dashboards & Netzwerke")
        st.markdown("Visualisierungen der Gast- und Themen-Netzwerke sowie der Themenanalyse.")

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

    with tab2:
        st.header("Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 25 Gäste nach Auftritten")
            st.dataframe(appearance_counts.head(25), use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Top 25 Gäste nach Themenvielfalt")
            st.dataframe(df_guest_range.head(25), use_container_width=True, hide_index=True)
            
        with st.expander("Gesamte Ranglisten anzeigen"):
            st.subheader("Alle Gäste nach Auftritten")
            st.dataframe(appearance_counts, use_container_width=True, hide_index=True)
            st.subheader("Alle Gäste nach Themenvielfalt")
            st.dataframe(df_guest_range, use_container_width=True, hide_index=True)

    with tab3:
        st.header("Personensuche")
        
        guest_names = sorted(df_guests_with_topics['name'].unique())
        search_term = st.selectbox("Wähle eine Person aus der Liste:", options=guest_names, index=None, placeholder="Namen auswählen...")

        if search_term:
            person_data = df_guests_with_topics[df_guests_with_topics['name'] == search_term]
            st.subheader(f"Analyse für: {search_term}")
            
            appearance_count = appearance_counts[appearance_counts['name'] == search_term]['appearances'].iloc[0]
            topic_range_data = df_guest_range[df_guest_range['name'] == search_term]
            topic_range_count = topic_range_data['topic_label'].iloc[0] if not topic_range_data.empty else 0
            
            col1, col2 = st.columns(2)
            col1.metric("Anzahl Auftritte", appearance_count)
            col2.metric("Themenvielfalt (Anzahl einzigartiger Themen)", topic_range_count)
            
            st.markdown("---")
            st.subheader("Alle Auftritte")
            
            display_df = person_data[['date', 'show', 'title', 'topic_label']].copy()
            display_df.rename(columns={'date': 'Datum', 'show': 'Sendung', 'title': 'Titel der Sendung', 'topic_label': 'Thema'}, inplace=True)
            display_df.sort_values(by='Datum', ascending=False, inplace=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab4:
        st.header("Über das Projekt")
        st.markdown("""
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
        """)
else:
    st.info("Lade Daten... Falls dies länger dauert, stelle sicher, dass die Analysedateien im 'data'-Ordner vorhanden sind.")
