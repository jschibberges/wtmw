# Wer talkt mit wem?

Eine Datenplattform zur Analyse von Gästen, Themen und Netzwerken in deutschen politischen Talkshows.

Ausgewertet werden: **Anne Will**, **Caren Miosga**, **Hart aber Fair**, **Markus Lanz**, **Maischberger** und **Maybrit Illner**.

---

## Was kann das Projekt?

- **Wer war wann wo?** — Alle Gäste aller Folgen auf einen Blick, filterbar nach Zeitraum und Sendung
- **Wer trifft wen?** — Netzwerkvisualisierung der Gäste-Ko-Auftritte
- **Worüber wurde geredet?** — Automatische Themenmodellierung via BERTopic
- **Wer kommt aus welchem Lager?** — Parteizuordnung und Gastklassifizierung (Politik, Wissenschaft, Medien, …)
- **Wer ist am vielseitigsten?** — Ranking nach Themenbreite der Gäste

---

## Schnellstart

### Voraussetzungen

- Python 3.10+
- [conda](https://docs.conda.io/) empfohlen für Umgebungsverwaltung

### Installation

```bash
git clone https://github.com/dein-user/wtmw.git
cd wtmw

pip install -r requirements.txt
python -m spacy download de_core_news_md
```

### Pipeline ausführen

```bash
# 1. Daten scrapen
python scrape_talkshows.py

# 2. Analyse & Themenmodellierung
python analyze_talkshows.py

# 3. Dashboard starten
streamlit run app.py
```

Das Dashboard ist dann unter http://localhost:8501 erreichbar.

---

## Projektstruktur

```
wtmw/
├── scrape_talkshows.py                # Web-Scraper (fernsehserien.de)
├── analyze_talkshows.py               # Analyse-Pipeline (BERTopic, Netzwerke, Klassifizierung)
├── nlp_utils.py                       # Deutsche NLP-Utilities (spaCy, Stopwörter, Lemmatisierung)
├── guest_classification.py            # Regelbasierte Gastklassifizierung
├── guest_classification_embedding.py  # Embedding-Fallback für Klassifizierung
├── app.py                             # Streamlit-Dashboard
├── app_helpers.py                     # Hilfsfunktionen für die App
│
├── data/
│   ├── {Show}_data.json               # Rohdaten pro Sendung
│   ├── all_data_with_topics.xlsx      # Hauptauswertung mit Themen
│   ├── guests_with_topics.xlsx        # Gäste mit Themen und Kategorien
│   ├── guest_topic_range.xlsx         # Ranking nach Themenbreite
│   ├── best_parameters.json           # Optimierte BERTopic-Hyperparameter
│   ├── topic_labels.json              # Menschenlesbare Topic-Namen (editierbar)
│   ├── name_corrections.json          # Manuelle Namenskorrekturen
│   └── talkshow_topic_model/          # Gespeichertes BERTopic-Modell
│
└── tests/                             # Unit-Tests
```

---

## Architektur

### Daten-Pipeline

```
fernsehserien.de
      │
      ▼
scrape_talkshows.py  ──►  {Show}_data.json
      │
      ▼
analyze_talkshows.py
  ├── Gastbereinigung & Konsolidierung  (nlp_utils.py)
  ├── Regelbasierte Klassifizierung     (guest_classification.py)
  ├── Embedding-Fallback                (guest_classification_embedding.py)
  ├── BERTopic-Themenmodellierung
  └── Netzwerkanalyse
      │
      ▼
  all_data_with_topics.xlsx
  guests_with_topics.xlsx
  talkshow_topic_model/
      │
      ▼
app.py  ──►  Streamlit-Dashboard
```

### Gastklassifizierung

Gäste werden in 9 Kategorien eingeteilt:

| Kategorie | Beispiele |
|---|---|
| Politics & Government | Abgeordnete, Minister, Botschafter |
| Media & Communication | Journalisten, Moderatoren, Chefredakteure |
| Academia & Expertise | Professoren, Forscher, Gutachter |
| Business & Economy | CEOs, Unternehmer, Ökonomen |
| Civil Society & Advocacy | Aktivisten, NGO-Vertreter, Gewerkschaften |
| Arts & Culture | Schauspieler, Autoren, Musiker |
| Sports | Sportler, Trainer |
| Religion | Theologen, Kirchenvertreter |
| Citizens & Everyday Voices | Betroffene, Privatpersonen |

Die Klassifizierung läuft zweistufig: zuerst regelbasiert (Regex auf Rollenbezeichnungen + Parteizugehörigkeit), dann per Embedding-Ähnlichkeit zu Kategorie-Prototypen für nicht erkannte Gäste.

### Themenmodellierung

BERTopic mit:

- **Embeddings**: `intfloat/multilingual-e5-base` (mehrsprachig)
- **Dimensionsreduktion**: UMAP (automatisch getunt via Grid Search)
- **Clustering**: HDBSCAN (automatisch getunt)
- **Vokabular**: c-TF-IDF mit deutschen Stopwörtern + Rollenfilter (`TOPIC_ROLE_STOP`)
- **Labels**: Gespeichert in `data/topic_labels.json`, ohne Retraining editierbar

---

## Konfiguration

### Topic-Namen anpassen

`data/topic_labels.json` enthält menschenlesbare Namen pro Topic-ID:

```json
{
  "0": "Ukraine-Krieg & Russland",
  "2": "Corona-Pandemie",
  "4": "Energiekrise & Klimapolitik"
}
```

Änderungen werden nach einem Streamlit-Neustart sichtbar — kein Retraining notwendig.

### Gästenamen korrigieren

```json
// data/name_corrections.json
{
  "Falsch Geschrieben": "Korrekt Geschrieben"
}
```

Danach `python analyze_talkshows.py` erneut ausführen.

### Neue Sendung hinzufügen

1. URL in `scrape_talkshows.py` ergänzen
2. JSON-Loader und `all_data`-Aggregation eintragen
3. Scraping-Aufruf in `main()` hinzufügen
4. `show_files`-Liste in `analyze_talkshows.py` aktualisieren

---

## BERTopic neu trainieren

```bash
# Hyperparameter-Tuning erzwingen:
rm data/best_parameters.json

python analyze_talkshows.py
```

Tuning-Ergebnisse werden in `data/bertopic_tuning_results.csv` gespeichert.

**Hardware-Hinweis**: Das Modell nutzt automatisch Apple Silicon MPS, CUDA oder CPU. Auf MPS wird float16 verwendet; bei Speichermangel wird die Batch-Größe automatisch reduziert (64 → 1).

---

## Diagnose-Skripte

| Skript | Zweck |
|---|---|
| `compare_classification.py` | Regelbasierte vs. Embedding-Klassifizierung vergleichen |
| `inspect_citizens.py` | Kategorie "Citizens & Everyday Voices" analysieren |
| `compare_topic_labels.py` | c-TF-IDF-Labels vs. KeyBERTInspired-Reranking vergleichen |
| `test_ngram_labels.py` | Unigramm- vs. Bigramm-Topic-Labels testen |

---

## Abhängigkeiten

| Bereich | Pakete |
|---|---|
| Scraping | beautifulsoup4, requests |
| Daten | pandas, openpyxl |
| NLP | spacy, sentence-transformers, bertopic, umap-learn, hdbscan, scikit-learn, safetensors |
| Visualisierung | streamlit, matplotlib, networkx, pyvis |
| Optional | rapidfuzz, cologne_phonetics |

---

## Lizenz

MIT
