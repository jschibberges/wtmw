# Entwicklerdokumentation

Technische Details zu Installation, Pipeline, Projektstruktur und Konfiguration von
**Wer talkt mit wem?**. Worum es inhaltlich geht, steht in der [README](../README.md).

---

## Installation & Ausführung

### Installation (conda, empfohlen)

```bash
git clone https://github.com/jschibberges/wtmw.git
cd wtmw

conda env create -f environment.yml
conda activate wtmw
```

`environment.yml` installiert Python 3.12 und alle Pakete aus `requirements-dev.txt` (inkl. `pytest` und
des spaCy-Modells `de_core_news_md`). Nach Änderungen an den requirements-Dateien:

```bash
conda env update -f environment.yml --prune
```

<details>
<summary>Ohne conda (pip)</summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```
</details>

Die Paketversionen sind in `requirements.txt` fest vorgegeben (getestet mit Python 3.12) und gelten
gleichermaßen für die lokale Umgebung und die Weekly Pipeline.

### Pipeline ausführen

```bash
# 1. Daten scrapen
python scrape_talkshows.py

# 2. Analyse & Themenmodellierung
python analyze_talkshows.py

# 3. Dashboard starten
streamlit run app.py

# Alternativ: Reflex-Dashboard starten
reflex run
```

Das Streamlit-Dashboard ist dann unter http://localhost:8501 erreichbar. Es hat denselben Aufbau wie das
Reflex-Dashboard (Übersicht, Gäste mit Personenprofil, Themen, Netzwerke) und nutzt dessen Daten-Builder
aus `wtmw_reflex/page_data.py`, sodass beide Dashboards dieselben Zahlen zeigen.
Das Reflex-Dashboard startet mit `reflex run` Frontend und Backend gemeinsam (http://localhost:3000).

Beim ersten Lauf lädt `analyze_talkshows.py` das Embedding-Modell `intfloat/multilingual-e5-base`
von Hugging Face (ca. 1 GB) und ggf. die Stopwort-Listen nach `data/`.

### Tests

```bash
pytest
```

Auf GitHub laufen die Tests automatisch bei jedem Push und bei Pull Requests nach `main`/`dev`
(`.github/workflows/tests.yml`, Python 3.12 mit den Versionen aus `requirements-dev.txt`).

### Automatische Aktualisierung (GitHub Actions)

`.github/workflows/pipeline.yml` („Weekly Pipeline“) scrapt und analysiert jeden **Montag um 04:00 Uhr
(Europe/Berlin)** und committet die aktualisierten Dateien in `data/` zurück ins Repo.

- GitHub führt Zeitpläne nur auf dem **Default-Branch** aus – der Workflow muss dort liegen.
- Manuell starten: *Actions → Weekly Pipeline → Run workflow*.
- Die Pipeline nutzt dieselben Paketversionen und dasselbe spaCy-Modell (`de_core_news_md`) wie die
  lokale Umgebung – beides kommt aus `requirements.txt`.

### Aktualisierung bestehender Folgen

Der Scraper lädt nur Folgen, die noch nicht in `data/{Show}_data.json` stehen. Ausnahme: Folgen der letzten
14 Tage (`REFRESH_RECENT_DAYS` in `scrape_talkshows.py`) werden bei jedem Lauf erneut abgerufen, damit
nachträglich ergänzte Gästelisten und Beschreibungen übernommen werden.

---

## Projektstruktur

```
wtmw/
├── scrape_talkshows.py        # 1. Scraper (fernsehserien.de → data/{Show}_data.json)
├── analyze_talkshows.py       # 2. Analyse (Gastbereinigung, Klassifizierung, BERTopic, Netzwerke)
├── app.py                     # 3a. Streamlit-Dashboard (gleiche Seiten wie Reflex)
├── .streamlit/config.toml     #     Streamlit-Theme (Farben des Reflex-Dashboards)
├── rxconfig.py                # 3b. Reflex-Konfiguration (`reflex run`)
│
├── wtmw/                      # Gemeinsame Bibliotheksmodule
│   ├── nlp_utils.py           #   Deutsche NLP-Utilities (spaCy, Stopwörter, Gastnamen-Bereinigung)
│   ├── guest_classification.py           # Regelbasierte Gastklassifizierung
│   ├── guest_classification_embedding.py # Embedding-Fallback für die Klassifizierung
│   ├── topic_labels.py        #   Topic-Labels über Retrainings stabil halten
│   ├── date_utils.py          #   Datums-Parsing und Analysezeitraum
│   └── app_helpers.py         #   Gemeinsame Helfer beider Dashboards (Topic-Namen, Gast-Metadaten)
│
├── wtmw_reflex/               # Reflex-Dashboard
│   ├── wtmw_reflex.py         #   Seiten, State und Routen
│   ├── dashboard_data.py      #   Laden, Filtern, Netzwerk-Assets
│   └── page_data.py           #   Daten-Builder für Übersicht, Gäste, Themen, Profil (auch von app.py genutzt)
├── assets/                    # CSS für das Reflex-Dashboard
│   ├── reflex_dashboard.css        # Basis-Styling (Tokens, Navigation, Karten)
│   └── reflex_dashboard_pages.css  # Seiten-Layouts (wird danach geladen)
│
├── data/                      # Rohdaten, Analyse-Ergebnisse, Modell, manuelle Korrekturen
│   ├── {Show}_data.json       #   Rohdaten pro Sendung
│   ├── all_data_with_topics.xlsx   # Hauptauswertung mit Themen
│   ├── guests_with_topics.xlsx     # Gäste mit Themen und Kategorien
│   ├── guest_topic_range.xlsx      # Ranking nach Themenbreite
│   ├── best_parameters.json        # Optimierte BERTopic-Hyperparameter
│   ├── bertopic_tuning_results.csv # Alle Kombinationen des letzten Tunings
│   ├── topic_labels.json           # Menschenlesbare Topic-Namen (editierbar)
│   ├── name_corrections.json       # Manuelle Namenskorrekturen
│   ├── guest_validation_overrides.json # Manuelle Entscheidungen zur Gast-Validierung
│   ├── *.html / *.png / *.gexf     # Netzwerk- und Topic-Visualisierungen
│   └── talkshow_topic_model/       # Gespeichertes BERTopic-Modell
│
├── scripts/                   # Wartungs- und Diagnose-Skripte (siehe unten)
├── tests/                     # pytest-Tests
│
├── environment.yml            # conda-Umgebung (Python 3.12 + requirements-dev.txt)
├── requirements.txt           # Paketversionen (fest)
├── requirements-dev.txt       # + pytest
└── .github/workflows/         # Weekly Pipeline (pipeline.yml) und Tests (tests.yml)
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
  ├── Gastbereinigung & Konsolidierung  (wtmw/nlp_utils.py)
  ├── Regelbasierte Klassifizierung     (wtmw/guest_classification.py)
  ├── Embedding-Fallback                (wtmw/guest_classification_embedding.py)
  ├── BERTopic-Themenmodellierung
  └── Netzwerkanalyse
      │
      ▼
  all_data_with_topics.xlsx
  guests_with_topics.xlsx
  talkshow_topic_model/
      │
      ▼
app.py / wtmw_reflex/  ──►  Streamlit- bzw. Reflex-Dashboard
```

### Gastklassifizierung

Die zehn Kategorien sind in der [README](../README.md#wer-sitzt-da) beschrieben.

Die Klassifizierung läuft zweistufig: zuerst regelbasiert (Regex auf Rollenbezeichnungen + Parteizugehörigkeit), dann per Embedding-Ähnlichkeit zu Kategorie-Prototypen für nicht erkannte Gäste.

### Themenmodellierung

BERTopic mit:

- **Embeddings**: `intfloat/multilingual-e5-base` (mehrsprachig)
- **Dimensionsreduktion**: UMAP (automatisch getunt via Grid Search)
- **Clustering**: HDBSCAN (automatisch getunt)
- **Vokabular**: c-TF-IDF mit deutschen Stopwörtern + Rollenfilter (`TOPIC_ROLE_STOP`)
- **Labels**: Gespeichert in `data/topic_labels.json`, ohne Retraining editierbar; werden nach jedem Training über gemeinsame Folgen (Fallback: Keywords) den neuen Topic-IDs zugeordnet

---

## Konfiguration

### Topic-Namen anpassen

`data/topic_labels.json` enthält menschenlesbare Namen pro Topic-ID, zusammen mit den Keywords des Topics:

```json
{
  "-1": "Sonstige / Nicht zugeordnet",
  "0": {"label": "Ukraine-Krieg & Russland", "keywords": ["ukrainisch", "russisch", "putin", "..."]},
  "7": {"label": null, "keywords": ["..."]},
  "_unassigned": [{"label": "Brexit & Großbritannien", "keywords": ["brexit", "..."]}]
}
```

Da BERTopic die Topic-IDs bei jedem Training neu vergibt, ordnet `analyze_talkshows.py` die Labels
nach jedem Training den neuen IDs zu (`wtmw/topic_labels.py`) und schreibt die Datei neu:

1. **Über die Folgen:** Ein neues Topic übernimmt das Label des alten Topics, mit dem es die meisten
   Folgen teilt (Grundlage: `all_data_with_topics.xlsx` des letzten Laufs, mind. 30 % Überlappung).
   Teilt sich ein Topic, erhält der größere Teil das Label.
2. **Über die Keywords** (Fallback): für geparkte Labels und falls die alte Zuordnung fehlt oder nicht
   zum gespeicherten Modell passt (mind. die Hälfte der Keywords gleich).

Das Log zeigt für jedes Label, wohin es gewandert ist und auf welchem Weg. Danach in der Datei prüfen:

- **`"label": null`** — neues Topic ohne passendes Label. Einfach einen Namen eintragen.
- **`_unassigned`** — Labels, deren Topic im aktuellen Modell nicht mehr vorkommt. Sie werden bei jedem
  Lauf erneut (über die Keywords) geprüft.
- Ein Eintrag darf auch nur ein String sein (`"5": "Mein Label"`); die Keywords werden dann beim nächsten
  Lauf aus dem aktuell gespeicherten Modell ergänzt.

Änderungen am Label werden nach einem Neustart des Dashboards sichtbar — kein Retraining notwendig.

### Gästenamen korrigieren

```json
// data/name_corrections.json
{
  "Falsch Geschrieben": "Korrekt Geschrieben"
}
```

Danach `python analyze_talkshows.py` erneut ausführen.

### Gast-Validierung prüfen

Der Scraper bewertet jeden Gastnamen als `accept`, `review` oder `reject` (z. B. werden Werbetexte oder
Organisationsnamen verworfen). Unsichere Fälle landen in `data/guest_validation_review.xlsx`:

1. In der Excel-Datei die Spalte `manual_status` auf `accept` oder `reject` setzen (optional `manual_note`).
2. `python scripts/sync_guest_validation_overrides.py` überträgt die Entscheidungen nach
   `data/guest_validation_overrides.json`.
3. Die Overrides gelten ab dem nächsten Scraping – für bereits gespeicherte Folgen nur, wenn diese erneut
   abgerufen werden (z. B. innerhalb des 14-Tage-Fensters).

### Neue Sendung hinzufügen

1. Episodenguide-URL in `scrape_talkshows.py` ergänzen (`alternative_url_…`)
2. Modulvariable, Laden in `_load_existing_data()` und die `all_data`-Aggregation eintragen
3. Scraping-Aufruf und Statistik in `main()` hinzufügen
4. `show_files`-Liste in `load_all_show_data()` (`analyze_talkshows.py`) aktualisieren

---

## BERTopic neu trainieren

Das Topic-Modell wird bei jedem Lauf von `analyze_talkshows.py` neu trainiert. Die Hyperparameter aus
`data/best_parameters.json` werden automatisch neu getunt (Grid Search über 144 Kombinationen), wenn die
Datei fehlt oder `document_count` darin kleiner als 80 % der aktuellen Episodenzahl ist
(`get_or_tune_parameters()` in `analyze_talkshows.py`).

```bash
# Hyperparameter-Tuning erzwingen:
rm data/best_parameters.json

python analyze_talkshows.py
```

Tuning-Ergebnisse werden in `data/bertopic_tuning_results.csv` gespeichert.

**Hardware-Hinweis**: Das Modell nutzt automatisch Apple Silicon MPS, CUDA oder CPU. Auf MPS wird float16 verwendet; bei Speichermangel wird die Batch-Größe automatisch reduziert (64 → 1).

---

## Wartungs- und Diagnose-Skripte

Aufruf aus dem Projektverzeichnis, z. B. `python scripts/compare_classification.py`.

| Skript | Zweck |
|---|---|
| `scripts/compare_classification.py` | Regelbasierte vs. Embedding-Klassifizierung vergleichen |
| `scripts/inspect_citizens.py` | Kategorie "Citizens & Everyday Voices" analysieren |
| `scripts/compare_topic_labels.py` | c-TF-IDF-Labels vs. KeyBERTInspired-Reranking vergleichen |
| `scripts/compare_ngram_labels.py` | Unigramm- vs. Bigramm-/Trigramm-Topic-Labels vergleichen |
| `scripts/sync_guest_validation_overrides.py` | Manuelle Gast-Prüfung → `data/guest_validation_overrides.json` |

---

## Abhängigkeiten

Feste Versionen stehen in `requirements.txt`.

| Bereich | Pakete |
|---|---|
| Scraping | beautifulsoup4, requests |
| Daten | pandas, openpyxl |
| NLP | spacy (+ `de_core_news_md`), sentence-transformers, bertopic, umap-learn, hdbscan, scikit-learn |
| Visualisierung | streamlit, reflex, matplotlib, networkx, pyvis |
| Konsole | rich |
| Tests | pytest |
