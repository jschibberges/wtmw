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

### Installation (conda, empfohlen)

```bash
git clone https://github.com/jschibberges/wtmw.git
cd wtmw

conda env create -f environment.yml
conda activate wtmw
```

`environment.yml` installiert Python 3.12, alle Pakete aus `requirements-dev.txt` (inkl. `pytest`) und das
spaCy-Modell `de_core_news_md`. Nach Änderungen an den requirements-Dateien:

```bash
conda env update -f environment.yml --prune
```

<details>
<summary>Ohne conda (pip)</summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m spacy download de_core_news_md
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
```

Das Dashboard ist dann unter http://localhost:8501 erreichbar.

Beim ersten Lauf lädt `analyze_talkshows.py` das Embedding-Modell `intfloat/multilingual-e5-base`
von Hugging Face (ca. 1 GB) und ggf. die Stopwort-Listen nach `data/`.

### Tests

```bash
pytest
```

### Automatische Aktualisierung (GitHub Actions)

`.github/workflows/pipeline.yml` („Weekly Pipeline“) scrapt und analysiert jeden **Montag um 04:00 Uhr
(Europe/Berlin)** und committet die aktualisierten Dateien in `data/` zurück ins Repo.

- GitHub führt Zeitpläne nur auf dem **Default-Branch** aus – der Workflow muss dort liegen.
- Manuell starten: *Actions → Weekly Pipeline → Run workflow*.
- Die Pipeline installiert das kleinere spaCy-Modell `de_core_news_sm`.

### Aktualisierung bestehender Folgen

Der Scraper lädt nur Folgen, die noch nicht in `data/{Show}_data.json` stehen. Ausnahme: Folgen der letzten
14 Tage (`REFRESH_RECENT_DAYS` in `scrape_talkshows.py`) werden bei jedem Lauf erneut abgerufen, damit
nachträglich ergänzte Gästelisten und Beschreibungen übernommen werden.

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
├── topic_labels.py                    # Topic-Labels über Retrainings stabil halten
├── sync_guest_validation_overrides.py # Manuelle Gast-Prüfung → guest_validation_overrides.json
│
├── environment.yml                    # conda-Umgebung (Python 3.12 + requirements-dev.txt + spaCy-Modell)
├── requirements.txt                   # Paketversionen (fest)
├── requirements-dev.txt               # + pytest
├── .github/workflows/pipeline.yml     # Weekly Pipeline (Scraping + Analyse + Commit)
│
├── data/
│   ├── {Show}_data.json               # Rohdaten pro Sendung
│   ├── all_data_with_topics.xlsx      # Hauptauswertung mit Themen
│   ├── guests_with_topics.xlsx        # Gäste mit Themen und Kategorien
│   ├── guest_topic_range.xlsx         # Ranking nach Themenbreite
│   ├── best_parameters.json           # Optimierte BERTopic-Hyperparameter
│   ├── topic_labels.json              # Menschenlesbare Topic-Namen (editierbar)
│   ├── name_corrections.json          # Manuelle Namenskorrekturen
│   ├── guest_validation_overrides.json # Manuelle Entscheidungen zur Gast-Validierung
│   ├── *.html / *.png / *.gexf        # Netzwerk- und Topic-Visualisierungen
│   └── talkshow_topic_model/          # Gespeichertes BERTopic-Modell
│
├── scripts/                           # Diagnose-Skripte (siehe unten)
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

Gäste werden in 10 Kategorien eingeteilt:

| Kategorie | Beispiele |
|---|---|
| Politics & Government | Abgeordnete, Minister, Botschafter |
| Media & Communication | Journalisten, Moderatoren, Chefredakteure |
| Academia & Expertise | Professoren, Forscher, Gutachter |
| Business & Economy | CEOs, Unternehmer, Ökonomen |
| Civil Society & Advocacy | Aktivisten, NGO-Vertreter, Gewerkschaften |
| Arts & Culture | Schauspieler, Autoren, Musiker |
| Sports | Sportler, Trainer |
| Religion & Spirituality | Theologen, Kirchenvertreter |
| Citizens & Everyday Voices | Betroffene, Privatpersonen |
| Influencers & Digital Creators | YouTuber, Podcaster, Influencer |

Die Klassifizierung läuft zweistufig: zuerst regelbasiert (Regex auf Rollenbezeichnungen + Parteizugehörigkeit), dann per Embedding-Ähnlichkeit zu Kategorie-Prototypen für nicht erkannte Gäste.

### Themenmodellierung

BERTopic mit:

- **Embeddings**: `intfloat/multilingual-e5-base` (mehrsprachig)
- **Dimensionsreduktion**: UMAP (automatisch getunt via Grid Search)
- **Clustering**: HDBSCAN (automatisch getunt)
- **Vokabular**: c-TF-IDF mit deutschen Stopwörtern + Rollenfilter (`TOPIC_ROLE_STOP`)
- **Labels**: Gespeichert in `data/topic_labels.json`, ohne Retraining editierbar; werden nach jedem Training per Keyword-Abgleich den neuen Topic-IDs zugeordnet

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
nach jedem Training über die Keywords den neuen IDs zu (`topic_labels.py`) und schreibt die Datei neu:

- **`"label": null`** — neues Topic ohne passendes Label. Einfach einen Namen eintragen.
- **`_unassigned`** — Labels, deren Topic im aktuellen Modell nicht mehr vorkommt (weniger als die Hälfte
  der Keywords stimmt überein). Sie werden bei jedem Lauf erneut geprüft.
- Ein Eintrag darf auch nur ein String sein (`"5": "Mein Label"`); die Keywords werden dann beim nächsten
  Lauf aus dem aktuell gespeicherten Modell ergänzt.

Änderungen am Label werden nach einem Streamlit-Neustart sichtbar — kein Retraining notwendig.

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
2. `python sync_guest_validation_overrides.py` überträgt die Entscheidungen nach
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

Tuning-Ergebnisse werden in `bertopic_tuning_results.csv` (im Projektverzeichnis) gespeichert.

**Hardware-Hinweis**: Das Modell nutzt automatisch Apple Silicon MPS, CUDA oder CPU. Auf MPS wird float16 verwendet; bei Speichermangel wird die Batch-Größe automatisch reduziert (64 → 1).

---

## Diagnose-Skripte

Aufruf aus dem Projektverzeichnis, z. B. `python scripts/compare_classification.py`.

| Skript | Zweck |
|---|---|
| `scripts/compare_classification.py` | Regelbasierte vs. Embedding-Klassifizierung vergleichen |
| `scripts/inspect_citizens.py` | Kategorie "Citizens & Everyday Voices" analysieren |
| `scripts/compare_topic_labels.py` | c-TF-IDF-Labels vs. KeyBERTInspired-Reranking vergleichen |
| `scripts/test_ngram_labels.py` | Unigramm- vs. Bigramm-Topic-Labels testen (kein pytest-Test) |

---

## Abhängigkeiten

Feste Versionen stehen in `requirements.txt`.

| Bereich | Pakete |
|---|---|
| Scraping | beautifulsoup4, requests |
| Daten | pandas, openpyxl |
| NLP | spacy (+ `de_core_news_md`), sentence-transformers, bertopic, umap-learn, hdbscan, scikit-learn |
| Visualisierung | streamlit, altair, matplotlib, networkx, pyvis |
| Konsole | rich |
| Tests | pytest |

---

## Lizenz

MIT
