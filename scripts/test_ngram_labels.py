"""
Testet Bigram-Labels für BERTopic-Topics ohne vollständiges Retraining.

Simuliert BERTopic's update_topics()-Schritt: lädt bestehende Dokument-Topic-
Zuordnungen aus all_data_with_topics.xlsx, wendet clean_description_for_labels()
an und berechnet c-TF-IDF mit verschiedenen n_gram_ranges.

Zeigt nebeneinander:
  (1,1)  Unigramm – aktueller Stand (aus topics.json)
  (1,2)  Bigram
  (1,3)  Trigram (wie im Code konfiguriert, aber bisher nicht im Output)

Ausführen:
    conda run -n mediaanalysis python test_ngram_labels.py
    conda run -n mediaanalysis python test_ngram_labels.py --show_roles  # ohne Rollenfilter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ---------------------------------------------------------------------------
# c-TF-IDF (BERTopic-Approximation)
# ---------------------------------------------------------------------------

def compute_ctfidf(
    docs_by_topic: dict[int, list[str]],
    ngram_range: tuple[int, int],
    top_n: int,
    min_df: int = 2,
    max_df: float = 0.85,
    extra_stopwords: set | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """
    Berechnet c-TF-IDF nach BERTopic-Methode:
      - Concateniert alle Dokumente eines Topics zu einem Meta-Dokument.
      - CountVectorizer über alle Meta-Dokumente.
      - TF = Term-Häufigkeit im Meta-Dokument (normalisiert auf Dok-Länge).
      - IDF = log(1 + Gesamt-Anz-Dokumente / Anz-Topics-mit-dem-Term).
    """
    topic_ids = sorted(k for k in docs_by_topic if k != -1)

    # Meta-Dokument pro Topic – Separator "|||" enthält keine Buchstaben,
    # wird vom token_pattern nicht gematcht und bricht damit Bigrams über
    # Episodengrenzen hinweg (verhindert "spd spd", "zdf spd" etc.)
    meta_docs = [" ||| ".join(docs_by_topic[tid]) for tid in topic_ids]

    stop_words: set[str] = set()
    try:
        from nlp_utils import get_german_stopwords, TOPIC_ROLE_STOP
        stop_words = get_german_stopwords()
        stop_words.update(TOPIC_ROLE_STOP)
    except ImportError:
        pass
    if extra_stopwords:
        stop_words.update(extra_stopwords)

    # CountVectorizer bekommt nur Token-Stopwörter (keine Multi-Token-Einträge),
    # da es sonst bei Bigram-Vocabulary eine UserWarning wirft.
    # Alles >1 Token aus der Stopwortliste wird ignoriert (betrifft kaum dt. Einzelwörter).
    single_token_stop = {w for w in stop_words if " " not in w}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Your stop_words may be inconsistent",
            category=UserWarning,
        )
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=list(single_token_stop),
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b[a-zäöüß]{3,}\b",  # Kleinbuchstaben ≥3 Zeichen inkl. Umlaute
        )
        tf_matrix = vectorizer.fit_transform(meta_docs)  # (n_topics, n_vocab)
    vocab = vectorizer.get_feature_names_out()

    # Normalisierung: tf / Dokumentlänge (wie BERTopic)
    row_sums = np.array(tf_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    tf_norm = tf_matrix.multiply(1.0 / row_sums[:, None])

    # IDF: log(1 + N / df), N = Anzahl Topics
    n_topics = len(topic_ids)
    df = np.array((tf_matrix > 0).sum(axis=0)).flatten()  # Dokumentfrequenz pro Term
    idf = np.log(1 + n_topics / (df + 1))  # +1 Smoothing

    ctfidf = tf_norm.multiply(idf).tocsr()  # csr für Row-Indexing

    results: dict[int, list[tuple[str, float]]] = {}
    for i, tid in enumerate(topic_ids):
        scores = np.array(ctfidf[i].todense()).flatten()
        top_idx = scores.argsort()[::-1][:top_n * 3]  # mehr holen, dann filtern
        terms = [(vocab[j], float(scores[j])) for j in top_idx if scores[j] > 0]
        results[tid] = terms[:top_n]

    return results


# ---------------------------------------------------------------------------
# Duplikat-Reduktion: kürzere Terme filtern wenn in längerem enthalten
# ---------------------------------------------------------------------------

def dedupe_terms(terms: list[tuple[str, float]], keep_n: int) -> list[tuple[str, float]]:
    """Entfernt Unigramme/Bigramme die Teilmenge eines längeren Terms sind."""
    sorted_by_len = sorted(terms, key=lambda x: (-len(x[0].split()), -x[1]))
    kept: list[tuple[str, float]] = []
    kept_token_sets: list[set] = []

    for word, score in sorted_by_len:
        toks = set(word.split())
        if any(toks <= ks for ks in kept_token_sets):
            continue
        kept.append((word, score))
        kept_token_sets.append(toks)
        if len(kept) >= keep_n:
            break

    return kept


# ---------------------------------------------------------------------------
# Rollenfilter + Personen-Namensfilter
# ---------------------------------------------------------------------------

def _get_role_stop() -> frozenset[str]:
    try:
        from nlp_utils import TOPIC_ROLE_STOP
        return TOPIC_ROLE_STOP
    except ImportError:
        return frozenset()


def _build_name_token_stop() -> frozenset[str]:
    """
    Lädt alle bereinigten Gastnamen aus guests_with_topics.xlsx (oder den JSON-Dateien)
    und extrahiert daraus individuelle Namens-Token (Vorname, Nachname, Doppelname).
    Diese werden als Stopwörter verwendet, damit Personennamen nicht als Topic-Terme
    auftauchen – Ortsnamen ("ukraine", "gaza") bleiben unberührt.
    """
    import re
    tokens: set[str] = set()

    # Primär: guests_with_topics.xlsx (hat name_clean-Spalte)
    guests_xlsx = DATA_DIR / "guests_with_topics.xlsx"
    if guests_xlsx.exists():
        try:
            df_g = pd.read_excel(guests_xlsx)
            name_col = next((c for c in ("name_clean", "name", "Name") if c in df_g.columns), None)
            if name_col:
                for name in df_g[name_col].dropna().unique():
                    for tok in re.findall(r"[a-zäöüß]{2,}", str(name).lower()):
                        tokens.add(tok)
                print(f"[namen] {len(tokens)} Namens-Token aus {len(df_g[name_col].dropna().unique())} Gästen geladen.")
                return frozenset(tokens)
        except Exception as e:
            print(f"[namen] guests_with_topics.xlsx nicht lesbar: {e}")

    # Fallback: JSON-Dateien
    import glob
    for path in glob.glob(str(DATA_DIR / "*_data.json")):
        try:
            with open(path, encoding="utf-8") as f:
                episodes = json.load(f)
            for ep in (episodes if isinstance(episodes, list) else []):
                for g in (ep.get("guests") or []):
                    name = g.get("name_clean") or g.get("name") or ""
                    for tok in re.findall(r"[a-zäöüß]{2,}", name.lower()):
                        tokens.add(tok)
        except Exception:
            pass

    # Sehr kurze oder häufige Vornamen rausschmeißen (< 3 Zeichen landen eh nicht im Vectorizer)
    tokens = {t for t in tokens if len(t) >= 3}
    print(f"[namen] {len(tokens)} Namens-Token aus JSON-Dateien geladen.")
    return frozenset(tokens)


def filter_roles(terms: list[tuple[str, float]], role_stop: frozenset) -> list[tuple[str, float]]:
    """Entfernt Terme die Rollenbezeichnungen oder Personennamen enthalten."""
    result = []
    for word, score in terms:
        tokens = word.lower().split()
        if any(t in role_stop for t in tokens):
            continue
        result.append((word, score))
    return result


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def load_data() -> tuple[dict[int, list[str]], dict[str, list]]:
    """Lädt Dokumente+Topiczuordnungen und aktuelle Terme aus topics.json."""
    # Aktuelle Terme aus gespeichertem Modell
    topics_json = DATA_DIR / "talkshow_topic_model" / "topics.json"
    with open(topics_json, encoding="utf-8") as f:
        saved = json.load(f)
    current_reps = saved["topic_representations"]
    topic_sizes = saved.get("topic_sizes", {})

    # Dokumente aus Excel
    xlsx = DATA_DIR / "all_data_with_topics.xlsx"
    print(f"Lade {xlsx.name}…")
    df = pd.read_excel(xlsx)

    # Spalten auto-detektieren
    text_col = next((c for c in ("description", "Description", "text", "Text") if c in df.columns), None)
    if text_col is None:
        text_col = next(c for c in df.columns if df[c].dtype == object)

    topic_col = next((c for c in ("topic", "Topic", "topic_id") if c in df.columns), None)
    if topic_col is None:
        raise ValueError(f"Keine Topic-Spalte. Spalten: {list(df.columns)}")

    print(f"Text: '{text_col}', Topic: '{topic_col}'")

    # Texte mit clean_description_for_labels aufbereiten.
    # LABEL_CONFIG schließt Verben aus (keep_pos = NOUN, PROPN, ADJ),
    # damit Verb-Bigrams wie "kommentieren erklären" nicht in Labels auftauchen.
    try:
        from nlp_utils import clean_description_for_labels, LABEL_CONFIG
        print("Bereinige Texte mit clean_description_for_labels(LABEL_CONFIG)…")
        texts_cleaned = df[text_col].fillna("").apply(
            lambda t: clean_description_for_labels(t, cfg=LABEL_CONFIG)
        ).tolist()
    except Exception as e:
        print(f"clean_description_for_labels nicht verfügbar ({e}), nutze Rohtext.")
        texts_cleaned = df[text_col].fillna("").str.lower().tolist()

    # Gruppieren
    df["_cleaned"] = texts_cleaned
    docs_by_topic: dict[int, list[str]] = {}
    for tid_raw, group in df.groupby(topic_col):
        try:
            tid = int(tid_raw)
        except (ValueError, TypeError):
            continue
        docs = group["_cleaned"].tolist()
        docs_by_topic[tid] = docs

    print(f"{len(docs_by_topic)} Topics, {len(df)} Dokumente gesamt.")
    return docs_by_topic, current_reps, topic_sizes


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main(top_n: int = 8, show_roles: bool = False) -> None:
    docs_by_topic, current_reps, topic_sizes = load_data()
    role_stop = frozenset() if show_roles else _get_role_stop()

    # Personennamen-Token als zusätzliche Stopwörter
    name_stop = frozenset() if show_roles else _build_name_token_stop()
    combined_stop = role_stop | name_stop

    print("\nBerechne c-TF-IDF für drei n_gram_ranges…")
    results: dict[str, dict[int, list[tuple[str, float]]]] = {}
    for label, ngr in [("(1,1) Uni", (1, 1)), ("(1,2) Bi ", (1, 2)), ("(1,3) Tri", (1, 3))]:
        # Namens-Token direkt in den Vectorizer geben (filtert bereits beim Vokabular-Aufbau)
        raw = compute_ctfidf(docs_by_topic, ngram_range=ngr, top_n=top_n * 2,
                             extra_stopwords=name_stop)
        # Rollenfilter + Dedupe (post-hoc, für Terme die der Vectorizer nicht erwischt)
        cleaned = {}
        for tid, terms in raw.items():
            filtered = filter_roles(terms, combined_stop)
            deduped = dedupe_terms(filtered, keep_n=top_n)
            cleaned[tid] = deduped
        results[label] = cleaned
        print(f"  {label}: fertig")

    # Ausgabe
    non_outlier_ids = sorted(k for k in docs_by_topic if k != -1)
    print("\n" + "=" * 100)
    print(f"  NGRAM-VERGLEICH  (top_n={top_n}, Rollenfilter={'aus' if show_roles else 'ein'})")
    print("=" * 100)

    for tid in non_outlier_ids:
        size = topic_sizes.get(str(tid), "?")

        # Aktuelles Label aus topics.json (zur Referenz)
        saved_words_raw = [w for w, _ in current_reps.get(str(tid), [])[:top_n]]
        saved_words = [w for w in saved_words_raw if w.lower() not in role_stop]
        saved_label = " | ".join(saved_words[:6]) or ("⚠ nur Rollen" if saved_words_raw else "–")

        print(f"\n  Topic {tid:>3}  (n={size:>4})")
        print(f"    SAVED: {saved_label}")

        for label, reps in results.items():
            terms = reps.get(tid, [])
            text = " | ".join(w for w, _ in terms[:6]) if terms else "–"
            print(f"    {label}: {text}")

    # Vorschlag: beste Labels (Bigramm bevorzugt)
    print("\n" + "=" * 100)
    print("  VORGESCHLAGENE LABELS FÜR topic_labels.json (bevorzugt Bigramme):")
    print("=" * 100)
    print("\n{")
    bi_reps = results["(1,2) Bi "]
    tri_reps = results["(1,3) Tri"]
    for tid in non_outlier_ids:
        size = topic_sizes.get(str(tid), "?")
        # Nimm Top-4 Terme aus Bigramm-Variante; Fallback auf Trigramm
        terms = bi_reps.get(tid, []) or tri_reps.get(tid, [])
        if terms:
            label_parts = [w.capitalize() for w, _ in terms[:4]]
            label = " · ".join(label_parts)
        else:
            label = f"Topic {tid}"
        print(f'  "{tid}": "{label}",  // n={size}')
    print("}")
    print("\nFertig.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--show_roles", action="store_true",
                        help="Rollenfilter deaktivieren (zeigt auch Außenpolitiker etc.)")
    args = parser.parse_args()
    main(top_n=args.top_n, show_roles=args.show_roles)
