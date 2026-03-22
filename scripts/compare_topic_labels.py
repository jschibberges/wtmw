"""
Vergleich: BERTopic-Standard-Keywords vs. KeyBERTInspired-Terme.

Lädt das gespeicherte Modell (topics.json + topic_embeddings.safetensors),
berechnet für jedes Topic KeyBERTInspired-Terme und zeigt beide Label-Varianten
nebeneinander.

Wichtig: Als Kandidatenmenge werden die c-TF-IDF-Terme aus topics.json
verwendet (nicht Rohvokabular), da c-TF-IDF Stoppwörter und generische
Terme bereits herausfiltert. KeyBERTInspired re-rankt diese Kandidaten
dann semantisch per Kosinus-Ähnlichkeit zum Topic-Centroid + MMR.

Ausführen:
    conda run -n mediaanalysis python compare_topic_labels.py

Optional: --top_n N          Anzahl anzuzeigender Terme (Standard: 8)
          --mmr_lambda L      Diversitäts-Trade-off 0.0–1.0 (Standard: 0.6)
          --candidates N      Anzahl c-TF-IDF-Kandidaten pro Topic (Standard: 25)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "talkshow_topic_model"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidates: list[str],
    top_n: int,
    lambda_: float,
) -> list[str]:
    """Maximal Marginal Relevance: Diversifiziert die Top-Terme."""
    if len(candidates) == 0:
        return []

    scores = np.array([_cosine_sim(query_vec, v) for v in candidate_vecs])
    selected_idx: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_n, len(candidates))):
        if not selected_idx:
            # Erstes Element: bestes Relevanz-Ergebnis
            best = int(np.argmax(scores[remaining]))
            selected_idx.append(remaining[best])
            remaining.remove(remaining[best])
        else:
            # MMR-Score: Balance zwischen Relevanz und Diversität
            sel_vecs = candidate_vecs[selected_idx]
            mmr_scores = []
            for i in remaining:
                rel = scores[i]
                max_sim = max(_cosine_sim(candidate_vecs[i], sel_vecs[j])
                               for j in range(len(sel_vecs)))
                mmr_scores.append(lambda_ * rel - (1 - lambda_) * max_sim)
            best_local = int(np.argmax(mmr_scores))
            best_global = remaining[best_local]
            selected_idx.append(best_global)
            remaining.remove(best_global)

    return [candidates[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# Post-hoc-Filter: Rollenbeschreibungen aus Term-Listen entfernen
# ---------------------------------------------------------------------------

def _load_role_stop() -> frozenset[str]:
    """Importiert TOPIC_ROLE_STOP aus nlp_utils, falls verfügbar."""
    try:
        from nlp_utils import TOPIC_ROLE_STOP
        return TOPIC_ROLE_STOP
    except ImportError:
        # Minimale Inline-Fallback-Liste für den Fall, dass nlp_utils nicht importierbar
        return frozenset({
            "politiker", "politikerin", "außenpolitiker", "außenpolitikerin",
            "vorsitzende", "vorsitzender", "parteivorsitzende", "parteivorsitzender",
            "fraktionsvorsitzende", "fraktionsvorsitzender",
            "generalsekretär", "generalsekretärin",
            "bundeskanzler", "bundeskanzlerin", "minister", "ministerin",
            "bürgermeister", "bürgermeisterin", "landrat", "landrätin",
            "senator", "senatorin", "botschafter", "botschafterin",
            "experte", "expertin", "wissenschaftler", "wissenschaftlerin",
            "forscher", "forscherin", "professor", "professorin",
            "militärexperte", "militärexpertin",
            "politikwissenschaftler", "politikwissenschaftlerin",
            "politologe", "politologin", "soziologe", "soziologin",
            "historiker", "historikerin", "ökonom", "ökonomin",
            "epidemiologe", "epidemiologin", "virologe", "virologin",
            "islamwissenschaftler", "islamwissenschaftlerin",
            "terrorismusexperte", "terrorismusexpertin",
            "extremismusforscher", "extremismusforscherin",
            "verfassungsrechtler", "verfassungsrechtlerin",
            "jurist", "juristin", "kriminologe", "kriminologin",
            "journalist", "journalistin", "reporter", "reporterin",
            "korrespondent", "korrespondentin",
            "moderator", "moderatorin", "chefredakteur", "chefredakteurin",
            "kommentator", "kommentatorin", "publizist", "publizistin",
            "unternehmer", "unternehmerin", "manager", "managerin",
            "aktivist", "aktivistin", "sprecher", "sprecherin",
            "autor", "autorin", "arzt", "ärztin", "chefarzt", "chefärztin",
            "mitglied", "auswärtigen",
        })


_ROLE_STOP: frozenset[str] = _load_role_stop()


def _filter_role_words(terms: list[str], extra_slots: int = 3) -> list[str]:
    """
    Entfernt Rollenbeschreibungen aus einer Term-Liste und füllt ggf. Lücken
    aus dem restlichen Pool auf (extra_slots: wie viele Reserveeinträge genutzt werden).
    """
    return [t for t in terms if t.lower() not in _ROLE_STOP]


# ---------------------------------------------------------------------------
# KeyBERTInspired-Berechnung auf Basis von c-TF-IDF-Kandidaten
# ---------------------------------------------------------------------------

def compute_keybert_inspired(
    topic_representations: dict[str, list],
    topic_embeddings: dict[int, np.ndarray],
    embed_fn,
    top_n: int = 8,
    candidates_per_topic: int = 25,
    mmr_lambda: float = 0.6,
) -> dict[int, list[str]]:
    """
    KeyBERTInspired auf Basis der c-TF-IDF-Terme aus topics.json:
      1. Verwendet die top-candidates_per_topic c-TF-IDF-Terme als Kandidaten.
         (Diese sind bereits stoppwort-gefiltert und diskriminativ.)
      2. Embedded alle einzigartigen Kandidat-Terme in einem Batch.
      3. Re-rankt pro Topic per Kosinus-Ähnlichkeit zum Topic-Centroid.
      4. Diversifiziert mit MMR.

    Vorteil gegenüber Rohvokabular: Keine generischen Wörter wie "wird",
    "hat", "die" da c-TF-IDF diese bereits herausfiltert.
    """
    # Alle einzigartigen Kandidat-Terme sammeln (ohne Rollenbeschreibungen)
    all_candidates: set[str] = set()
    for tid_str, word_scores in topic_representations.items():
        if tid_str == "-1":
            continue
        for w, _ in word_scores[:candidates_per_topic]:
            if w.lower() not in _ROLE_STOP:
                all_candidates.add(w)

    all_candidates_list = sorted(all_candidates)
    print(f"[keybert] Embedde {len(all_candidates_list)} c-TF-IDF-Kandidaten "
          f"(nach Rollenfilter, vorher: {sum(len(ws[:candidates_per_topic]) for ws in topic_representations.values() if ws)})…")
    candidate_vecs = embed_fn(all_candidates_list)  # (n_candidates, dim)
    w2emb = {w: candidate_vecs[i] for i, w in enumerate(all_candidates_list)}

    results: dict[int, list[str]] = {}

    for tid_str, word_scores in topic_representations.items():
        if tid_str == "-1":
            continue
        topic_id = int(tid_str)

        topic_vec = topic_embeddings.get(topic_id)
        if topic_vec is None:
            # Fallback: STD-Terme direkt übernehmen (gefiltert)
            results[topic_id] = _filter_role_words([w for w, _ in word_scores])[:top_n]
            continue

        candidates = [w for w, _ in word_scores[:candidates_per_topic]
                      if w in w2emb]
        if not candidates:
            results[topic_id] = _filter_role_words([w for w, _ in word_scores])[:top_n]
            continue

        cand_vecs = np.stack([w2emb[w] for w in candidates])
        top_terms = _mmr(topic_vec, cand_vecs, candidates, top_n, mmr_lambda)
        results[topic_id] = top_terms

    return results


# ---------------------------------------------------------------------------
# Laden der gespeicherten Daten
# ---------------------------------------------------------------------------

def load_topics_json() -> dict:
    path = MODEL_DIR / "topics.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_topic_embeddings() -> dict[int, np.ndarray]:
    """Lädt die Topic-Centroid-Vektoren aus dem SafeTensors-Format."""
    try:
        from safetensors.numpy import load_file
        raw = load_file(MODEL_DIR / "topic_embeddings.safetensors")
        # Üblicher Key ist "topic_embeddings" mit Shape (n_topics, dim)
        arr = next(iter(raw.values()))  # (n_topics, dim)
        print(f"[load] Topic-Embeddings geladen: shape={arr.shape}")
        return arr
    except Exception as e:
        print(f"[load] Konnte topic_embeddings.safetensors nicht laden: {e}")
        return None


def load_docs_from_xlsx() -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """Lädt all_data_with_topics.xlsx und gruppiert Beschreibungen pro Topic."""
    xlsx_path = DATA_DIR / "all_data_with_topics.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {xlsx_path}")

    print(f"[load] Lade {xlsx_path.name}…")
    df = pd.read_excel(xlsx_path)

    # Spalte mit Beschreibungstext finden
    text_col = None
    for candidate in ("description", "Description", "text", "Text", "title", "Title"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # Fallback: erste String-Spalte
        str_cols = [c for c in df.columns if df[c].dtype == object]
        text_col = str_cols[0] if str_cols else df.columns[0]
    print(f"[load] Text-Spalte: '{text_col}', Topic-Spalte wird gesucht…")

    topic_col = None
    for candidate in ("topic", "Topic", "topic_id"):
        if candidate in df.columns:
            topic_col = candidate
            break
    if topic_col is None:
        raise ValueError(f"Keine Topic-Spalte gefunden. Vorhandene Spalten: {list(df.columns)}")
    print(f"[load] Topic-Spalte: '{topic_col}'")

    docs_by_topic: dict[int, list[str]] = {}
    for topic_id, group in df.groupby(topic_col):
        try:
            tid = int(topic_id)
        except (ValueError, TypeError):
            continue
        texts = group[text_col].dropna().astype(str).tolist()
        docs_by_topic[tid] = texts

    print(f"[load] {len(docs_by_topic)} Topics mit insgesamt "
          f"{sum(len(v) for v in docs_by_topic.values())} Dokumenten geladen.")
    return df, docs_by_topic


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main(top_n: int = 8, mmr_lambda: float = 0.6, candidates: int = 25) -> None:
    # 1. Daten laden
    topics_data = load_topics_json()
    current_reps: dict[str, list] = topics_data["topic_representations"]
    topic_sizes: dict[str, int] = topics_data.get("topic_sizes", {})

    topic_emb_arr = load_topic_embeddings()

    # docs_by_topic nur noch für den Fallback (Topic-Centroid aus Docs) benötigt
    _, docs_by_topic = load_docs_from_xlsx()

    # 2. Embedding-Modell laden
    print("\n[embed] Lade Embedding-Modell (intfloat/multilingual-e5-base)…")
    try:
        from analyze_talkshows import _get_embedder, _embed_texts
        model_name = "intfloat/multilingual-e5-base"
        embedder, _ = _get_embedder(model_name)
        embed_fn = lambda texts: _embed_texts(embedder, texts)
    except Exception as e:
        print(f"[embed] Konnte Embedder nicht laden: {e}")
        print("[embed] Versuche sentence_transformers direkt…")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/multilingual-e5-base")
        embed_fn = lambda texts: _model.encode(texts, normalize_embeddings=True,
                                                show_progress_bar=False)

    # 3. Topic-Centroid-Vektoren aufbereiten
    #    topic_emb_arr hat Shape (n_topics+1, dim) – Index 0 = Topic -1 (Outlier)
    #    Die Reihenfolge entspricht der aufsteigend sortierten Topic-ID-Liste
    topic_ids_sorted = sorted(
        [int(k) for k in current_reps.keys()], key=lambda x: x
    )

    if topic_emb_arr is not None:
        # Mapping: topic_id → embedding-Vektor
        # BERTopic speichert Outlier-Topic (-1) als erstes, dann 0, 1, 2, …
        ordered_ids = sorted(topic_ids_sorted)  # inkl. -1
        topic_embeddings: dict[int, np.ndarray] = {}
        for i, tid in enumerate(ordered_ids):
            if i < len(topic_emb_arr):
                topic_embeddings[tid] = topic_emb_arr[i]
    else:
        # Fallback: Mittelwert der Dokument-Embeddings pro Topic berechnen
        print("[embed] Berechne Topic-Centroids aus Dokumenten…")
        topic_embeddings = {}
        for tid, docs in docs_by_topic.items():
            if not docs:
                continue
            vecs = embed_fn(docs[:200])  # max 200 Docs für Geschwindigkeit
            topic_embeddings[tid] = vecs.mean(axis=0)

    # 4. KeyBERTInspired-Terme berechnen
    #    Kandidaten = c-TF-IDF-Terme aus topics.json (bereits stoppwort-gefiltert)
    print(f"\n[keybert] Berechne KeyBERTInspired-Terme "
          f"(top_n={top_n}, mmr_lambda={mmr_lambda}, candidates={candidates})…")
    print("[keybert] Kandidaten = c-TF-IDF-Terme (stoppwort-gefiltert, kein Rohvokabular)")
    keybert_terms = compute_keybert_inspired(
        topic_representations=current_reps,
        topic_embeddings=topic_embeddings,
        embed_fn=embed_fn,
        top_n=top_n,
        candidates_per_topic=candidates,
        mmr_lambda=mmr_lambda,
    )

    # 5. Vergleichsausgabe
    print("\n" + "=" * 90)
    print(f"  VERGLEICH: BERTopic-Standard-Keywords  vs.  KeyBERTInspired (top_n={top_n}, λ={mmr_lambda})")
    print("=" * 90)
    print(f"\n  {'Topic':>6}  {'n':>5}  {'BERTopic-Standard':<40}  KeyBERTInspired")
    print(f"  {'-'*6}  {'-'*5}  {'-'*40}  {'-'*40}")

    non_outlier_ids = [tid for tid in topic_ids_sorted if tid != -1]

    for tid in non_outlier_ids:
        size = topic_sizes.get(str(tid), "?")
        # Standard-Terme (BERTopic c-TF-IDF) – ebenfalls Rollenfilter für faire Vergleichbarkeit
        std_words_raw = [w for w, _ in current_reps.get(str(tid), [])]
        std_words_filtered = _filter_role_words(std_words_raw)[:top_n]
        std_label = " | ".join(std_words_filtered[:6]) or " | ".join(std_words_raw[:6]) + "  ⚠ nur Rollen"

        # KeyBERTInspired-Terme (bereits gefiltert durch compute_keybert_inspired)
        kb_words = keybert_terms.get(tid, [])
        kb_label = " | ".join(kb_words[:6]) if kb_words else "–"

        print(f"\n  Topic {tid:>3}  (n={size:>4})")
        print(f"    STD:  {std_label}")
        print(f"    KB :  {kb_label}")

    # 6. Qualitätsmetriken
    print("\n" + "=" * 90)
    print("  ÜBERSCHNEIDUNGSANALYSE (welche STD-Terme erscheinen auch in KB?)")
    print("=" * 90)
    overlaps = []
    for tid in non_outlier_ids:
        std_words = {w for w, _ in current_reps.get(str(tid), [])[:top_n]}
        kb_words = set(keybert_terms.get(tid, []))
        if std_words and kb_words:
            overlap = len(std_words & kb_words) / len(std_words | kb_words)
            overlaps.append(overlap)

    if overlaps:
        print(f"  Jaccard-Ähnlichkeit (Mittel): {np.mean(overlaps):.3f}")
        print(f"  Jaccard-Ähnlichkeit (Min):    {np.min(overlaps):.3f}")
        print(f"  Jaccard-Ähnlichkeit (Max):    {np.max(overlaps):.3f}")
        print(f"\n  → Niedrige Werte = KB bringt neue Terme (mehr Diversität).")
        print(f"    Hohe Werte = KB bestätigt STD-Terme (gute Konsistenz).")

    # 7. Vorschlag: Verbesserte Human-Readable Labels
    print("\n" + "=" * 90)
    print("  VORGESCHLAGENE HUMAN-READABLE LABELS (aus KeyBERTInspired)")
    print("=" * 90)
    print(f"\n  Kopiere die folgenden Einträge in data/topic_labels.json:\n")
    print("  {")
    for tid in non_outlier_ids:
        kb_words = keybert_terms.get(tid, [])
        if kb_words:
            # Erste 4 Terme für Label-Vorschlag, capitalized
            label_parts = [w.capitalize() for w in kb_words[:4]]
            label = " · ".join(label_parts)
        else:
            std_words = [w for w, _ in current_reps.get(str(tid), [])[:4]]
            label = " · ".join(w.capitalize() for w in std_words)
        size = topic_sizes.get(str(tid), "?")
        print(f'    "{tid}": "{label}",  // n={size}')
    print("  }")

    print("\nFertig.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=8,
                        help="Anzahl Terme pro Topic (Standard: 8)")
    parser.add_argument("--mmr_lambda", type=float, default=0.6,
                        help="MMR-Diversität: 0=max. Diversität, 1=nur Relevanz (Standard: 0.6)")
    parser.add_argument("--candidates", type=int, default=25,
                        help="Anzahl c-TF-IDF-Kandidaten pro Topic (Standard: 25, max. sinnvoll: top_n_words aus BERTopic-Config)")
    args = parser.parse_args()
    main(top_n=args.top_n, mmr_lambda=args.mmr_lambda, candidates=args.candidates)
