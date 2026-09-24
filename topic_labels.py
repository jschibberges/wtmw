"""
Kuratierte Topic-Namen stabil über Retrainings hinweg halten.

BERTopic vergibt die Topic-IDs bei jedem Training neu. Ein Label, das nur an
der ID hängt ("0": "Ukraine-Krieg"), landet nach dem nächsten Training ggf.
auf einem ganz anderen Cluster. Deshalb speichert data/topic_labels.json zu
jedem Label die Keywords des Topics, und nach jedem Training werden die Labels
per Keyword-Überlappung den neuen IDs zugeordnet.

Format von data/topic_labels.json:

    {
      "-1": "Sonstige / Nicht zugeordnet",
      "0": {"label": "Ukraine-Krieg & Russland", "keywords": ["ukrainisch", ...]},
      "7": {"label": null, "keywords": [...]},          # neues Topic, noch unbenannt
      "_unassigned": [{"label": "...", "keywords": [...]}]  # aktuell ohne Treffer
    }

Ein Eintrag darf auch nur ein String sein ("5": "Name"). Die Keywords werden
dann beim nächsten Lauf aus dem zuletzt gespeicherten Modell ergänzt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

OUTLIER_ID = "-1"
UNASSIGNED_KEY = "_unassigned"
# Verschiedene Topics überlappen typischerweise <= 0.2 (Maximum im aktuellen
# Modell: 0.4). Lieber ein Label parken als es falsch zuordnen.
MIN_SIMILARITY = 0.5
MAX_KEYWORDS = 10


def _tokens(keywords: Iterable[str]) -> set[str]:
    """Split keyword phrases into a lower-cased token set."""
    return {tok for kw in keywords for tok in str(kw).lower().split() if tok}


def keyword_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    """Overlap coefficient of the keyword tokens (0 = disjoint, 1 = subset)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def load_curated_labels(path: Path | str) -> dict:
    """Load topic_labels.json; returns {} if missing or malformed."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def label_text(entry: object) -> str | None:
    """Extract the display label from a string or dict entry."""
    if isinstance(entry, str):
        return entry.strip() or None
    if isinstance(entry, dict):
        label = entry.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return None


def labels_by_id(data: dict) -> dict[str, str]:
    """Flatten topic_labels.json to {topic_id: label} for display."""
    result: dict[str, str] = {}
    for key, entry in data.items():
        if key.startswith("_"):
            continue
        label = label_text(entry)
        if label:
            result[str(key)] = label
    return result


def load_model_keywords(topics_json_path: Path | str) -> dict[str, list[str]]:
    """Read {topic_id: keywords} from a saved BERTopic topics.json."""
    path = Path(topics_json_path)
    if not path.exists():
        return {}
    try:
        reps = json.loads(path.read_text(encoding="utf-8")).get("topic_representations", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return {
        str(tid): [term[0] if isinstance(term, (list, tuple)) else str(term) for term in terms][:MAX_KEYWORDS]
        for tid, terms in reps.items()
        if terms
    }


def _candidates(curated: dict, previous_keywords: dict[str, list[str]]) -> list[dict]:
    """Collect all labelled entries (incl. unassigned) with their keyword anchors."""
    candidates: list[dict] = []
    for key, entry in curated.items():
        if key == OUTLIER_ID or key.startswith("_"):
            continue
        label = label_text(entry)
        if not label:
            continue
        keywords = entry.get("keywords") if isinstance(entry, dict) else None
        if not keywords:
            # Plain string / no anchor yet: the label refers to the model saved last run.
            keywords = previous_keywords.get(str(key), [])
        candidates.append({"label": label, "keywords": list(keywords)})

    for entry in curated.get(UNASSIGNED_KEY, []) or []:
        label = label_text(entry)
        keywords = entry.get("keywords") if isinstance(entry, dict) else None
        if label and keywords:
            candidates.append({"label": label, "keywords": list(keywords)})
    return candidates


def remap_topic_labels(
    curated: dict,
    new_keywords: dict,
    previous_keywords: dict[str, list[str]] | None = None,
    min_similarity: float = MIN_SIMILARITY,
) -> tuple[dict, dict]:
    """
    Assign curated labels to the topics of a freshly trained model.

    Args:
        curated:           Content of topic_labels.json (old IDs).
        new_keywords:      {new_topic_id: [keywords]} of the new model.
        previous_keywords: {old_topic_id: [keywords]} of the previous model,
                           used for entries without stored keywords.
        min_similarity:    Minimum keyword overlap for a match.

    Returns:
        (new_curated, report) – new_curated is keyed by the new topic IDs,
        report lists matches, unassigned labels and unlabelled topics.
    """
    previous_keywords = previous_keywords or {}
    new_kw = {
        str(tid): [str(k) for k in kws][:MAX_KEYWORDS]
        for tid, kws in new_keywords.items()
        if str(tid) != OUTLIER_ID
    }
    candidates = _candidates(curated, previous_keywords)

    # Greedy one-to-one matching: best pairs first.
    pairs = sorted(
        (
            (keyword_similarity(cand["keywords"], kws), ci, tid)
            for ci, cand in enumerate(candidates)
            for tid, kws in new_kw.items()
        ),
        key=lambda p: (-p[0], p[1], p[2]),
    )
    assigned: dict[str, tuple[int, float]] = {}
    used: set[int] = set()
    for sim, ci, tid in pairs:
        if sim < min_similarity:
            break
        if ci in used or tid in assigned:
            continue
        assigned[tid] = (ci, sim)
        used.add(ci)

    result: dict = {}
    if OUTLIER_ID in curated:
        result[OUTLIER_ID] = curated[OUTLIER_ID]
    for tid in sorted(new_kw, key=lambda t: int(t) if t.lstrip("-").isdigit() else t):
        match = assigned.get(tid)
        result[tid] = {
            "label": candidates[match[0]]["label"] if match else None,
            "keywords": new_kw[tid],
        }
    unassigned = [candidates[ci] for ci in range(len(candidates)) if ci not in used]
    if unassigned:
        result[UNASSIGNED_KEY] = unassigned

    report = {
        "matched": [
            (candidates[ci]["label"], tid, round(sim, 2)) for tid, (ci, sim) in sorted(assigned.items(), key=lambda x: int(x[0]))
        ],
        "unassigned": [c["label"] for c in unassigned],
        "unlabelled": [tid for tid in result if tid not in assigned and tid != OUTLIER_ID and not tid.startswith("_")],
    }
    return result, report


def save_curated_labels(data: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_topic_labels_file(
    labels_path: Path | str,
    new_keywords: dict,
    previous_keywords: dict[str, list[str]] | None = None,
    min_similarity: float = MIN_SIMILARITY,
) -> dict:
    """Remap labels_path in place to the new topic IDs and print a short report."""
    curated = load_curated_labels(labels_path)
    remapped, report = remap_topic_labels(curated, new_keywords, previous_keywords, min_similarity)
    save_curated_labels(remapped, labels_path)

    print(f"Topic-Labels neu zugeordnet: {len(report['matched'])} übernommen.")
    for label, tid, sim in report["matched"]:
        print(f"  {tid:>3} ← {label} (Überlappung {sim:.2f})")
    if report["unassigned"]:
        print(f"  Ohne passendes Topic (in '{UNASSIGNED_KEY}' geparkt): {', '.join(report['unassigned'])}")
    if report["unlabelled"]:
        print(f"  Neue Topics ohne Label (in {Path(labels_path).name} benennen): {', '.join(report['unlabelled'])}")
    return report
