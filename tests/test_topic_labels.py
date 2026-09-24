from __future__ import annotations

import json
import random

import topic_labels as tl


UKRAINE = ["ukrainisch", "russisch", "putin", "kreml", "bundeswehr"]
CORONA = ["impfpflicht", "inzidenz", "impfstoff", "covid", "pandemie"]
RENTE = ["rente", "mütterrente", "rentenniveau", "altersarmut", "rentner"]


def test_keyword_similarity_splits_phrases_and_ignores_case():
    assert tl.keyword_similarity(["Ukraine Krieg"], ["ukraine", "krieg", "putin"]) == 1.0
    assert tl.keyword_similarity(UKRAINE, CORONA) == 0.0
    assert tl.keyword_similarity([], UKRAINE) == 0.0


def test_remap_follows_topics_when_ids_change():
    curated = {
        "-1": "Sonstige",
        "0": {"label": "Ukraine-Krieg", "keywords": UKRAINE},
        "1": {"label": "Corona", "keywords": CORONA},
    }
    new_keywords = {"-1": ["egal"], "0": CORONA[:4] + ["lockdown"], "1": UKRAINE}

    remapped, report = tl.remap_topic_labels(curated, new_keywords)

    assert remapped["-1"] == "Sonstige"
    assert remapped["0"]["label"] == "Corona"
    assert remapped["0"]["keywords"] == CORONA[:4] + ["lockdown"]
    assert remapped["1"]["label"] == "Ukraine-Krieg"
    assert report["unassigned"] == [] and report["unlabelled"] == []


def test_vanished_topic_is_parked_and_new_topic_left_unlabelled():
    curated = {
        "0": {"label": "Ukraine-Krieg", "keywords": UKRAINE},
        "1": {"label": "Rente", "keywords": RENTE},
    }
    remapped, report = tl.remap_topic_labels(curated, {"0": UKRAINE, "1": CORONA})

    assert remapped["0"]["label"] == "Ukraine-Krieg"
    assert remapped["1"] == {"label": None, "keywords": CORONA}
    assert remapped[tl.UNASSIGNED_KEY] == [{"label": "Rente", "keywords": RENTE}]
    assert report["unassigned"] == ["Rente"] and report["unlabelled"] == ["1"]

    # Kommt das Topic in einem späteren Lauf zurück, wird das geparkte Label wieder vergeben.
    again, report = tl.remap_topic_labels(remapped, {"0": UKRAINE, "1": CORONA, "2": RENTE})
    assert again["2"]["label"] == "Rente"
    assert tl.UNASSIGNED_KEY not in again


def test_plain_string_labels_use_previous_model_keywords():
    curated = {"0": "Ukraine-Krieg", "1": "Corona"}
    previous = {"0": UKRAINE, "1": CORONA}

    remapped, _ = tl.remap_topic_labels(curated, {"0": CORONA, "1": UKRAINE}, previous)

    assert remapped["0"]["label"] == "Corona"
    assert remapped["1"]["label"] == "Ukraine-Krieg"


def test_matching_is_one_to_one():
    curated = {"0": {"label": "Ukraine-Krieg", "keywords": UKRAINE}}
    remapped, report = tl.remap_topic_labels(curated, {"0": UKRAINE, "1": UKRAINE[:3] + ["x", "y"]})

    assert [remapped[t]["label"] for t in ("0", "1")] == ["Ukraine-Krieg", None]
    assert report["unlabelled"] == ["1"]


def test_labels_by_id_supports_both_formats_and_skips_meta_keys():
    data = {
        "-1": "Sonstige",
        "0": {"label": "Ukraine-Krieg", "keywords": UKRAINE},
        "1": {"label": None, "keywords": CORONA},
        tl.UNASSIGNED_KEY: [{"label": "Rente", "keywords": RENTE}],
    }
    assert tl.labels_by_id(data) == {"-1": "Sonstige", "0": "Ukraine-Krieg"}


def test_update_topic_labels_file_roundtrip(tmp_path):
    labels_path = tmp_path / "topic_labels.json"
    labels_path.write_text(json.dumps({"-1": "Sonstige", "0": "Ukraine-Krieg"}), encoding="utf-8")
    topics_json = tmp_path / "topics.json"
    topics_json.write_text(
        json.dumps({"topic_representations": {"-1": [["a", 0.1]], "0": [[k, 0.5] for k in UKRAINE]}}),
        encoding="utf-8",
    )
    previous = tl.load_model_keywords(topics_json)
    assert previous["0"] == UKRAINE

    tl.update_topic_labels_file(labels_path, {"-1": ["a"], "3": UKRAINE}, previous)

    saved = json.loads(labels_path.read_text(encoding="utf-8"))
    assert saved == {"-1": "Sonstige", "3": {"label": "Ukraine-Krieg", "keywords": UKRAINE}}


def test_real_labels_survive_simulated_retraining():
    """Echte Keywords, IDs gemischt, je 2 von ~10 Keywords ausgetauscht."""
    curated = tl.load_curated_labels(tl.Path(__file__).resolve().parents[1] / "data" / "topic_labels.json")
    topics = {k: v for k, v in curated.items() if not k.startswith("_") and isinstance(v, dict)}
    rng = random.Random(0)
    old_ids = sorted(topics, key=int)
    new_ids = old_ids[:]
    rng.shuffle(new_ids)
    new_keywords = {}
    for old_id, new_id in zip(old_ids, new_ids):
        kws = list(topics[old_id]["keywords"])
        for i in rng.sample(range(len(kws)), k=min(2, len(kws) - 1)):
            kws[i] = f"neu{old_id}_{i}"
        new_keywords[new_id] = kws

    remapped, report = tl.remap_topic_labels(curated, new_keywords)

    for old_id, new_id in zip(old_ids, new_ids):
        assert remapped[new_id]["label"] == topics[old_id]["label"]
    assert report["unassigned"] == []
