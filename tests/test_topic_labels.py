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
    # Nichts Aktuelles geht verloren; geparkt bleiben nur die Labels vergangener Topics.
    parked = {tl.label_text(e) for e in curated.get(tl.UNASSIGNED_KEY, [])}
    assert set(report["unassigned"]) == parked


def _assign(**topics):
    """_assign(t0=["a","b"], t1=[...]) → {uid: topic_id}"""
    return {uid: key[1:] for key, uids in topics.items() for uid in uids}


def test_doc_overlap_matches_even_when_keywords_differ():
    curated = {"0": {"label": "Ukraine-Krieg", "keywords": ["ukrainisch", "russisch", "putin", "kreml"]}}
    previous = _assign(t0=["e1", "e2", "e3", "e4"])
    new = _assign(t5=["e1", "e2", "e3", "e4", "e5"])

    remapped, report = tl.remap_topic_labels(
        curated,
        {"5": ["ukraine", "russland", "kiew", "selenskyj"]},
        previous_assignments=previous,
        new_assignments=new,
    )

    assert remapped["5"]["label"] == "Ukraine-Krieg"
    assert report["matched"] == [("Ukraine-Krieg", "5", 0.8, "Folgen")]


def test_doc_overlap_wins_over_misleading_keywords():
    curated = {
        "0": {"label": "Ukraine-Krieg", "keywords": UKRAINE},
        "1": {"label": "Corona", "keywords": CORONA},
    }
    previous = _assign(t0=["u1", "u2", "u3"], t1=["c1", "c2", "c3"])
    # Neues Topic 0 enthält die Corona-Folgen, beschreibt sich aber mit Ukraine-Wörtern
    new = _assign(t0=["c1", "c2", "c3"], t1=["u1", "u2", "u3"])

    remapped, _ = tl.remap_topic_labels(
        curated, {"0": UKRAINE, "1": CORONA}, previous_assignments=previous, new_assignments=new
    )

    assert remapped["0"]["label"] == "Corona"
    assert remapped["1"]["label"] == "Ukraine-Krieg"


def test_split_topic_label_goes_to_larger_part():
    curated = {"0": {"label": "Migration", "keywords": ["migration"]}}
    previous = _assign(t0=[f"e{i}" for i in range(10)])
    new = _assign(t0=[f"e{i}" for i in range(3)], t1=[f"e{i}" for i in range(3, 10)])

    remapped, report = tl.remap_topic_labels(
        curated, {"0": ["asyl"], "1": ["grenze"]}, previous_assignments=previous, new_assignments=new
    )

    assert remapped["1"]["label"] == "Migration"
    assert remapped["0"]["label"] is None
    assert report["unlabelled"] == ["0"]


def test_low_doc_overlap_without_keyword_match_is_parked():
    curated = {"0": {"label": "Rente", "keywords": RENTE}}
    previous = _assign(t0=["e1", "e2", "e3", "e4"])
    new = _assign(t0=["e1", "x1", "x2", "x3", "x4"])  # Jaccard 1/8

    remapped, report = tl.remap_topic_labels(
        curated, {"0": CORONA}, previous_assignments=previous, new_assignments=new
    )

    assert remapped["0"]["label"] is None
    assert report["unassigned"] == ["Rente"]


def test_unassigned_labels_fall_back_to_keywords():
    curated = {
        "0": {"label": "Ukraine-Krieg", "keywords": UKRAINE},
        tl.UNASSIGNED_KEY: [{"label": "Rente", "keywords": RENTE}],
    }
    previous = _assign(t0=["u1", "u2"])
    new = _assign(t3=["u1", "u2"], t4=["r1", "r2"])

    remapped, report = tl.remap_topic_labels(
        curated, {"3": ["krieg"], "4": RENTE}, previous_assignments=previous, new_assignments=new
    )

    assert remapped["3"]["label"] == "Ukraine-Krieg"
    assert remapped["4"]["label"] == "Rente"
    assert [m[3] for m in report["matched"]] == ["Folgen", "Keywords"]
    assert tl.UNASSIGNED_KEY not in remapped


def test_topic_ids_from_excel_floats_are_normalized():
    curated = {"2": {"label": "Corona", "keywords": ["x"]}}
    remapped, _ = tl.remap_topic_labels(
        curated,
        {"7": ["y"]},
        previous_assignments={"a": 2.0, "b": 2.0, "c": -1.0},
        new_assignments={"a": 7, "b": 7, "c": -1},
    )
    assert remapped["7"]["label"] == "Corona"


def test_real_labels_survive_simulated_retraining_with_new_keywords():
    """Echte Zuordnungen: IDs gemischt, 5 % der Folgen umverteilt, 20 neue Folgen,
    alle Keywords ausgetauscht – die Labels müssen allein über die Folgen folgen."""
    import pandas as pd

    root = tl.Path(__file__).resolve().parents[1] / "data"
    curated = tl.load_curated_labels(root / "topic_labels.json")
    prev = pd.read_excel(root / "all_data_with_topics.xlsx", usecols=["uid", "topic"]).dropna()
    previous = dict(zip(prev["uid"], prev["topic"].astype(int)))

    rng = random.Random(1)
    topic_ids = sorted({t for t in previous.values() if t != -1})
    shuffled = topic_ids[:]
    rng.shuffle(shuffled)
    id_map = dict(zip(topic_ids, shuffled)) | {-1: -1}
    new = {uid: id_map[t] for uid, t in previous.items()}
    for uid in rng.sample(sorted(new), k=len(new) // 20):
        new[uid] = rng.choice(topic_ids)
    for i in range(20):
        new[f"neu{i}"] = rng.choice(topic_ids)

    remapped, report = tl.remap_topic_labels(
        curated,
        {str(t): [f"anders{t}_{i}" for i in range(10)] for t in topic_ids},
        previous_assignments=previous,
        new_assignments=new,
    )

    for old_id, new_id in id_map.items():
        if old_id != -1:
            assert remapped[str(new_id)]["label"] == curated[str(old_id)]["label"]
    # Nichts Aktuelles geht verloren; geparkt bleiben nur die Labels vergangener Topics.
    parked = {tl.label_text(e) for e in curated.get(tl.UNASSIGNED_KEY, [])}
    assert set(report["unassigned"]) == parked
    assert {m[3] for m in report["matched"]} == {"Folgen"}


def test_format_flag_follows_label_through_remap_and_prefixes_display():
    curated = {
        "0": {"label": "Ukraine-Krieg", "keywords": UKRAINE},
        "1": {"label": "Politische Runde", "keywords": RENTE, "kind": tl.FORMAT_KIND},
    }
    remapped, _ = tl.remap_topic_labels(curated, {"0": RENTE, "1": UKRAINE})

    assert remapped["0"]["kind"] == tl.FORMAT_KIND
    assert "kind" not in remapped["1"]
    assert tl.labels_by_id(remapped) == {
        "0": "Format-Cluster: Politische Runde",
        "1": "Ukraine-Krieg",
    }


def test_parked_format_label_keeps_its_flag():
    curated = {"0": {"label": "Politische Runde", "keywords": RENTE, "kind": tl.FORMAT_KIND}}
    remapped, _ = tl.remap_topic_labels(curated, {"0": CORONA})

    assert remapped[tl.UNASSIGNED_KEY][0]["kind"] == tl.FORMAT_KIND
