"""
Untersucht die 'Citizens & Everyday Voices'-Kategorie nach dem Embedding-Fallback:
  - Wie viele kommen von Regeln vs. Embedding?
  - Confidence-Verteilung
  - Top-Rollen und Beispiele
  - Verdächtige Fehlzuordnungen (hohe Confidence, aber klingt nicht wie Bürger)

Ausführen:
    conda run -n mediaanalysis python inspect_citizens.py
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Daten laden & klassifizieren (gleiche Pipeline wie compare_classification.py)
# ---------------------------------------------------------------------------
print("Lade und verarbeite Daten…")
rows = []
for f in sorted(DATA_DIR.glob("*_data.json")):
    entries = json.loads(f.read_text(encoding="utf-8"))
    for ep in (entries or []):
        if not ep:
            continue
        for g in (ep.get("guests") or []):
            if not g:
                continue
            rows.append({
                "uid":         ep.get("uid", ""),
                "Talkshow":    ep.get("show", ""),
                "name":        g.get("name", ""),
                "role":        g.get("role", ""),
                "description": g.get("description", ""),
                "party":       g.get("party", ""),
                "date":        ep.get("date", ""),
            })

df_raw = pd.DataFrame(rows)

from nlp_utils import clean_guest_rows
from analyze_talkshows import consolidate_guests, _get_embedder, _embed_texts
from guest_classification import add_classification_columns
from guest_classification_embedding import build_prototype_embeddings, apply_embedding_fallback

df_cleaned, _ = clean_guest_rows(
    df_raw, name_col="name", role_col="role", party_col="party", show_col="Talkshow"
)
df_cons = consolidate_guests(df_cleaned)
if "name_clean" in df_cons.columns:
    df_cons.rename(columns={"name_clean": "name"}, inplace=True)

df_rules = add_classification_columns(df_cons.copy())
df_rules["CategorySource"] = df_rules["CategoryPrimary"].apply(
    lambda x: "rules" if pd.notna(x) else None
)

print("Lade Embeddings…")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
embedder, _ = _get_embedder(EMBEDDING_MODEL)
embed_fn = lambda texts: _embed_texts(embedder, texts)
prototypes = build_prototype_embeddings(embed_fn)

df_hybrid = apply_embedding_fallback(
    df_rules.copy(), embed_fn, prototypes,
    role_col="role_clean", desc_col="description",
)

# ---------------------------------------------------------------------------
# Fokus: Citizens & Everyday Voices
# ---------------------------------------------------------------------------
CAT = "Citizens & Everyday Voices"
citizens = df_hybrid[df_hybrid["CategoryPrimary"] == CAT].copy()
rules_only = df_rules[df_rules["CategoryPrimary"] == CAT]

print(f"\n{'=' * 70}")
print(f"ANALYSE: {CAT}")
print(f"{'=' * 70}")
print(f"  Regelbasiert:         {len(rules_only):>4}")
print(f"  Nach Embedding:       {len(citizens):>4}  (+{len(citizens) - len(rules_only)})")
print(f"  Davon via Embedding:  {(citizens['CategorySource'] == 'embedding').sum():>4}")

# Confidence-Verteilung
print(f"\n  Confidence-Verteilung (Embedding-Anteil):")
emb_citizens = citizens[citizens["CategorySource"] == "embedding"]["Confidence"]
bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for lo, hi in zip(bins, bins[1:]):
    n = ((emb_citizens >= lo) & (emb_citizens < hi)).sum()
    bar = "█" * (n // 5)
    print(f"  [{lo:.1f}–{hi:.1f})  {n:>4}  {bar}")
n_top = (emb_citizens >= 0.7).sum()
print(f"  [0.7–0.8]  {n_top:>4}  {'█' * (n_top // 5)}")

# Top-Rollen
print(f"\n  Häufigste role_clean-Werte (Embedding-Anteil, Top-20):")
from collections import Counter
all_roles = []
for v in emb_citizens.index.map(lambda i: citizens.loc[i, "role_clean"] if i in citizens.index else None):
    if isinstance(v, list): all_roles.extend(str(r) for r in v if r)
    elif v: all_roles.append(str(v))
# recollect properly
all_roles = []
for _, row in citizens[citizens["CategorySource"] == "embedding"].iterrows():
    v = row.get("role_clean")
    if isinstance(v, list): all_roles.extend(str(r) for r in v if r)
    elif v: all_roles.append(str(v))
for val, cnt in Counter(all_roles).most_common(20):
    print(f"    {cnt:4d}x  {val!r}")

# ---------------------------------------------------------------------------
# Verdächtige Einträge: Conf > 0.55 aber klingt nicht wie Normalbürger
# Zeige die Top-40 nach Confidence zum manuellen Durchsehen
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print(f"TOP-40 Citizens nach Confidence – zur manuellen Prüfung")
print(f"{'=' * 70}")

top = citizens.sort_values("Confidence", ascending=False).head(40)
for _, row in top.iterrows():
    name   = row["name"]
    conf   = row["Confidence"]
    source = row.get("CategorySource", "?")
    role   = row.get("role_clean", "")
    if isinstance(role, list): role = role[0] if role else ""
    desc   = row.get("description", "")
    if isinstance(desc, list): desc = desc[0] if desc else ""
    desc_s = str(desc)[:100] if desc else "–"
    print(f"\n  [{source:>9}  conf={conf:.3f}]  {name}")
    if role: print(f"    Rolle: {role}")
    print(f"    Desc:  {desc_s}")

# ---------------------------------------------------------------------------
# Gegencheck: Was ordnet das Modell als 2.-beste Kategorie ein?
# (Gibt Hinweis, ob Grenzfälle tatsächlich ambivalent sind)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print(f"ZWEITBESTE KATEGORIE für Embedding-Citizens (Top-Paare)")
print(f"{'=' * 70}")

# Wir müssen die Scores neu berechnen
import numpy as np

def _cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

def _row_to_text(row):
    parts = []
    for col in ("role_clean", "description"):
        v = row.get(col)
        if isinstance(v, list): parts.extend(str(x) for x in v if x)
        elif v: parts.append(str(v))
    return " | ".join(parts)

emb_cit = citizens[citizens["CategorySource"] == "embedding"].copy()
texts = [_row_to_text(row) for _, row in emb_cit.iterrows()]
nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]

if nonempty:
    idxs, txts = zip(*nonempty)
    embeddings = embed_fn(list(txts))
    pair_counts = Counter()
    for batch_i, orig_i in enumerate(idxs):
        emb = embeddings[batch_i]
        sims = {cat: _cosine_sim(emb, proto) for cat, proto in prototypes.items()}
        sorted_cats = sorted(sims, key=sims.get, reverse=True)
        best, second = sorted_cats[0], sorted_cats[1]
        pair_counts[(best, second)] += 1

    print(f"  {'Beste → Zweitbeste':<55}  {'n':>5}")
    print(f"  {'-'*62}")
    for (best, second), cnt in pair_counts.most_common(10):
        print(f"  {best:<28} → {second:<24}  {cnt:>5}")

print("\nFertig.")
