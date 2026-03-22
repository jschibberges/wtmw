"""
Vergleich: Regelbasierte Klassifizierung vs. Regelbasiert + Embedding-Fallback.

Ausführen (conda env mediaanalysis):
    conda run -n mediaanalysis python compare_classification.py

Ausgabe: Tabelle mit Kategorie-Verteilung und Confidence-Statistiken für
         beide Ansätze sowie eine Liste von Beispielen, die der Fallback
         neu klassifiziert hat.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Daten laden
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

print("Lade JSON-Rohdaten…")
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
                "uid":      ep.get("uid", ""),
                "Talkshow": ep.get("show", ""),
                "name":     g.get("name", ""),
                "role":     g.get("role", ""),
                "description": g.get("description", ""),
                "party":    g.get("party", ""),
                "date":     ep.get("date", ""),
            })

df_raw = pd.DataFrame(rows)
print(f"  {len(df_raw)} Gastzeilen aus {len(list(DATA_DIR.glob('*_data.json')))} Dateien\n")

# ---------------------------------------------------------------------------
# 2. Cleaning & Konsolidierung
# ---------------------------------------------------------------------------
from nlp_utils import clean_guest_rows
from analyze_talkshows import consolidate_guests

print("Cleaning & Konsolidierung…")
df_cleaned, _ = clean_guest_rows(
    df_raw, name_col="name", role_col="role", party_col="party", show_col="Talkshow"
)
df_consolidated = consolidate_guests(df_cleaned)
if "name_clean" in df_consolidated.columns:
    df_consolidated.rename(columns={"name_clean": "name"}, inplace=True)
print(f"  {len(df_consolidated)} konsolidierte Gäste\n")

# ---------------------------------------------------------------------------
# 3. Regelbasierte Klassifizierung
# ---------------------------------------------------------------------------
from guest_classification import add_classification_columns

print("Regelbasierte Klassifizierung…")
df_rules = add_classification_columns(df_consolidated.copy())
df_rules["CategorySource"] = df_rules["CategoryPrimary"].apply(
    lambda x: "rules" if pd.notna(x) else None
)

# ---------------------------------------------------------------------------
# 4. Embedding-Fallback
# ---------------------------------------------------------------------------
from analyze_talkshows import _get_embedder, _embed_texts
from guest_classification_embedding import build_prototype_embeddings, apply_embedding_fallback

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

print("Lade Embedding-Modell und berechne Prototypen…")
embedder, _ = _get_embedder(EMBEDDING_MODEL)
embed_fn = lambda texts: _embed_texts(embedder, texts)
prototypes = build_prototype_embeddings(embed_fn)

print("Embedding-Fallback…")
df_hybrid = apply_embedding_fallback(
    df_rules.copy(),
    embed_fn,
    prototypes,
    role_col="role_clean",
    desc_col="description",
)

# ---------------------------------------------------------------------------
# 5. Vergleichstabelle
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Politics & Government",
    "Media & Communication",
    "Academia & Expertise",
    "Business & Economy",
    "Arts & Culture",
    "Civil Society & Advocacy",
    "Sports",
    "Religion & Spirituality",
    "Citizens & Everyday Voices",
    "Influencers & Digital Creators",
]

total = len(df_rules)

print("\n" + "=" * 72)
print("VERGLEICH: Regelbasiert vs. Regelbasiert + Embedding-Fallback")
print("=" * 72)
print(f"\n{'Kategorie':<35}  {'Regeln':>8}  {'Hybrid':>8}  {'Δ':>6}")
print("-" * 62)

for cat in CATEGORIES:
    n_rules  = (df_rules["CategoryPrimary"] == cat).sum()
    n_hybrid = (df_hybrid["CategoryPrimary"] == cat).sum()
    delta = n_hybrid - n_rules
    marker = f"  (+{delta})" if delta > 0 else ""
    print(f"  {cat:<33}  {n_rules:>8}  {n_hybrid:>8}{marker}")

# Nicht klassifiziert
n_none_rules  = df_rules["CategoryPrimary"].isna().sum()
n_none_hybrid = df_hybrid["CategoryPrimary"].isna().sum()
print("-" * 62)
print(f"  {'Nicht klassifiziert':<33}  {n_none_rules:>8}  {n_none_hybrid:>8}  ({n_none_hybrid - n_none_rules:+d})")
print(f"  {'Gesamt':<33}  {total:>8}  {total:>8}")

# Abdeckung
cov_rules  = (total - n_none_rules)  / total * 100
cov_hybrid = (total - n_none_hybrid) / total * 100
print(f"\n  Abdeckung  {cov_rules:>6.1f}%          {cov_hybrid:>6.1f}%  ({cov_hybrid - cov_rules:+.1f}pp)")

# Confidence
print("\n" + "-" * 62)
print("  Confidence-Statistik (nur klassifizierte Gäste)")
for label, df_ in [("Regeln", df_rules), ("Hybrid", df_hybrid)]:
    classified = df_[df_["CategoryPrimary"].notna()]["Confidence"]
    print(f"  {label:<10}  mean={classified.mean():.3f}  "
          f"median={classified.median():.3f}  "
          f"min={classified.min():.3f}  "
          f"max={classified.max():.3f}")

# ---------------------------------------------------------------------------
# 6. Beispiele: vom Fallback neu klassifiziert
# ---------------------------------------------------------------------------
newly_classified = df_hybrid[
    df_rules["CategoryPrimary"].isna() & df_hybrid["CategoryPrimary"].notna()
].copy()

print(f"\n{'=' * 72}")
print(f"BEISPIELE – {len(newly_classified)} Gäste neu durch Embedding klassifiziert")
print("=" * 72)

sample = newly_classified.sort_values("Confidence", ascending=False).head(30)
for _, row in sample.iterrows():
    name = row["name"]
    cat  = row["CategoryPrimary"]
    conf = row["Confidence"]
    role = row.get("role_clean", "")
    if isinstance(role, list):
        role = role[0] if role else ""
    desc = row.get("description", "")
    if isinstance(desc, list):
        desc = desc[0] if desc else ""
    desc_short = str(desc)[:80] if desc else "–"
    print(f"\n  {name}  →  {cat}  (conf={conf:.3f})")
    if role:
        print(f"    Rolle: {role}")
    print(f"    Desc:  {desc_short}")

print("\nFertig.")
