"""
Embedding-basierter Fallback-Klassifizierer für Gäste ohne Regelkategorie.

Strategie:
  1. Pro Kategorie werden 5 repräsentative Prototyp-Texte definiert.
  2. Deren Embeddings werden gemittelt → ein Prototyp-Vektor pro Kategorie.
  3. Gäste ohne CategoryPrimary werden per Kosinus-Ähnlichkeit klassifiziert.
  4. Confidence wird linear aus der Ähnlichkeit skaliert, bleibt aber unter
     EMBEDDING_CONFIDENCE_MAX, damit regelbasierte und embeddingbasierte
     Klassifizierungen im Confidence-Wert unterscheidbar bleiben.

Integration in analyze_talkshows.py:
    from guest_classification_embedding import (
        build_prototype_embeddings,
        apply_embedding_fallback,
    )
    embed_fn = lambda texts: _embed_texts(embedder, texts)
    prototypes = build_prototype_embeddings(embed_fn)
    df_categorized = apply_embedding_fallback(df_categorized, embed_fn, prototypes)
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Prototyp-Texte (Deutsch) – je 5 pro Kategorie
# ---------------------------------------------------------------------------
CATEGORY_PROTOTYPES: dict[str, list[str]] = {
    "Politics & Government": [
        "Mitglied des Bundestages, Abgeordneter der SPD-Fraktion",
        "Bundesministerin für Inneres, Mitglied der Bundesregierung",
        "Vorsitzender der CDU/CSU-Fraktion im Deutschen Bundestag",
        "Staatssekretär im Auswärtigen Amt, erfahrener Diplomat",
        "Bürgermeisterin einer deutschen Großstadt, Kommunalpolitikerin",
    ],
    "Media & Communication": [
        "Chefredakteur der Süddeutschen Zeitung, Journalist",
        "ARD-Auslandskorrespondentin in Washington, Fernsehjournalistin",
        "Fernsehmoderator beim ZDF, Talkshow-Gastgeber",
        "Kolumnistin und Publizistin bei der ZEIT",
        "Wirtschaftsjournalist und Redakteur beim Spiegel",
    ],
    "Academia & Expertise": [
        "Professor für Politikwissenschaft an der Freien Universität Berlin",
        "Wissenschaftlerin am Deutschen Institut für Wirtschaftsforschung DIW",
        "Virologin und Leiterin eines Forschungsinstituts, Expertin für Infektionskrankheiten",
        "Historiker und Experte für Zeitgeschichte und den Zweiten Weltkrieg",
        "Klimaforscher an der Universität Hamburg, Sachverständiger für Energiepolitik",
    ],
    "Civil Society & Advocacy": [
        "Vorsitzende des Deutschen Gewerkschaftsbundes DGB",
        "Generalsekretär von Amnesty International Deutschland",
        "Sprecherin der Klimaschutzbewegung Fridays for Future",
        "Präsident des Zentralrates der Juden in Deutschland",
        "Leiterin einer gemeinnützigen Organisation für Flüchtlingshilfe",
    ],
    "Business & Economy": [
        "Vorstandsvorsitzender eines deutschen DAX-Konzerns, CEO",
        "Unternehmerin und Mitgründerin eines Tech-Startups im Silicon Valley",
        "Unternehmensberater und Wirtschaftsexperte für mittelständische Unternehmen",
        "Investorin und Bankerin, Mitglied im Aufsichtsrat mehrerer Unternehmen",
        "Geschäftsführerin eines mittelständischen Familienunternehmens",
    ],
    "Arts & Culture": [
        "Schriftstellerin und Autorin mehrerer preisgekrönter Romane",
        "Regisseur und Filmemacher, Gewinner des Deutschen Filmpreises",
        "Schauspielerin, bekannt aus Film und Theater, Mitglied der Deutschen Akademie",
        "Musiker und Sänger, Träger des Bundesverdienstkreuzes",
        "Kabarettist und Comedian, Satiriker im öffentlich-rechtlichen Fernsehen",
    ],
    "Sports": [
        "Fußballtrainer in der Bundesliga, ehemaliger Nationalspieler",
        "Olympiasiegerin im Schwimmen, Weltrekordlerin und Leistungssportlerin",
        "Ex-Profifußballer und Sportkommentator beim ZDF",
        "Bundestrainer der deutschen Handball-Nationalmannschaft",
        "Tennisprofi und Wimbledon-Siegerin, Botschafterin für Jugendsport",
    ],
    "Religion & Spirituality": [
        "Bischof der evangelischen Kirche in Deutschland EKD",
        "Imam einer Berliner Moschee, Islamwissenschaftler und Integrationsbeauftragter",
        "Rabbiner und Vertreter der jüdischen Gemeinde in Deutschland",
        "Theologe und Religionsphilosoph an einer deutschen Universität",
        "Kardinal der römisch-katholischen Kirche, Vorsitzender der Deutschen Bischofskonferenz",
    ],
    "Citizens & Everyday Voices": [
        # Prototypen betonen ausdrücklich: Privatperson, persönliche Betroffenheit, kein Experten-Kontext
        "Rentner, als Privatperson eingeladen – berichtet über steigende Energiekosten aus eigener Erfahrung",
        "Hartz-IV-Empfängerin, alleinerziehende Mutter, schildert ihren Alltag als Betroffene",
        "Holocaust-Überlebender, Zeitzeuge – erzählt von persönlichen Erlebnissen im Zweiten Weltkrieg",
        "Obdachloser, berichtet als Betroffener über sein Leben auf der Straße",
        "Abgeschobene Schülerin, schildert ihre persönliche Geschichte als Betroffene",
    ],
    "Influencers & Digital Creators": [
        "YouTuberin mit über einer Million Abonnenten, Content Creatorin über Politik",
        "TikTok-Influencer und Instagram-Blogger über Lifestyle und Gesellschaft",
        "Podcaster über Politik und Gesellschaft, bekannt aus Social Media",
        "Digitale Aktivistin und Social-Media-Expertin für Klimaschutz",
        "Streamer und Content Creator, Mitglied der Creators-Union",
    ],
}

# Maximale Confidence für Embedding-Klassifizierungen.
# Bewusst unter 1.0, damit regelbasierte (→ bis 1.0) und embeddingbasierte
# Klassifizierungen im Confidence-Wert unterscheidbar bleiben.
EMBEDDING_CONFIDENCE_MAX = 0.75

# Kategorien, die eine höhere Mindest-Ähnlichkeit benötigen, um false positives
# zu vermeiden. Citizens ist erfahrungsgemäß ein "catch-all" bei schwachem Signal.
CATEGORY_MIN_SIMILARITY_OVERRIDES: dict[str, float] = {
    "Citizens & Everyday Voices": 0.62,
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _sim_to_confidence(sim: float, min_sim: float) -> float:
    """Lineare Skalierung: [min_sim, 1.0] → [0.0, EMBEDDING_CONFIDENCE_MAX]."""
    if sim < min_sim:
        return 0.0
    conf = (sim - min_sim) / (1.0 - min_sim) * EMBEDDING_CONFIDENCE_MAX
    return round(min(EMBEDDING_CONFIDENCE_MAX, conf), 3)


def _row_to_text(row: pd.Series, role_col: str, desc_col: str) -> str:
    parts: list[str] = []
    for col in (role_col, desc_col):
        val = row.get(col)
        if isinstance(val, list):
            parts.extend(str(v) for v in val if v)
        elif val:
            parts.append(str(val))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

EmbedFn = Callable[[list[str]], np.ndarray]


def build_prototype_embeddings(embed_fn: EmbedFn) -> dict[str, np.ndarray]:
    """
    Berechnet den gemittelten Embedding-Vektor pro Kategorie.

    Args:
        embed_fn: Callable, das eine Liste von Strings entgegennimmt und
                  ein np.ndarray der Form (n, dim) zurückgibt.
    Returns:
        Dict {Kategoriename → Prototyp-Vektor (1D ndarray)}.
    """
    all_texts: list[str] = []
    cat_slices: dict[str, tuple[int, int]] = {}
    start = 0
    for cat, texts in CATEGORY_PROTOTYPES.items():
        all_texts.extend(texts)
        cat_slices[cat] = (start, start + len(texts))
        start += len(texts)

    print(f"[embedding-fallback] Berechne Prototyp-Embeddings "
          f"({len(all_texts)} Texte, {len(CATEGORY_PROTOTYPES)} Kategorien)…")
    embeddings = embed_fn(all_texts)  # (n_total, dim)

    return {
        cat: embeddings[s:e].mean(axis=0)
        for cat, (s, e) in cat_slices.items()
    }


def apply_embedding_fallback(
    df: pd.DataFrame,
    embed_fn: EmbedFn,
    prototype_embeddings: Optional[dict[str, np.ndarray]] = None,
    role_col: str = "role_clean",
    desc_col: str = "description",
    min_similarity: float = 0.35,
) -> pd.DataFrame:
    """
    Klassifiziert Gäste ohne CategoryPrimary per Kosinus-Ähnlichkeit zu
    Kategorie-Prototypen.

    Args:
        df:                   Gäste-DataFrame, muss 'CategoryPrimary' enthalten.
        embed_fn:             Embedding-Funktion (Liste[str] → np.ndarray).
        prototype_embeddings: Vorberechnete Prototyp-Vektoren (optional, wird
                              sonst via build_prototype_embeddings erstellt).
        role_col:             Spaltenname für Rollentext.
        desc_col:             Spaltenname für Beschreibungstext.
        min_similarity:       Mindest-Kosinus-Ähnlichkeit für eine Zuweisung.

    Returns:
        Kopie des DataFrames mit aktualisierten Spalten CategoryPrimary,
        Confidence und CategorySource ('rules' | 'embedding' | None).
    """
    df = df.copy()

    # CategorySource markiert den Ursprung jeder Klassifizierung
    if "CategorySource" not in df.columns:
        df["CategorySource"] = df["CategoryPrimary"].apply(
            lambda x: "rules" if pd.notna(x) else None
        )

    mask = df["CategoryPrimary"].isna()
    n_unclassified = mask.sum()
    if n_unclassified == 0:
        print("[embedding-fallback] Alle Gäste bereits klassifiziert – kein Fallback nötig.")
        return df

    if prototype_embeddings is None:
        prototype_embeddings = build_prototype_embeddings(embed_fn)

    # Texte für alle unkategorisierten Gäste sammeln
    unclassified = df[mask]
    texts = [_row_to_text(row, role_col, desc_col) for _, row in unclassified.iterrows()]

    # Leere Texte ausfiltern, aber Index-Zuordnung behalten
    nonempty_idx = [i for i, t in enumerate(texts) if t.strip()]
    nonempty_texts = [texts[i] for i in nonempty_idx]

    categories: list[Optional[str]] = [None] * n_unclassified
    confidences: list[float] = [0.0] * n_unclassified

    if nonempty_texts:
        print(f"[embedding-fallback] Klassifiziere {len(nonempty_texts)} Gäste "
              f"ohne Regelkategorie (min_sim={min_similarity})…")
        embeddings = embed_fn(nonempty_texts)  # (n, dim) – ein Batch-Call

        # party_norm-Spalte für Override-Check vorhalten
        party_col_data = unclassified.get("party_norm", pd.Series([None] * n_unclassified,
                                                                    index=unclassified.index))

        for batch_pos, orig_pos in enumerate(nonempty_idx):
            emb = embeddings[batch_pos]
            sims = {cat: _cosine_sim(emb, proto)
                    for cat, proto in prototype_embeddings.items()}
            sorted_cats = sorted(sims, key=sims.get, reverse=True)
            best_cat = sorted_cats[0]
            best_sim = sims[best_cat]

            # Per-Kategorie-Schwellwert (z. B. Citizens braucht höheren Wert)
            effective_min = CATEGORY_MIN_SIMILARITY_OVERRIDES.get(best_cat, min_similarity)

            if best_sim < effective_min:
                # Beste Kategorie unter Schwelle: zweite Option prüfen
                for alt_cat in sorted_cats[1:]:
                    alt_sim = sims[alt_cat]
                    alt_min = CATEGORY_MIN_SIMILARITY_OVERRIDES.get(alt_cat, min_similarity)
                    if alt_sim >= alt_min:
                        best_cat, best_sim = alt_cat, alt_sim
                        break
                else:
                    continue  # alle Optionen unter Schwelle → kein Eintrag

            # Override: party_norm gesetzt → kann nicht Citizens sein
            row_index = unclassified.index[orig_pos]
            party_norm = party_col_data.get(row_index)
            if (best_cat == "Citizens & Everyday Voices"
                    and party_norm and str(party_norm).strip()
                    and str(party_norm).strip().lower() not in ("nan", "none", "null")):
                # Zweite beste Option ohne Citizens-Beschränkung nehmen
                for alt_cat in sorted_cats[1:]:
                    if alt_cat != "Citizens & Everyday Voices":
                        alt_sim = sims[alt_cat]
                        alt_min = CATEGORY_MIN_SIMILARITY_OVERRIDES.get(alt_cat, min_similarity)
                        if alt_sim >= alt_min:
                            best_cat, best_sim = alt_cat, alt_sim
                            break
                else:
                    best_cat = "Politics & Government"  # sicherer Fallback bei Partei-Match

            categories[orig_pos] = best_cat
            confidences[orig_pos] = _sim_to_confidence(best_sim, min_similarity)

    # Ergebnisse zurückschreiben
    unclassified_index = unclassified.index
    df.loc[unclassified_index, "CategoryPrimary"] = categories
    df.loc[unclassified_index, "Confidence"] = confidences
    df.loc[unclassified_index, "CategorySource"] = [
        "embedding" if c else None for c in categories
    ]

    found = sum(1 for c in categories if c is not None)
    print(f"[embedding-fallback] {found}/{n_unclassified} zusätzlich klassifiziert "
          f"({found / n_unclassified * 100:.1f}% der vorher unkategorisierten Gäste).")
    return df
