from __future__ import annotations

import re
import ast
from typing import Any, Dict, List


RULES: Dict[str, Dict[str, List[str]]] = {
    "Politics & Government": {
        "strong": [
            r"\b(MdB|MdL|MdEP)\b",
            r"\b(Bundestag(s(abgeordnete|mitglied))?)\b",
            r"\b(Minister(präsident)?|Staatssekretär(in)?|Minister(in)?)\b",
            r"\b(Bürgermeister(in)?)\b",
            r"\b(Botschafter(in)?|Diplomat(in)?|Konsul(in)?)\b",
            r"\b(Fraktions(vorsitz|chef|vize))\b",
            r"\bPolitiker(in)?\b",                          # neu: explizite Berufsbezeichnung
            r"\bBundeskanzler(in)?\b",                      # neu: Angela Merkel & Co.
            r"\bSenator(in)?\b",                            # neu: Stadtstaaten-Politik
            r"\bOberbürgermeister(in)?\b",                  # neu: war bisher nur Bürgermeister(in)
            r"\bLandrat\b|\bLandrätin\b",                   # neu: Kommunalebene
        ],
        "weak": [
            r"\b(Abgeordnete|Regierung|Parlament|Kabinett)\b",
            r"\b(Partei|Parteichef(in)?|Generalsekretär(in)?)\b",
            r"\b(SPD|CDU|CSU|FDP|Grüne|Gruene|AfD|Linke|Piraten)\b",
        ],
    },
    "Media & Communication": {
        "strong": [
            # Komposita-fix: kein führendes \b → matcht auch Fernsehmoderator, Talkshowmoderator etc.
            r"(Chefredakteur\w*|Moderator\w*|Anchor\b)",
            r"\b(Korrespondent(in)?|Auslands(korrespondent|reporter)\w*)\b",
            r"\b(Redaktionsleiter(in)?|Ressortleiter(in)?)\b",
        ],
        "weak": [
            # Komposita-fix: matcht Wirtschafts-, Wissenschafts-, Sportjournalist etc.
            r"(Journalist\w*|Reporter\w*|Redakteur\w*)",
            r"\b(ARD|ZDF|WELT|FAZ|Süddeutsche|SZ|ZEIT|Spiegel|taz|ntv|phoenix)\b",
            r"\b(Publizist(in)?|Kolumnist(in)?|Kommentator(in)?)\b",
        ],
    },
    "Academia & Expertise": {
        "strong": [
            r"\b(Professor(in)?|Prof\.?)\b",
            r"\b(Leiter(in)?|Direktor(in)?)\b.*\b(Institut|Forschungszentrum|Lehrstuhl)\b",
            r"\b(Verfassungsrechtler(in)?|Ökonom(in)?|Epidemiologe\w*|Virologe\w*|Soziologe\w*|Historiker\w*)\b",
            r"\bPolitikwissenschaftler(in)?\b",              # neu: häufig unkategorisiert
            r"\bPolitolog(e|in)\b",                          # neu: Synonym
        ],
        "weak": [
            r"\b(Dr\.|Wissenschaftler(in)?|Forscher(in)?|Dozent(in)?)\b",
            r"\b(Think[- ]?Tank|Fellow|Senior Fellow)\b",
            r"\w+forscher(in)?\b",                           # neu: Klima-, Migrations-, Extremismusforscher
            r"\bMilitärexpert\w+\b",                         # neu: Militärexperte/in
            r"\b(Jurist(in)?|Rechtsanwalt|Rechtsanwältin)\b", # neu
            r"\bKriminolog\w+\b",                            # neu: Kriminologe/in
            r"\b(Arzt|Ärztin)\b",                            # neu
        ],
    },
    "Civil Society & Advocacy": {
        "strong": [
            r"\b(Vorsitzend\w+|Präsident(in)?|Generalsekretär(in)?)\b.*\b(Verband|Verein|Gesellschaft|NGO)\b",
            r"\b(Zentralrat|Gewerkschaft|Bürgerrechts|Menschenrechts)\b",
        ],
        "weak": [
            r"\b(Aktivist(in)?|Sprecher(in)?)\b",
            r"\b(Caritas|Diakonie|Amnesty|Greenpeace)\b",
            r"\bUmweltaktivist(in)?\b|\bKlimaaktivist(in)?\b",  # neu: spez. Aktivismus-Formen
        ],
    },
    "Business & Economy": {
        "strong": [
            r"\b(CEO|Geschäftsführer(in)?|Vorständ(in|e)?|Vorstandsvorsitzend\w+|CFO|COO)\b",
            r"\b(Gründer(in)?|Mitgründer(in)?)\b",
        ],
        "weak": [
            r"\b(Unternehmer(in)?|Manager(in)?|IHK|Wirtschaftsverband)\b",
            r"\b(Banker(in)?|Investor(in)?|Ökonomie|Wirtschaft)\b",
            r"\b(Bäckermeister|Landwirt|Bauunternehmer)\b",
            r"\bUnternehmensberater(in)?\b",                 # neu
        ],
    },
    "Arts & Culture": {
        "strong": [
            r"\b(Autor(in)?|Schriftsteller(in)?|Regisseur(in)?|Filmemacher(in)?|Schauspieler(in)?)\b",
            r"\b(Musiker(in)?|Sänger(in)?|Komponist(in)?|Dirigent(in)?)\b",
            r"\b(Comedian|Kabarettist(in)?|Poetry[- ]?Slammer(in)?)\b",
        ],
        "weak": [
            r"\b(Künstler(in)?|Kultur|Intellektuelle?r?)\b",
            r"\b(Kritiker(in)?)\b",
        ],
    },
    "Sports": {
        "strong": [
            r"\b(Bundesliga|DFB|FIFA)\b",
            r"\b(Trainer(in)?|Schiedsrichter(in)?|Bundestrainer)\b",
        ],
        "weak": [
            r"\b(Sportler(in)?|Athlet(in)?|Fußballer(in)?|Basketballer(in)?|Handballer(in)?|Tennisspieler(in)?)\b",
            r"\b(Olympia|EM|WM)\b",
            r"\bEx-\w*(profi|spieler(in)?|nationalspieler\w*)\b",  # neu: Ex-Fußballprofi etc.
        ],
    },
    "Religion & Spirituality": {
        "strong": [
            r"\b(Imam|Pfarrer(in)?|Priester(in)?|Rabbi|Bischof|Nonne|Pater)\b",
            r"\bTheolog(e|in)\b",                            # neu
        ],
        "weak": [
            r"\b(Kirche|Moschee|Synagoge|Religionsgemeinschaft)\b",
        ],
    },
    "Citizens & Everyday Voices": {
        "strong": [
            r"\b(Betroffene(r)?|Zeitzeuge|Zeitzeugin)\b",
        ],
        "weak": [
            r"\b(Bürger(in)?|Anwohner(in)?|Patient(in)?|Rentner(in)?)\b",
            r"\b(Taxifahrer(in)?|Lehrer(in)?|Bäcker(in)?)\b",
        ],
    },
    "Influencers & Digital Creators": {
        "strong": [
            r"\b(YouTuber(in)?|Influencer(in)?|Streamer(in)?)\b",
            r"\b(Podcaster(in)?|Content[- ]?Creator)\b",
        ],
        "weak": [
            r"\b(Blogger(in)?|Instagram|TikTok)\b",
        ],
    },
}

PRIORITY = [
    "Politics & Government",
    "Media & Communication",
    "Academia & Expertise",
    "Civil Society & Advocacy",
    "Business & Economy",
    "Arts & Culture",
    "Sports",
    "Religion & Spirituality",
    "Citizens & Everyday Voices",
    "Influencers & Digital Creators",
]

STRONG_W = 3.0
WEAK_W = 1.0
PARTY_W = 4.0


def _to_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed if v is not None]
            except Exception:
                pass
        return [s]
    return [str(value)]


def _score_category(text: str, party_norm: Any) -> dict[str, float]:
    scores = {c: 0.0 for c in RULES.keys()}
    for cat, groups in RULES.items():
        for pat in groups.get("strong", []):
            if re.search(pat, text, flags=re.IGNORECASE):
                scores[cat] += STRONG_W
        for pat in groups.get("weak", []):
            if re.search(pat, text, flags=re.IGNORECASE):
                scores[cat] += WEAK_W
    if party_norm and str(party_norm).strip() and str(party_norm).strip().lower() not in ("nan", "none", "null"):
        scores["Politics & Government"] += PARTY_W
    return scores


_SCORE_REFERENCE = PARTY_W + STRONG_W  # 7.0 – "sehr sicher": Partei-Match + ein STRONG-Treffer


def _confidence_from_scores(scores: dict[str, float]) -> float:
    """Berechnet Konfidenz aus zwei Komponenten:

    - strength:   Wie viel absolute Evidenz liegt für die beste Kategorie vor?
                  Normiert auf PARTY_W + STRONG_W (= 7.0) als realistisches Maximum.
    - separation: Wie klar dominiert die beste Kategorie vor der zweitbesten?
                  1.0 = keine Konkurrenz, 0.0 = Gleichstand.

    Richtwerte:
      Ein WEAK-Treffer allein   → ~0.44
      Ein STRONG-Treffer allein → ~0.63
      Nur PARTY-Match           → ~0.72
      PARTY + STRONG            → 1.00
      Gleichstand zweier STRONG → ~0.28
    """
    if not scores:
        return 0.0
    vals = sorted(scores.values(), reverse=True)
    best = vals[0]
    if best <= 0:
        return 0.0
    second = vals[1] if len(vals) > 1 else 0.0

    strength   = min(1.0, best / _SCORE_REFERENCE)
    separation = (best - second) / best

    conf = 0.65 * strength + 0.35 * separation
    return round(min(1.0, conf), 3)


def classify_text(texts: list[str], party_norm: Any):
    text_joined = " | ".join([t for t in texts if t]).strip()
    if not text_joined and not (party_norm and str(party_norm).strip()):
        return [], None, 0.0, {}
    scores = _score_category(text_joined, party_norm)
    matched = [c for c, s in scores.items() if s > 0]
    if matched:
        max_score = max(scores[c] for c in matched)
        candidates = [c for c in matched if scores[c] == max_score]
        if len(candidates) > 1:
            primary = next((c for c in PRIORITY if c in candidates), candidates[0])
        else:
            primary = candidates[0]
    else:
        primary = None
    confidence = _confidence_from_scores(scores)
    return matched, primary, confidence, scores


def classify_row(role_clean: Any, description: Any, party_norm: Any = None) -> dict[str, Any]:
    texts = _to_text_list(role_clean) + _to_text_list(description)
    categories, primary, confidence, scores = classify_text(texts, party_norm)
    return {
        "Categories": categories,
        "CategoryPrimary": primary,
        "Confidence": confidence,
        "CategoryScores": scores,
    }


def add_classification_columns(
    df,
    role_col: str = "role_clean",
    desc_col: str = "description",
    party_col: str = "party_norm",
):
    # apply() statt iterrows – deutlich schneller für große DataFrames
    recs = df.apply(
        lambda row: classify_row(row.get(role_col), row.get(desc_col), row.get(party_col)),
        axis=1,
    )
    df["Categories"]     = [r["Categories"]     for r in recs]
    df["CategoryPrimary"]= [r["CategoryPrimary"] for r in recs]
    df["Confidence"]     = [r["Confidence"]      for r in recs]
    df["CategoryScores"] = [r["CategoryScores"]  for r in recs]
    return df
