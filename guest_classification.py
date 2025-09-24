
from __future__ import annotations
import re, ast
from typing import Any, Dict, List, Tuple, Optional

RULES: Dict[str, Dict[str, List[str]]] = {
    "Politics & Government": {
        "strong": [
            r"\b(MdB|MdL|MdEP)\b",
            r"\b(Bundestag(s(abgeordnete|mitglied))?)\b",
            r"\b(Minister(präsident)?|Staatssekretär(in)?|Minister(in)?)\b",
            r"\b(Bürgermeister(in)?)\b",
            r"\b(Botschafter(in)?|Diplomat(in)?|Konsul(in)?)\b",
            r"\b(Fraktions(vorsitz|chef|vize))\b",
        ],
        "weak": [
            r"\b(Abgeordnete|Regierung|Parlament|Kabinett)\b",
            r"\b(Partei|Parteichef(in)?|Generalsekretär(in)?)\b",
            r"\b(SPD|CDU|CSU|FDP|Grüne|Gruene|AfD|Linke|Piraten)\b",
        ],
    },
    "Media & Communication": {
        "strong": [
            r"\b(Chefredakteur(in)?|Moderator(in)?|Anchor)\b",
            r"\b(Korrespondent(in)?|Auslands(korrespondent|reporter)(in)?)\b",
            r"\b(Redaktionsleiter(in)?|Ressortleiter(in)?)\b",
        ],
        "weak": [
            r"\b(Journalist(in)?|Reporter(in)?|Redakteur(in)?)\b",
            r"\b(ARD|ZDF|WELT|FAZ|Süddeutsche|SZ|ZEIT|Spiegel|taz|ntv|phoenix)\b",
            r"\b(Publizist(in)?|Kolumnist(in)?|Kommentator(in)?)\b",
        ],
    },
    "Academia & Expertise": {
        "strong": [
            r"\b(Professor(in)?|Prof\.?)\b",
            r"\b(Leiter(in)?|Direktor(in)?)\b.*\b(Institut|Forschungszentrum|Lehrstuhl)\b",
            r"\b(Verfassungsrechtler(in)?|Ökonom(in)?|Epidemiologe|Virologe|Soziologe|Politologe|Historiker)\b",
        ],
        "weak": [
            r"\b(Dr\.|Wissenschaftler(in)?|Forscher(in)?|Dozent(in)?)\b",
            r"\b(Think[- ]?Tank|Fellow|Senior Fellow)\b",
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
        ],
    },
    "Religion & Spirituality": {
        "strong": [
            r"\b(Imam|Pfarrer(in)?|Priester(in)?|Rabbi|Bischof|Nonne|Pater)\b",
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

def _to_text_list(value: Any):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed if v is not None]
            except Exception:
                pass
        return [s]
    return [str(value)]

def _score_category(text: str, party_norm):
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

def _confidence_from_scores(scores):
    best = max(scores.values()) if scores else 0.0
    second = 0.0
    if scores:
        vals = sorted(scores.values(), reverse=True)
        second = vals[1] if len(vals) > 1 else 0.0
    if best <= 0:
        return 0.0
    margin = best - second
    conf = min(1.0, (best / (best + second + 1e-6)) * 0.6 + min(0.4, margin / 6.0))
    return round(conf, 3)

def classify_text(texts, party_norm):
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

def classify_row(role_clean, description, party_norm=None):
    texts = _to_text_list(role_clean) + _to_text_list(description)
    categories, primary, confidence, scores = classify_text(texts, party_norm)
    return {
        "Categories": categories,
        "CategoryPrimary": primary,
        "Confidence": confidence,
        "CategoryScores": scores,
    }

def add_classification_columns(df, role_col="role_clean", desc_col="description", party_col="party_norm"):
    recs = []
    for _, r in df.iterrows():
        recs.append(classify_row(r.get(role_col), r.get(desc_col), r.get(party_col)))
    df["Categories"] = [rec["Categories"] for rec in recs]
    df["CategoryPrimary"] = [rec["CategoryPrimary"] for rec in recs]
    df["Confidence"] = [rec["Confidence"] for rec in recs]
    df["CategoryScores"] = [rec["CategoryScores"] for rec in recs]
    return df
