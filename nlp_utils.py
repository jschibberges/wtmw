# nlp_utils_lite.py
from __future__ import annotations
import re, unicodedata
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np
import pandas as pd
import networkx as nx
from functools import lru_cache

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --- optional deps: try to use if present, else fall back ---
try:
    # Better similarity
    from rapidfuzz import fuzz
    _rfuzz = True
except Exception:
    import difflib
    _rfuzz = False

try:
    # For downloading stopwords
    import requests
except ImportError:
    requests = None

try:
    # German phonetics for last names (nice-to-have)
    from cologne_phonetics import cologne_phonetics
    _cologne = True
except Exception:
    _cologne = False

try:
    # Lemmatization (labels cleanup)
    import spacy
    try:
        _NLP_DE = spacy.load("de_core_news_md", disable=["parser","ner"])
    except Exception:
        _NLP_DE = spacy.load("de_core_news_sm", disable=["parser","ner"])
except Exception:
    _NLP_DE = None


# ==============================
# 0) CONFIG (tiny, editable)
# ==============================
@dataclass
class LiteConfig:
    # Consolidation thresholds
    auto_merge: float = 0.85
    review_low: float = 0.70  # not used in auto; can export review pairs if you want

    # Description cleaning
    keep_pos: tuple = ("NOUN", "PROPN", "ADJ", "VERB")
    min_tokens_cleaned: int = 6
    remove_urls: bool = True
    remove_emojis: bool = True

    # Vectorizer guardrails (used outside in your modeling)
    min_df: int = 2
    max_df: float = 0.6
    ngram_range: tuple = (1, 3)


# Simple lists you can extend
PARTICLES = {"von","zu","vom","zur","van","de","del","der"}
PARTY_ALIASES = {
    "cdu":"CDU","csu":"CSU","spd":"SPD","afd":"AfD","fdp":"FDP",
    "die linke":"LINKE","linke":"LINKE","dielinke":"LINKE",
    "b'90":"GRUENE","b’90":"GRUENE","b90":"GRUENE",
    "bündnis 90":"GRUENE","bündnis 90/die grünen":"GRUENE",
    "grüne":"GRUENE","grünen":"GRUENE","gruene":"GRUENE","gruenen":"GRUENE",
    "bsw":"BSW","parteilos":"parteilos"
    }
MEDIA_ORG = {
    "zdf","ard","wdr","ndr","mdr","rbb","swr","br","orf","rtl","sat1","sat.1","servustv",
    "pro7","pro sieben","phoenix","welt","faz","sz","bild","taz","zeit","spiegel","focus","stern","dpa"
}
ROLE_WORDS = {
    # Generic roles
    "abgeordneter","abgeordnete","minister","ministerin","staatssekretär","staatssekretärin","altersforscher",
    "vorsitzender","vorsitzende","präsident","präsidentin","sprecher","sprecherin",
    "journalist","journalistin","experte","expertin","autor","autorin","unternehmer","unternehmerin",
    "wissenschaftler","wissenschaftlerin","politologe","politologin","kommentator","kommentatorin",
    "moderator","moderatorin","bürgermeister","bürgermeisterin","bezirksbürgermeister","mitglied","landtag","bundestag","eu-parlament",
    "korrespondent","korrespondentin","altersforscherin",
    # Specific roles from feedback
    "bundesaußenminister", "bundesfinanzminister", "bundesjustizministerin", "bundesvorsitzender", "Fraktionsvorsitzende",
    # Affixes and titles
    "mdep","mdb","mdl","a.d.","i.r.",
    # Junk words often appearing with names
    "büro", "verein", "beamtenbund", "arbeitgeberverbands", "gesamtmetall", "ausschusses", "untersuchungsausschusses", "bundestages"
}

GERMAN_EXTRA_STOP = {
    "heute","gestern","morgen","sendung","folge","talk","talkshow",
    "thema","gäste","gast","moderator","moderation"
}

# Words that are not part of a name but might appear in the name field
JUNK_WORDS_IN_NAMES = {
    "gegen", "begleitete", "seine", "frau", "deren", "selbsttötung", "derzeit", "hoch", "verschuldet",
    "auswärtigen", "deutschen", "terrorgruppe", "nsu", "marzahn", "aus", "ehem.", "ehemalig",
    "ehemaliger", "ehemalige", "Familie", "Familien", "Fachbereich", "Gast","Gäste"
}

LOW_INFORMATION_LEMMAS = {
    "deutsch", "deutschland", "bundesrepublik", "jahr", "jaehrig", "jährig",
    "jähr", "jährlich", "uhr",
}


TITLES_RE = re.compile(r"\b(prof\.?\s*dr\.?|prof\.?|dr\.|dipl\.-\w+|md[bpl]|mdep|ra|ll\.?m\.?|ma|mba|ba|bsc|msc|phd|a\.d\.|i\.r\.)\b", re.IGNORECASE)
LEADING_NUM = re.compile(r"^\s*(?:\d+[\)\.:\-]|[-–—•*])\s*")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile(r"([\U00010000-\U0010FFFF])", flags=re.UNICODE)
ALPHA_DE_CHARS = r"A-Za-zÄÖÜäöüß"
TOKEN_RE = re.compile(rf"[{ALPHA_DE_CHARS}][{ALPHA_DE_CHARS}\-']*")
CLOSE_TAIL_RE = re.compile(r"\s*[\)\]\}>]+$")

BOILERPLATE_PATTERNS = [
    re.compile(r"\bheute\s+zu\s+gast:.*$", re.IGNORECASE),
    re.compile(r"\b(?:zu\s+sehen|ab|um)\s+\d{1,2}[:.]\d{2}\s*uhr\b", re.IGNORECASE),
    re.compile(r"\bjetzt\s+live\b", re.IGNORECASE),
    re.compile(r"\bmehr\s+infos?:.*$", re.IGNORECASE),
]

# ==============================
# 1) SMALL HELPERS
# ==============================
def _nfkc(s): return unicodedata.normalize("NFKC", s)
def _normalize_quotes(s): return (s.replace("’","'").replace("‘","'").replace("´","'")
                                   .replace("“",'"').replace("”",'"'))
def _fold(s):  # German fold
    return (s.replace("ä","ae").replace("ö","oe").replace("ü","ue")
              .replace("Ä","Ae").replace("Ö","Oe").replace("Ü","Ue")
              .replace("ß","ss"))

def _is_person_like(s: str) -> bool:
    # ≥2 alphabetic tokens, no digits, not dominated by role/org words
    if not s or re.search(r"\d", s):
        return False
    toks = TOKEN_RE.findall(s)
    if len(toks) < 2:
        return False
    low = [t.lower() for t in toks]
    if any(t in ROLE_WORDS for t in low):
        return False
    return True

def _is_party_or_org_only(s: str) -> bool:
    fold = _fold(s.lower())
    return fold in PARTY_ALIASES or fold in MEDIA_ORG

def _normalize_role(role: str) -> str | None:
    if role is None or (isinstance(role, float) and pd.isna(role)):
        return None
    r = _nfkc(_normalize_quotes(str(role))).strip()
    r = re.sub(r"\s+", " ", r)
    # short role junk
    if r.lower() in {"gast","gäste","thema","talk","talkshow"}: 
        return None
    if _is_party_or_org_only(r):
        return None
    return r or None

def _normalize_party(party: str) -> str | None:
    if party is None or (isinstance(party, float) and pd.isna(party)):
        return None
    key = _fold(str(party).strip().lower())
    return PARTY_ALIASES.get(key, party if party else None)

def _split_tokens_fold(name: str):
    toks = [t for t in _fold(name.lower()).split() if t not in PARTICLES]
    first = toks[0] if toks else ""
    last  = toks[-1] if len(toks) > 1 else ""
    return first, last

def _de_umlaut_fold(s: str) -> str:
    return (s.replace("ä","ae").replace("ö","oe").replace("ü","ue")
              .replace("Ä","Ae").replace("Ö","Oe").replace("Ü","Ue")
              .replace("ß","ss"))

def _squash_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _is_trailing_paren_junk(raw: str) -> bool:
    if raw is None:
        return False
    s = _nfkc(_normalize_quotes(str(raw))).strip()
    return bool(s and s.endswith(")"))


# ==============================
# 2) GUEST NAME CLEANING
# ==============================
def _clean_name_core(raw: str) -> str:
    s = _nfkc(_normalize_quotes(str(raw))).strip()
    s = LEADING_NUM.sub("", s)
    s = URL_RE.sub(" ", s)
    s = re.sub(r"\s*[\(\[\{<].*?[\)\]\}>]\s*", " ", s)             # drop (...) blocks
    s = TITLES_RE.sub(" ", s)                                      # drop titles
    s = re.split(r"\s+(?:–|—|als|,|-)\s+", s, maxsplit=1)[0]      # cut after comma/dash tail, but not for intra-word hyphens
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[)\]}>]+$", "", s).strip()
    return s

# ==============================
# 3) GUEST CONSOLIDATION (compact)
# ==============================
def clean_guest_rows(df: pd.DataFrame,
                     name_col="name",
                     role_col="role",
                     party_col="party",
                     show_col="Talkshow"):
    """Return (df_clean, df_dropped) with reasons for drops and normalized fields."""
    df = df.copy()

    # --- early drop of trailing-paren junk ---
    mask_trailing = df[name_col].map(_is_trailing_paren_junk)
    df_trailing = df[mask_trailing].copy()
    df_trailing["reject_reason"] = "trailing_paren_junk"

    # continue with the rest (on the non-junk)
    df = df[~mask_trailing].copy()

    # Clean raw name -> candidate
    df["name_clean"] = df[name_col].map(_clean_name_core)

    # Build rejection reasons
    reasons = []
    for s in df["name_clean"]:
        if not s or s.strip() == "":
            reasons.append("empty_after_cleaning")
            continue

        if _is_party_or_org_only(s):
            reasons.append("party_or_org_token")
            continue

        # Reject if it looks like an organization
        if re.search(r"\b(e\.V\.|gGmbH|GmbH|AG|e\. V)\b", s, re.IGNORECASE):
            reasons.append("org_pattern_found")
            continue

        tokens = TOKEN_RE.findall(s)
        if len(tokens) == 0:
            reasons.append("no_alpha_token")
            continue

        # Allow single-token names if they look substantial (e.g., "Cher", "Pelé")
        if len(tokens) == 1 and len(tokens[0]) <= 2:
            reasons.append("too_short_token")
            continue

        # Reject if name contains junk words that indicate it's a phrase
        if any(t.lower() in JUNK_WORDS_IN_NAMES for t in tokens):
            reasons.append("contains_junk_phrase_words")
            continue

        # Reject if any token is a role word
        if any(t.lower() in ROLE_WORDS for t in tokens):
            reasons.append("contains_role_word")
            continue

        reasons.append(None)
    df["reject_reason"] = reasons

    df_clean = df[df["reject_reason"].isna()].copy()
    df_dropped = df[~df["reject_reason"].isna()].copy()

    # Normalize role & party; keep original columns too
    if role_col in df_clean.columns:
        df_clean["role_clean"] = df_clean[role_col].map(_normalize_role)
    if party_col in df_clean.columns:
        df_clean["party_norm"] = df_clean[party_col].map(_normalize_party)

    # Optional: light guard against one-token names on ultra-common surnames
    # (comment this out if unnecessary)
    first_last = df_clean["name_clean"].map(_split_tokens_fold)
    df_clean["first_fold"] = first_last.map(lambda x: x[0])
    df_clean["last_fold"]  = first_last.map(lambda x: x[1])

    # Summary
    total = len(df)
    kept  = len(df_clean)
    print(f"[guest-clean] kept {kept}/{total} rows ({kept/total:.1%})")
    if len(df_dropped):
        print("  dropped by reason:")
        print(df_dropped["reject_reason"].value_counts())

    # Tidy columns
    drop_cols = ["first_fold","last_fold"]
    for c in drop_cols:
        if c in df_clean.columns:
            df_clean.drop(columns=[c], inplace=True)

    return df_clean.reset_index(drop=True), df_dropped.reset_index(drop=True)


# ==============================
# 4) DESCRIPTION CLEANING (for topic labels)
# ==============================

# NLP cleaning deps
try:
    import spacy
    nlp_de = spacy.load("de_core_news_md", disable=["parser", "ner"])
except Exception:
    try:
        import spacy
        nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])
        print("Using spaCy 'sm' model. For better lemmatization run: python -m spacy download de_core_news_md")
    except Exception:
        nlp_de = None
        print("spaCy German model not available; proceeding without lemmatization.")

@lru_cache(maxsize=None)
def get_german_stopwords() -> set[str]:
    """
    Loads German stopwords from two standard sources, downloading them if necessary.
    Combines them with a custom list of domain-specific stopwords.
    The result is cached to avoid repeated file I/O in a single run.

    Returns:
        A set of German stopwords.
    """
    if requests is None:
        raise ImportError("The 'requests' library is required to download stopwords. Please run 'pip install requests'.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "solariz": "https://raw.githubusercontent.com/solariz/german_stopwords/master/german_stopwords_plain.txt",
        "iso": "https://raw.githubusercontent.com/stopwords-iso/stopwords-de/master/stopwords-de.txt"
    }
    
    stopword_set = set()

    for name, url in urls.items():
        filepath = DATA_DIR / f"stopwords_de_{name}.txt"
        
        if not filepath.exists():
            print(f"Downloading stopwords from '{name}' source...")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                filepath.write_text(response.text, encoding='utf-8')
                print(f"Saved to {filepath}")
            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not download stopwords from {url}. Error: {e}")
                continue

        if filepath.exists():
            try:
                content = filepath.read_text(encoding='utf-8')
                # Filter out comments and empty lines
                words = {line.strip() for line in content.splitlines() if line.strip() and not line.startswith('#')}
                stopword_set.update(words)
            except Exception as e:
                print(f"Warning: Could not read stopwords from {filepath}. Error: {e}")

    # Add custom domain-specific stopwords
    EXTRA_STOPWORDS = {
        # General talkshow/news context
        "gast", "gäste", "gaeste", "thema", "themen", "talk", "talkshow", "sendung",
        "folge", "folgen", "heute", "morgen", "gestern", "moderator", "moderation",
        "gastmoderator", "diskussion", "diskutieren", "diskutiert",
        "themenwoche", "interview", "interviews", "analyse", "analysen", "bericht", 
        "berichte", "kommentar", "kommentare", "nachgehakt",
        
        # Common roles (already in ROLE_WORDS, but good to have here for text cleaning)
        "journalist", "journalisten", "journalistin", "reporter", "reporterin",
        "korrespondent", "korrespondentin", "kommentator", "kommentatorin",
        "experte", "expertin", "experten",
        "politiker", "politikerin", "politik", "politologe", "politologin",
        "wissenschaftler", "wissenschaftlerin", "wissenschaft",
        "autor", "autorin", "fraktionsvorsitzend", "bundesvorsitzender",
        
        # Common entities/concepts and junk
        "deutsch", "deutsche", "deutschen", "deutscher", "deutschland", "bundesrepublik",
        "gesellschaft", "wirtschaft", "medien", "fernsehen",
        "jahr", "jahre", "jährige", "jährigen", "jähriger", "jähriges", "jährigem", "jaehrig", "uhr",
    }
    stopword_set.update(EXTRA_STOPWORDS)

    return stopword_set

def _strip_urls(s: str) -> str:
    return URL_RE.sub(" ", s)

def _strip_emojis(s: str) -> str:
    return EMOJI_RE.sub(" ", s)

def normalize_umlauts(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def clean_description_for_labels(text: str, cfg: LiteConfig = LiteConfig()) -> str:
    """Produce a lemmatized, POS-filtered, stopword-trimmed text for topic labeling."""
    if text is None:
        return ""
    s = _nfkc(_normalize_quotes(str(text)))
    if cfg.remove_urls:
        s = _strip_urls(s)
    if cfg.remove_emojis:
        s = _strip_emojis(s)
    s = _squash_spaces(s)
    for pat in BOILERPLATE_PATTERNS:
        s = pat.sub(" ", s)
    s = _squash_spaces(s)

    # No spaCy available → simple fallback: lowercase + token filter
    stop = get_german_stopwords()
    if _NLP_DE is None:
        toks = [t for t in TOKEN_RE.findall(s.lower())
                if len(t) >= 2 and t not in stop]
        return " ".join(toks)

    # Proper lemmatization + POS filter
    doc = _NLP_DE(s)
    # Combine spaCy's default list with our custom list for comprehensive coverage
    stop.update({w.lower() for w in _NLP_DE.Defaults.stop_words})
    toks = []
    for t in doc:
        if t.is_space or t.is_punct or t.like_url or t.like_email:
            continue
        if cfg.keep_pos and t.pos_ not in cfg.keep_pos:
            continue
        lemma = (t.lemma_ or t.text).lower().strip("._:;,'\"()[]!?-")
        if not lemma or lemma in stop or len(lemma) < 2:
            continue
        toks.append(lemma)
    return " ".join(toks)

def is_valid_cleaned_description(cleaned: str, cfg: LiteConfig = LiteConfig()) -> bool:
    return bool(cleaned) and (len(cleaned.split()) >= cfg.min_tokens_cleaned)

# helper: remove redundant n-grams (keep longer phrases first)
def _dedupe_topic_terms(topics_dict: dict[int, list[tuple[str, float]]],
                        keep_n: int = 10) -> dict[int, list[tuple[str, float]]]:
    new_repr: dict[int, list[tuple[str, float]]] = {}
    for tid, terms in topics_dict.items():
        if tid == -1 or not terms:
            new_repr[tid] = terms
            continue
        cand = sorted(terms, key=lambda x: (-len(x[0].split()), -x[1]))
        kept, kept_sets = [], []
        for w, s in cand:
            toks = tuple(t for t in w.split() if t)
            wset = set(toks)
            if any(wset <= ks for ks in kept_sets):  # drop if subset of a kept phrase
                continue
            kept.append((w, s))
            kept_sets.append(wset)
            if len(kept) >= keep_n:
                break
        new_repr[tid] = kept
    return new_repr

@lru_cache(maxsize=512)
def _phrase_lemmas(phrase: str) -> tuple[str, ...]:
    phrase = (phrase or "").strip()
    if not phrase:
        return tuple()
    normalized = normalize_umlauts(phrase.lower())
    if nlp_de is not None:
        doc = nlp_de(normalized)
        return tuple(
            (t.lemma_ or t.text).lower().strip("._:;,'\"()[]!?-")
            for t in doc if (t.lemma_ or t.text)
        )
    tokens = re.findall(r"\b\w+\b", normalized)
    return tuple(tokens)

def _filter_topic_terms(terms_by_topic: dict[int, list[tuple[str, float]]]) -> dict[int, list[tuple[str, float]]]:
    filtered: dict[int, list[tuple[str, float]]] = {}
    for tid, terms in terms_by_topic.items():
        if tid == -1 or not terms:
            filtered[tid] = terms
            continue
        seen_keys: set[tuple[str, ...]] = set()
        cleaned: list[tuple[str, float]] = []
        for word, score in terms:
            lemmas = tuple(l for l in _phrase_lemmas(word) if l)
            if not lemmas:
                continue
            if all(l in LOW_INFORMATION_LEMMAS for l in lemmas):
                continue
            key = lemmas
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cleaned.append((word, score))
        if not cleaned:
            cleaned = terms
        filtered[tid] = cleaned
    return filtered
