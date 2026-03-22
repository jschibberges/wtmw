#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
from datetime import datetime, timedelta, date
import locale
import hashlib
import base64
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    Progress = None
    Table = None
    _RICH_AVAILABLE = False


# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Canonical show name mapping – normalises scraping inconsistencies from fernsehserien.de
_SHOW_NAME_NORMALIZATIONS: dict[str, str] = {
    "Hart Aber Fair": "hart aber fair",
    "Hart aber Fair": "hart aber fair",
    "HART ABER FAIR": "hart aber fair",
}

uids: set = set()

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}
REQUEST_TIMEOUT = (5, 20)
MAX_WORKERS = 6
# Some guides are not reliably ordered newest -> oldest, so early stop is
# disabled by default. Re-enable only if ordering has been verified.
KNOWN_UID_STREAK_STOP = None
CHECKPOINT_EVERY = 5
VALIDATION_OVERRIDES_PATH = DATA_DIR / "guest_validation_overrides.json"
VALIDATION_REVIEW_PATH = DATA_DIR / "guest_validation_review.xlsx"
_thread_local = threading.local()
console = Console(soft_wrap=True) if _RICH_AVAILABLE and Console is not None else None


def log_info(message: str) -> None:
    if console is not None:
        console.print(f"[cyan]{message}[/cyan]")
    else:
        print(message)


def log_success(message: str) -> None:
    if console is not None:
        console.print(f"[green]{message}[/green]")
    else:
        print(message)


def log_warning(message: str) -> None:
    if console is not None:
        console.print(f"[yellow]{message}[/yellow]")
    else:
        print(message)


def log_error(message: str) -> None:
    if console is not None:
        console.print(f"[red]{message}[/red]")
    else:
        print(message)


def log_section(title: str) -> None:
    if console is not None:
        console.rule(f"[bold blue]{title}[/bold blue]")
    else:
        print(f"\n--- {title} ---")


def print_key_value_table(title: str, rows: list[tuple[str, object]]) -> None:
    if console is None or Table is None:
        print(title)
        for key, value in rows:
            print(f"  {key}: {value}")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(table)


def _build_session() -> requests.Session:
    """Create a requests session with retry/backoff and standard browser headers."""
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _get_thread_session() -> requests.Session:
    """Return a per-thread session so concurrent fetches still reuse connections safely."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _build_session()
        _thread_local.session = session
    return session

def load_json_file(filepath):
    """Loads data from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns: 
        The loaded JSON data as a Python dictionary or list, or None if 
        an error occurs during loading.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # Explicitly handle encoding
            data = json.load(f)
        return data
    except FileNotFoundError:
        log_warning(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        log_error(f"Invalid JSON format in {filepath}")
        return []
    except Exception as e:  # Catch other potential errors
        log_error(f"Unexpected error while loading {filepath}: {e}")
        return []

def save_json_file(filepath, data):
    """Saves data to a JSON file.

    Args:
        filepath: The path to the JSON file.
        data: The Python dictionary or list to save.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        log_success(f"Saved data to {filepath}")
    except Exception as e:
        log_error(f"Error while saving to {filepath}: {e}")


@lru_cache(maxsize=1)
def _load_guest_validation_overrides() -> dict[str, dict]:
    """Load manual guest validation overrides from JSON."""
    if not VALIDATION_OVERRIDES_PATH.exists():
        return {}
    data = load_json_file(VALIDATION_OVERRIDES_PATH)
    if not isinstance(data, dict):
        return {}
    normalized: dict[str, dict] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            status = value
            note = ""
        elif isinstance(value, dict):
            status = value.get("status")
            note = str(value.get("note", "")).strip()
        else:
            continue
        if status not in {"accept", "reject"}:
            continue
        normalized[_normalize_override_key(key)] = {"status": status, "note": note}
    return normalized


def _normalize_override_key(name: str) -> str:
    """Normalize a guest name key for stable override lookups."""
    return re.sub(r"\s+", " ", str(name or "").strip())


def _merge_episode_records(existing_data: list[dict], new_records: list[dict]) -> list[dict]:
    """Merge episode records by UID while preserving existing order."""
    merged: dict[str, dict] = {}
    ordered_records: list[dict] = []

    for record in existing_data + new_records:
        uid = record.get("uid")
        if uid is None:
            ordered_records.append(record)
            continue
        if uid not in merged:
            ordered_records.append(record)
        merged[uid] = record

    result: list[dict] = []
    seen_uids: set[str] = set()
    for record in ordered_records:
        uid = record.get("uid")
        if uid is None:
            result.append(record)
            continue
        if uid in seen_uids:
            continue
        result.append(merged[uid])
        seen_uids.add(uid)
    return result


def _flush_checkpoint(
    output_path: Path | None,
    existing_data: list[dict] | None,
    pending_records: list[dict],
) -> list[dict]:
    """Persist pending scraped episodes for a show and return the updated dataset."""
    if not output_path or not pending_records:
        return existing_data or []

    merged_data = _merge_episode_records(existing_data or [], pending_records)
    save_json_file(output_path, merged_data)
    pending_records.clear()
    return merged_data


def _sanitize_episode_records(records: list | None) -> list[dict]:
    """Keep only dict-shaped episode records with a uid."""
    cleaned: list[dict] = []
    for record in records or []:
        if isinstance(record, dict) and record.get("uid"):
            cleaned.append(record)
    return cleaned

all_annewill_data: list = []
all_carenmiosga_data: list = []
all_hartaberfair_data: list = []
all_markuslanz_data: list = []
all_maischberger_data: list = []
all_illner_data: list = []
all_data: list = []


def _load_existing_data() -> None:
    """Load existing scraped data from JSON files into module-level variables.

    Called at the start of main() to avoid side effects on import.
    Uses DATA_DIR-relative paths for portability.
    """
    global all_annewill_data, all_carenmiosga_data, all_hartaberfair_data
    global all_markuslanz_data, all_maischberger_data, all_illner_data, all_data, uids

    all_annewill_data = _sanitize_episode_records(load_json_file(DATA_DIR / "AnneWill_data.json"))
    all_carenmiosga_data = _sanitize_episode_records(load_json_file(DATA_DIR / "CarenMiosga_data.json"))
    all_hartaberfair_data = _sanitize_episode_records(load_json_file(DATA_DIR / "HartAberFair_data.json"))
    all_markuslanz_data = _sanitize_episode_records(load_json_file(DATA_DIR / "MarkusLanz_data.json"))
    all_maischberger_data = _sanitize_episode_records(load_json_file(DATA_DIR / "Maischberger_data.json"))
    all_illner_data = _sanitize_episode_records(load_json_file(DATA_DIR / "Illner_data.json"))

    all_data = (
        all_annewill_data + all_carenmiosga_data + all_hartaberfair_data +
        all_markuslanz_data + all_maischberger_data + all_illner_data
    )

    uids = {data["uid"] for data in all_data if "uid" in data}
    print_key_value_table(
        "Existing Dataset Snapshot",
        [
            ("Anne Will", len(all_annewill_data)),
            ("Caren Miosga", len(all_carenmiosga_data)),
            ("Hart aber Fair", len(all_hartaberfair_data)),
            ("Markus Lanz", len(all_markuslanz_data)),
            ("Maischberger", len(all_maischberger_data)),
            ("Maybrit Illner", len(all_illner_data)),
            ("Total episodes", len(all_data)),
            ("Known UIDs", len(uids)),
        ],
    )


alternative_url_will = "https://www.fernsehserien.de/anne-will/episodenguide"
alternative_url_miosga = "https://www.fernsehserien.de/caren-miosga/episodenguide"
alternative_url_hartaberfair = "https://www.fernsehserien.de/hart-aber-fair/episodenguide"
alternative_url_maischberger = "https://www.fernsehserien.de/maischberger-ard/episodenguide"
alternative_url_illner = "https://www.fernsehserien.de/maybrit-illner/episodenguide"
alternative_url_markuslanz = "https://www.fernsehserien.de/markus-lanz/episodenguide"

# Locale für deutsche Monatsnamen einmalig setzen (nicht bei jedem standardize_date()-Aufruf)
try:
    locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, "de_DE")
    except locale.Error:
        pass  # Fallback: Locale bleibt wie gesetzt; Datumsformat 2 funktioniert dann ggf. nicht

PARTY_STRINGS = ["CDU", "CSU", "SPD", "freie wähler", "FDP", "BSW", "AfD", "DIE LINKE", "Linke", "parteilos"]
PARTY_NORMALIZATION_MAP = {
    # Variations for "Bündnis 90/Die Grünen"
    "B’90/Grüne":              "Bündnis 90/Die Grünen",  # einfaches Apostroph
    "B´90/Grüne":              "Bündnis 90/Die Grünen",  # Gravis-Apostroph
    "B´90/Die Grünen":         "Bündnis 90/Die Grünen",
    "Bündnis 90/ Die Grünen":  "Bündnis 90/Die Grünen",  # Leerzeichen nach /
    "Bündnis 90 / Die Grünen": "Bündnis 90/Die Grünen",  # Leerzeichen um /
    "Bündnis 90/Die Grünen":   "Bündnis 90/Die Grünen",  # kanonische Form
}
GREEN_PARTY_PATTERN = re.compile(r"B[^.]{0,10}90[^.]*?[gG]rün\w{1,2}\b", flags=re.IGNORECASE)
NAME_TOKEN_RE = re.compile(r"[^\W\d_][^\W\d_'’`´.-]*", flags=re.UNICODE)
NAME_PARTICLES = {"von", "zu", "zur", "zum", "de", "del", "der", "van", "la", "le"}
NOBLE_TITLES = {"prinz", "fürst", "fürstin", "freifrau", "baron", "graf", "gräfin"}
ORG_TERMS = {
    "initiative", "zeitung", "institut", "stiftung", "universität", "schule",
    "klinik", "museum", "verband", "verein", "zentrum", "kirche", "polizei",
    "bundeswehr", "redaktion", "kanal", "nachrichten", "fraktion", "partei",
    "ministerium", "bundestag", "firma", "gmbh", "ag",
}
ROLE_TERMS = {
    "minister", "ministerin", "bundeskanzler", "bundeskanzlerin", "abgeordneter",
    "abgeordnete", "journalist", "journalistin", "moderator", "moderatorin",
    "präsident", "präsidentin", "vorsitzender", "vorsitzende", "direktor",
    "direktorin", "experte", "expertin", "unternehmer", "unternehmerin",
    "bürgermeister", "bürgermeisterin", "sprecher", "sprecherin",
}
ROLE_SUFFIXES = (
    "minister", "ministerin", "kanzler", "kanzlerin", "präsident", "präsidentin",
    "abgeordneter", "abgeordnete", "journalist", "journalistin", "moderator",
    "moderatorin", "vorsitzender", "vorsitzende", "direktor", "direktorin",
    "sprecher", "sprecherin", "experte", "expertin", "redakteur", "redakteurin",
)
PROMO_PATTERNS = (
    "hier auf", "erfahrt ihr", "diskutiert", "wir sorgen für", "bildet euch",
    "mehr infos", "jetzt live", "abonnier", "abonniert", "folgt uns",
)
STOPWORD_LIKE_NAME_TOKENS = {
    "hier", "auf", "und", "oder", "mit", "ohne", "für", "von", "der", "die",
    "das", "des", "den", "dem", "ihr", "wir", "euch", "uns", "alle", "etwas",
    "passiert", "sorgen", "diskutiert", "bildet", "welt",
}
GROUP_TERMS = {"familie", "team", "initiative"}

# --- Utility Functions ---
def create_hash(url, length=8):
    """
    Create a unique and short hash for a given URL.
    
    Args:
        url (str): The URL to hash.
        length (int): The desired length of the short hash. Default is 8.
    
    Returns:
        str: A short hash of the URL.
    """
    # Generate a SHA256 hash of the URL
    sha256_hash = hashlib.sha256(url.encode()).digest()
    # Encode the hash in Base64 for shorter representation
    base64_hash = base64.urlsafe_b64encode(sha256_hash).decode()
    # Truncate to the desired length
    return base64_hash[:length]

def extract_party(text):
    """
    Extract items from the list that are partial matches in the string (case insensitive).
    
    Args:
        text (str): The string to check against.
    
    Returns:
        string: The matching string from the text.
    """
    if not text:
        return None

    # Normalize common variations
    for variation, canonical in PARTY_NORMALIZATION_MAP.items():
        if variation in text:
            return canonical

    # Check for standard party names
    text_lower = text.lower()
    for party in PARTY_STRINGS:
        if party.lower() in text_lower:
            return party

    # Check for regex patterns for "Grüne"
    if GREEN_PARTY_PATTERN.search(text) or "Die Grünen" in text:
        return "Bündnis 90/Die Grünen"

    return None

def clean_name(name:str):
    """
    Cleans a guest's name string, removing titles and extracting party affiliation.
    """
    if not name:
        return "", None

    # Remove titles
    name = name.replace("Prof.","").replace("Dr.","").strip()

    # Split name from other info (party, role)
    # Use regex to find the first occurrence of a separator
    match = re.search(r'[,(:]', name)
    if match:
        separator_index = match.start()
        main_name = name[:separator_index].strip()
        other_info = name[separator_index:].strip()
    else:
        main_name = name
        other_info = ""

    # Clean up name from leading colons if any
    if ":" in main_name:
        main_name = main_name.split(":")[-1].strip()

    party = extract_party(other_info)

    return main_name, party


def get_guest_validation_override(cleaned_name: str, raw_name: str = "") -> tuple[str, str] | None:
    """Return a manual validation override for a guest if configured."""
    overrides = _load_guest_validation_overrides()
    for candidate in (_normalize_override_key(cleaned_name), _normalize_override_key(raw_name)):
        if candidate and candidate in overrides:
            override = overrides[candidate]
            note = override.get("note", "")
            reason = "manual_override"
            if note:
                reason = f"{reason}:{note}"
            return override["status"], reason
    return None


def classify_guest_name(candidate_name: str, role_text: str = "") -> tuple[str, str]:
    """Classify a scraped guest candidate as accept, review, or reject.

    The goal is not to enforce classic legal names, but to reject obvious
    non-person artifacts while preserving mononyms, artist names, and alias forms.
    """
    name = re.sub(r"\s+", " ", str(candidate_name or "").strip())
    role = re.sub(r"\s+", " ", str(role_text or "").strip())
    low = name.casefold()
    role_low = role.casefold()

    if not name:
        return "reject", "empty_name"
    if len(name) > 80:
        return "reject", "name_too_long"
    if any(pat in low or pat in role_low for pat in PROMO_PATTERNS):
        return "reject", "promo_text"
    if re.search(r"https?://|www\.", name, flags=re.IGNORECASE):
        return "reject", "contains_url"
    if "?" in name:
        return "reject", "sentence_punctuation"
    if "!" in name or ";" in name:
        return "review", "unusual_punctuation"

    tokens = NAME_TOKEN_RE.findall(name)
    tokens_low = [t.casefold().strip(".-") for t in tokens if t]
    token_count = len(tokens_low)
    if token_count == 0:
        return "reject", "no_name_tokens"
    if token_count > 6:
        return "reject", "too_many_tokens"

    if low in {party.casefold() for party in PARTY_STRINGS}:
        return "reject", "party_token"
    if any(term in tokens_low for term in GROUP_TERMS):
        return "reject", "group_label"
    if any(term in tokens_low for term in ORG_TERMS):
        return "review", "org_term_in_name"
    has_role_like_token = any(
        token in ROLE_TERMS or any(token.endswith(suffix) for suffix in ROLE_SUFFIXES)
        for token in tokens_low
    )
    if has_role_like_token and token_count <= 3:
        return "reject", "role_phrase"

    stopword_hits = sum(token in STOPWORD_LIKE_NAME_TOKENS for token in tokens_low)
    if stopword_hits >= max(2, token_count // 2 + 1):
        return "reject", "sentence_like_name"

    has_alias = " alias " in f" {low} "
    has_nickname_quotes = any(ch in name for ch in ('"', "„", "“", "‚", "‘"))
    if has_alias or has_nickname_quotes:
        if token_count <= 6:
            return "accept", "artist_or_alias_pattern"

    if token_count == 1:
        token = tokens_low[0]
        if len(token) >= 4 and token not in ORG_TERMS and token not in ROLE_TERMS:
            return "accept", "mononym"
        return "review", "weak_single_token"

    if token_count <= 6 and (
        any(token in NAME_PARTICLES for token in tokens_low)
        or any(token in NOBLE_TITLES for token in tokens_low)
    ):
        return "accept", "extended_person_name"

    if token_count in (2, 3, 4):
        return "accept", "person_like_multi_token"

    return "review", "unusual_name_shape"


def export_guest_validation_review(all_show_data: list[dict], output_path: Path = VALIDATION_REVIEW_PATH) -> None:
    """Export unresolved non-accept guest candidates for manual review."""
    rows: list[dict] = []
    for episode in all_show_data:
        if not isinstance(episode, dict):
            continue
        for guest in episode.get("guests") or []:
            if not isinstance(guest, dict):
                continue
            raw_name = str(guest.get("name", "")).strip()
            role_text = str(guest.get("role") or guest.get("description") or "").strip()
            cleaned_name, _ = clean_name(raw_name)
            override = get_guest_validation_override(cleaned_name, raw_name)
            if override is not None:
                continue

            status = guest.get("validation_status")
            reason = guest.get("validation_reason")
            if status is None or reason is None:
                status, reason = classify_guest_name(cleaned_name, role_text)

            if status == "accept":
                continue

            rows.append(
                {
                    "cleaned_name": cleaned_name,
                    "raw_name": raw_name,
                    "validation_status": status,
                    "validation_reason": reason,
                    "role": role_text,
                    "show": episode.get("show"),
                    "date": episode.get("date"),
                    "episode_title": episode.get("title"),
                    "episode_link": episode.get("link"),
                    "manual_status": "",
                    "manual_note": "",
                }
            )

    if not rows:
        pd.DataFrame(
            columns=[
                "cleaned_name", "raw_name", "validation_status", "validation_reason",
                "role", "show", "date", "episode_title", "episode_link",
                "manual_status", "manual_note",
            ]
        ).to_excel(output_path, index=False)
        return

    review_df = pd.DataFrame(rows)
    review_df.drop_duplicates(
        subset=["cleaned_name", "validation_status", "validation_reason"],
        inplace=True,
    )
    review_df.sort_values(
        by=["validation_status", "cleaned_name", "date"],
        inplace=True,
        na_position="last",
    )
    review_df.to_excel(output_path, index=False)

def standardize_date(date_string):
    """
    Standardizes date strings to DD.MM.YYYY format.

    Args:
        date_string: The date string to standardize.

    Returns:
        A standardized date string in DD.MM.YYYY format, or None if the
        input is invalid or cannot be parsed.

    Note: locale.LC_TIME is set once at module level (de_DE.UTF-8) for performance
    and thread-safety. No setlocale() call here.
    """
    try:
        # Attempt parsing with different formats
        try:
            # Format 1: DD.MM.YYYY
            date_object = datetime.strptime(date_string, '%d.%m.%Y')
        except ValueError:
            try:
                # Format 2: DD. Month YYYY
                date_object = datetime.strptime(date_string, '%d. %B %Y')
            except ValueError:
                try:
                   # Format 3: DD.MM.YYYY (with potential space after the day)
                    date_object = datetime.strptime(date_string.replace(" ", ""), '%d.%m.%Y')
                except ValueError:
                    return None  # Invalid format

        return date_object.strftime('%d.%m.%Y')

    except Exception as e:  # Catches other potential errors (e.g., TypeError)
        log_warning(f"Error parsing date '{date_string}': {e}")
        return None

def subtract_date(dt_obj):
    """
    Formats a datetime.datetime object to DD.MM.YYYY after subtracting one day.

    Args:
        dt_obj: A datetime.datetime object

    Returns:
         A string representing the date in DD.MM.YYYY format, one day
         before the provided datetime object.  Returns None if input is not a 
         datetime object.
    """
    if not isinstance(dt_obj, (datetime, date)):
        return None  # Or raise a TypeError if you prefer

    previous_day = dt_obj - timedelta(days=1)
    return previous_day.strftime('%d.%m.%Y')


def get_episode_urls_from_guide(guide_url):
    """
    Scrapes an episode guide page to find all individual episode URLs.

    Args:
        guide_url (str): The URL of the episode guide page.

    Returns:
        list: A list of unique URLs for individual episodes.
    """
    base_url = "https://www.fernsehserien.de"
    episode_urls = []
    try:
        session = _get_thread_session()
        response = session.get(
            guide_url,
            headers={"Referer": "https://www.fernsehserien.de/"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # The episode rows are <a> tags with itemprop="episode".
        # We find all of them directly, as they can be in multiple list containers.
        episode_rows = soup.find_all('a', itemprop='episode')
        
        for row in episode_rows:
            relative_link = row.get('href')
            if relative_link:
                episode_urls.append(base_url + relative_link)
                
    except requests.RequestException as e:
        log_error(f"Error fetching guide URL {guide_url}: {e}")
    
    return list(dict.fromkeys(episode_urls))


def _fetch_episode_details(episode_url: str):
    """Fetch and parse a single episode page."""
    session = _get_thread_session()
    response = session.get(
        episode_url,
        headers={"Referer": "https://www.fernsehserien.de/"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    episode_soup = BeautifulSoup(response.content, 'html.parser')

    episode_info = {"uid": create_hash(episode_url), "link": episode_url, "guests": []}

    # --- Date Extraction and Check ---
    episode_date_obj = None
    broadcast_container = episode_soup.find('ea-angaben')
    if not broadcast_container:
        log_warning("Skipping episode without broadcast info container ('ea-angaben').")
        return None

    date_tag = broadcast_container.find('ea-angabe-datum')
    if date_tag:
        date_text = date_tag.get_text(strip=True)
        date_match = re.search(r'(\d{1,2}\.\d{1,2}\.\d{4})', date_text)
        if date_match:
            date_str = date_match.group(1)
            date_str_standardized = standardize_date(date_str)
            if date_str_standardized:
                try:
                    episode_date_obj = datetime.strptime(date_str_standardized, '%d.%m.%Y').date()
                except (ValueError, TypeError):
                    log_warning(f"Could not parse date from '{date_str}'.")

    if not episode_date_obj or episode_date_obj > date.today():
        if episode_date_obj:
            log_warning(f"Skipping future episode from {episode_date_obj.strftime('%d.%m.%Y')}.")
        else:
            log_warning("Skipping episode without a valid date.")
        return None

    episode_info['date'] = episode_date_obj.strftime('%d.%m.%Y')

    show_name_h1 = episode_soup.find('h1', class_='serien-titel')
    if show_name_h1 and show_name_h1.a:
        raw_show = show_name_h1.a.get_text(strip=True)
        episode_info['show'] = _SHOW_NAME_NORMALIZATIONS.get(raw_show, raw_show)
    else:
        episode_info['show'] = None

    station_tag = broadcast_container.find('ea-angabe-sender')
    if station_tag:
        episode_info['station'] = station_tag.get_text(strip=True)
    else:
        episode_info['station'] = None

    episode_title_tag = episode_soup.find('h1', class_='episode-title')
    if not episode_title_tag:
        episode_title_tag = episode_soup.find('h3', class_='episode-output-titel')

    if episode_title_tag and episode_title_tag.find('span', itemprop='name'):
        episode_info['title'] = episode_title_tag.find('span', itemprop='name').get_text(strip=True)
    else:
        episode_info['title'] = None

    inhalt_div = episode_soup.find('div', class_='episode-output-inhalt-inner')
    if inhalt_div:
        inhalt_clone = BeautifulSoup(str(inhalt_div), 'html.parser')
        for tag in inhalt_clone.find_all(['span', 'werbung', 'ins', 'script']):
            tag.decompose()

        full_text = inhalt_clone.get_text(separator='|||', strip=True)
        parts = re.split(r'Die Gäste:|||', full_text, maxsplit=1, flags=re.IGNORECASE)
        description_parts = []

        if parts[0]:
            description_parts.append(parts[0])

        if len(parts) > 1:
            lines_after_guests = [line.strip() for line in parts[1].split('|||') if line.strip()]
            for i, line in enumerate(lines_after_guests):
                if line and (len(line.split()) > 10 or line.endswith('.') or line.endswith('?')):
                    description_parts.extend(lines_after_guests[i:])
                    break

        episode_info['description'] = ' '.join(description_parts).replace('|||', ' ').strip()
    else:
        episode_info['description'] = ""

    if "Die Gäste: " in episode_info.get('description'):
        episode_info['description'] = episode_info['description'].split("Die Gäste: ")[0].strip()

    cast_crew_list = episode_soup.find('ul', class_='cast-crew')
    if cast_crew_list:
        guest_items = cast_crew_list.find_all('li', itemscope=True, itemtype="http://schema.org/Person")
        for item in guest_items:
            dd_tag = item.find('dd')
            if dd_tag and dd_tag.p:
                p_strings = list(dd_tag.p.stripped_strings)
                if p_strings and p_strings[0].lower() == 'gast':
                    name_tag = item.find('dt', itemprop='name')
                    if name_tag:
                        name = name_tag.get_text(strip=True)
                        description = " ".join(p_strings[1:]) if len(p_strings) > 1 else ""

                        cleaned_name, party = clean_name(name)
                        if not party:
                            party = extract_party(description)
                        override = get_guest_validation_override(cleaned_name, name)
                        if override is not None:
                            validation_status, validation_reason = override
                        else:
                            validation_status, validation_reason = classify_guest_name(
                                cleaned_name, description
                            )
                        if validation_status == "reject":
                            continue
                        episode_info['guests'].append({
                            "name": cleaned_name,
                            "party": party,
                            "role": description,
                            "description": description,
                            "validation_status": validation_status,
                            "validation_reason": validation_reason,
                        })

    return episode_info

def get_episode_details(
    episode_urls: list,
    *,
    max_workers: int = MAX_WORKERS,
    stop_after_known_streak: int | None = KNOWN_UID_STREAK_STOP,
    on_episode=None,
) -> list:
    """
    Scrapes a single episode page for its details.
    """
    cnt_max = len(episode_urls)
    candidate_urls: list[tuple[int, str]] = []
    known_uid_streak = 0

    for idx, episode_url in enumerate(episode_urls, start=1):
        uid = create_hash(episode_url)
        if uid in uids:
            known_uid_streak += 1
            if stop_after_known_streak and known_uid_streak >= stop_after_known_streak:
                log_info(
                    f"Stopping early after {known_uid_streak} known episodes in a row; "
                    "assuming the guide is ordered newest to oldest."
                )
                break
            continue

        known_uid_streak = 0
        uids.add(uid)
        candidate_urls.append((idx, episode_url))

    if not candidate_urls:
        return []

    episode_data: list[tuple[int, dict]] = []
    worker_count = max(1, min(max_workers, len(candidate_urls)))

    progress = None
    task_id = None
    if _RICH_AVAILABLE and Progress is not None and console is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    progress_cm = progress if progress is not None else None
    if progress_cm is not None:
        progress_cm.__enter__()
        task_id = progress.add_task("Fetching episode details", total=len(candidate_urls))

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_fetch_episode_details, episode_url): (idx, episode_url)
                for idx, episode_url in candidate_urls
            }
            for future in as_completed(future_map):
                idx, episode_url = future_map[future]
                try:
                    episode_info = future.result()
                    if episode_info is not None:
                        episode_data.append((idx, episode_info))
                        if on_episode is not None:
                            on_episode(episode_info)
                except requests.RequestException as e:
                    log_error(f"Error fetching URL {episode_url}: {e}")
                except Exception as e:
                    log_error(f"Unexpected error while scraping {episode_url}: {e}")
                finally:
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)
    finally:
        if progress_cm is not None:
            progress_cm.__exit__(None, None, None)

    episode_data.sort(key=lambda item: item[0])
    return [episode for _, episode in episode_data]

def scrape_fernsehserien_episodeguide(
    url,
    *,
    existing_data: list[dict] | None = None,
    output_path: Path | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
):
    """
    Scrapes episode data from a fernsehserien.de episodenguide URL.

    This function first scrapes the main episode guide page to get links to
    individual episode pages. Then, it visits each episode page to extract
    detailed information, including guests.

    Args:
        url (str): The URL of the episodenguide page.

    Returns:
        tuple: (merged_show_data, stats) where stats summarizes the current run.
    """
    log_info(f"Scraping episode guide: {url}")
    episode_urls = get_episode_urls_from_guide(url)
    existing_uid_count = len(
        {record.get("uid") for record in (existing_data or []) if isinstance(record, dict) and record.get("uid")}
    )
    unknown_urls = sum(1 for episode_url in episode_urls if create_hash(episode_url) not in uids)
    print_key_value_table(
        "Guide Scan Summary",
        [
            ("Guide URL", url),
            ("Episode links found", len(episode_urls)),
            ("Known episodes", len(episode_urls) - unknown_urls),
            ("Episodes to fetch", unknown_urls),
        ],
    )

    checkpoint_data = list(existing_data or [])
    pending_records: list[dict] = []

    def handle_episode(episode_info: dict) -> None:
        nonlocal checkpoint_data
        pending_records.append(episode_info)
        if output_path and len(pending_records) >= checkpoint_every:
            checkpoint_data = _flush_checkpoint(output_path, checkpoint_data, pending_records)

    current_run_episode_data = get_episode_details(episode_urls, on_episode=handle_episode)

    if output_path:
        checkpoint_data = _flush_checkpoint(output_path, checkpoint_data, pending_records)
        current_run_valid_episodes = len(current_run_episode_data)
        net_new_episodes = max(0, len(checkpoint_data) - existing_uid_count)
        stats = {
            "episode_links_found": len(episode_urls),
            "known_episodes": len(episode_urls) - unknown_urls,
            "episodes_to_fetch": unknown_urls,
            "current_run_valid_episodes": current_run_valid_episodes,
            "current_run_episode_data": current_run_episode_data,
            "net_new_episodes": net_new_episodes,
            "stored_episodes": len(checkpoint_data),
            "output_path": output_path,
        }
        print_key_value_table(
            "Show Result",
            [
                ("Stored episodes", len(checkpoint_data)),
                ("Current-run valid episodes", current_run_valid_episodes),
                ("Net new episodes", net_new_episodes),
                ("Output", output_path),
            ],
        )
        return checkpoint_data, stats

    stats = {
        "episode_links_found": len(episode_urls),
        "known_episodes": len(episode_urls) - unknown_urls,
        "episodes_to_fetch": unknown_urls,
        "current_run_valid_episodes": len(current_run_episode_data),
        "current_run_episode_data": current_run_episode_data,
        "net_new_episodes": max(0, len(current_run_episode_data)),
        "stored_episodes": len(current_run_episode_data),
        "output_path": output_path,
    }
    return current_run_episode_data, stats

def main():
    """Main function to run the scraping process."""
    global all_annewill_data, all_carenmiosga_data, all_hartaberfair_data
    global all_markuslanz_data, all_maischberger_data, all_illner_data

    log_section("Talkshow Scraper")
    DATA_DIR.mkdir(exist_ok=True)
    _load_existing_data()

    # Using fernsehserien.de scraper
    log_section("Caren Miosga")
    miosga_fernsehserien_data, miosga_stats = scrape_fernsehserien_episodeguide(
        alternative_url_miosga,
        existing_data=all_carenmiosga_data,
        output_path=DATA_DIR / 'CarenMiosga_data.json',
    )
    if miosga_fernsehserien_data:
        all_carenmiosga_data = miosga_fernsehserien_data
    
    log_section("Anne Will")
    will_fernsehserien_data, will_stats = scrape_fernsehserien_episodeguide(
        alternative_url_will,
        existing_data=all_annewill_data,
        output_path=DATA_DIR / 'AnneWill_data.json',
    )
    if will_fernsehserien_data:
        all_annewill_data = will_fernsehserien_data
    
    log_section("Hart aber Fair")
    hartaberfair_fernsehserien_data, hartaberfair_stats = scrape_fernsehserien_episodeguide(
        alternative_url_hartaberfair,
        existing_data=all_hartaberfair_data,
        output_path=DATA_DIR / 'HartAberFair_data.json',
    )
    if hartaberfair_fernsehserien_data:
        all_hartaberfair_data = hartaberfair_fernsehserien_data
    
    log_section("Maischberger")
    maischberger_fernsehserien_data, maischberger_stats = scrape_fernsehserien_episodeguide(
        alternative_url_maischberger,
        existing_data=all_maischberger_data,
        output_path=DATA_DIR / 'Maischberger_data.json',
    )
    if maischberger_fernsehserien_data:
        all_maischberger_data = maischberger_fernsehserien_data
    
    log_section("Markus Lanz")
    markuslanz_fernsehserien_data, markuslanz_stats = scrape_fernsehserien_episodeguide(
        alternative_url_markuslanz,
        existing_data=all_markuslanz_data,
        output_path=DATA_DIR / 'MarkusLanz_data.json',
    )
    if markuslanz_fernsehserien_data:
        all_markuslanz_data = markuslanz_fernsehserien_data
    
    log_section("Maybrit Illner")
    illner_fernsehserien_data, illner_stats = scrape_fernsehserien_episodeguide(
        alternative_url_illner,
        existing_data=all_illner_data,
        output_path=DATA_DIR / 'Illner_data.json',
    )
    if illner_fernsehserien_data:
        all_illner_data = illner_fernsehserien_data

    # Aggregate all data and save to Excel
    log_section("Aggregation")
    all_data_final = _sanitize_episode_records(
        all_annewill_data + all_carenmiosga_data + all_hartaberfair_data +
        all_markuslanz_data + all_maischberger_data + all_illner_data
    )
    
    current_run_valid_episodes = _sanitize_episode_records(
        [
            *miosga_stats["current_run_episode_data"],
            *will_stats["current_run_episode_data"],
            *hartaberfair_stats["current_run_episode_data"],
            *maischberger_stats["current_run_episode_data"],
            *markuslanz_stats["current_run_episode_data"],
            *illner_stats["current_run_episode_data"],
        ]
    )
    
    # Deduplicate based on UID
    unique_data = {item['uid']: item for item in all_data_final}.values()
    df = pd.DataFrame(list(unique_data))
    df.to_excel(DATA_DIR / "all_data.xlsx", index=False)
    export_guest_validation_review(list(unique_data))
    print_key_value_table(
        "Aggregation Summary",
        [
            ("Total persisted episodes", len(df)),
            ("Current-run valid episodes", len(current_run_valid_episodes)),
            (
                "Current-run net new episodes",
                sum(
                    stats["net_new_episodes"]
                    for stats in [
                        miosga_stats,
                        will_stats,
                        hartaberfair_stats,
                        maischberger_stats,
                        markuslanz_stats,
                        illner_stats,
                    ]
                ),
            ),
            ("All-data workbook", DATA_DIR / "all_data.xlsx"),
            ("Validation review", VALIDATION_REVIEW_PATH),
        ],
    )

    df = pd.DataFrame(current_run_valid_episodes)
    df.to_excel(DATA_DIR / "all_data2.xlsx", index=False)
    log_success(f"Saved incremental scrape workbook to {DATA_DIR / 'all_data2.xlsx'}")

if __name__ == "__main__":
    main()
