from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def _load_topic_labels() -> dict[str, str]:
    """Lädt data/topic_labels.json, falls vorhanden. Cached."""
    path = _DATA_DIR / "topic_labels.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _most_common_value(series: pd.Series) -> str | None:
    """Return the most frequent non-empty value in a series."""
    if series is None:
        return None
    valid = series.dropna().astype(str).str.strip()
    valid = valid[valid != ""]
    if valid.empty:
        return None
    counts = valid.value_counts()
    return counts.index[0] if not counts.empty else None


def _summarize_roles(series: pd.Series) -> str | None:
    """Build a short, unique role summary for a guest."""
    if series is None:
        return None
    roles = [role.strip() for role in series.dropna().astype(str) if role and role.strip()]
    if not roles:
        return None
    unique_roles = list(dict.fromkeys(roles))
    summary = ", ".join(unique_roles[:3])
    if len(unique_roles) > 3:
        summary += ", ..."
    return summary


def _role_parts(role: object) -> list[str]:
    """Split a raw role string into cleaned comma-separated parts."""
    if role is None or (isinstance(role, float) and pd.isna(role)):
        return []
    text = " ".join(str(role).split()).strip(" ,")
    if not text:
        return []
    return [part.strip(" ,;") for part in re.split(r"[;,]", text) if part and part.strip(" ,;")]


def _normalize_role_fragment(text: str, primary_party: str | None) -> str:
    """Map common role variants to a stable display form."""
    normalized = " ".join(text.split()).strip(" ,;")
    if not normalized:
        return ""

    if primary_party:
        party_pattern = re.escape(primary_party)
        normalized = re.sub(rf"^{party_pattern}[- ]+", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(rf"[- ]+{party_pattern}$", "", normalized, flags=re.IGNORECASE)
        normalized = normalized.strip(" ,;")

    lower = normalized.casefold()

    pattern_map = [
        (
            r"\b(bundesminister für gesundheit|bundesgesundheitsminister|früherer bundesgesundheitsminister)\b",
            "Bundesminister für Gesundheit",
        ),
        (
            r"\b(mdb|mitglied des deutschen bundestages|mitglied des bundestags|bundestagsabgeordneter|bundestagsabgeordnete|abgeordneter des deutschen bundestages)\b",
            "Mitglied des Deutschen Bundestages",
        ),
        (
            r"\b(gesundheitsexperte|gesundheitspolitiker|gesundheitsexperte der spd)\b",
            "Gesundheitspolitiker",
        ),
        (
            r"\b(stellvertretender fraktionsvorsitzender|stellvertretender spd-fraktionsvorsitzender|stellvertretender vorsitzender der bundestagsfraktion|stellvertretender vorsitzender der spd-bundestagsfraktion)\b",
            "Stellvertretender Fraktionsvorsitzender",
        ),
        (
            r"\b(epidemiologe gesundheitsökonom|gesundheitsökonom und epidemiologe)\b",
            "Gesundheitsökonom und Epidemiologe",
        ),
    ]

    for pattern, replacement in pattern_map:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            return replacement

    replacements = {
        "Bundestags": "Bundestages",
        "stellv.": "stellvertretender",
        "stv.": "stellvertretender",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _canonical_role_key(text: str) -> str:
    """Normalize a role fragment for duplicate detection."""
    normalized = " ".join(text.split()).strip(" ,")
    replacements = {
        "Bundestags": "Bundestages",
        "stellv.": "stellvertretender",
        "stv.": "stellvertretender",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized.casefold()


def _role_priority(text: str) -> int:
    """Rank specific office/function labels ahead of generic descriptors."""
    key = _canonical_role_key(text)
    if "bundesminister" in key:
        return 5
    if "fraktionsvorsitz" in key:
        return 4
    if "bundestag" in key or key == "mitglied des deutschen bundestages":
        return 3
    if "epidemiologe" in key or "gesundheitsökonom" in key or "gesundheitspolitiker" in key:
        return 2
    if key == "politiker":
        return 0
    return 1


def _should_suppress_role(text: str, all_keys: set[str]) -> bool:
    """Hide overly generic roles when more specific ones are available."""
    key = _canonical_role_key(text)
    if key == "politiker":
        specific_markers = (
            "bundesminister",
            "bundestag",
            "fraktionsvorsitz",
            "epidemiologe",
            "gesundheitsökonom",
            "gesundheitspolitiker",
        )
        return any(marker in candidate for marker in specific_markers for candidate in all_keys)
    return False


def _summarize_roles_for_guest(group: pd.DataFrame, primary_party: str | None) -> str | None:
    """Build a short role summary, de-duplicated and ordered by most recent date."""
    if group.empty or "role" not in group.columns:
        return None

    if "date" in group.columns:
        date_series = pd.to_datetime(group["date"], errors="coerce")
    else:
        date_series = pd.Series([pd.NaT] * len(group), index=group.index)

    fragments: dict[str, dict[str, object]] = {}
    party_key = _canonical_role_key(primary_party) if primary_party else None

    for idx, role in group["role"].items():
        for fragment in _role_parts(role):
            fragment = _normalize_role_fragment(fragment, primary_party)
            fragment_key = _canonical_role_key(fragment)
            if not fragment_key:
                continue
            if party_key and fragment_key == party_key:
                continue

            fragment_date = date_series.loc[idx] if idx in date_series.index else pd.NaT
            entry = fragments.setdefault(
                fragment_key,
                {
                    "text": fragment,
                    "latest_date": pd.NaT,
                    "count": 0,
                    "priority": _role_priority(fragment),
                    "first_index": len(fragments),
                },
            )
            entry["count"] = int(entry["count"]) + 1
            if pd.notna(fragment_date) and (
                pd.isna(entry["latest_date"]) or fragment_date > entry["latest_date"]
            ):
                entry["latest_date"] = fragment_date

    if not fragments:
        return None

    fragment_keys = set(fragments.keys())
    filtered_values = [
        item for item in fragments.values() if not _should_suppress_role(str(item["text"]), fragment_keys)
    ]
    if filtered_values:
        values_to_sort = filtered_values
    else:
        values_to_sort = list(fragments.values())

    ordered = sorted(
        values_to_sort,
        key=lambda item: (
            0 if pd.notna(item["latest_date"]) else 1,
            -(item["latest_date"].value if pd.notna(item["latest_date"]) else 0),
            -int(item["priority"]),
            -int(item["count"]),
            int(item["first_index"]),
        ),
    )
    summary_parts = [str(item["text"]) for item in ordered[:3]]
    summary = ", ".join(summary_parts)
    if len(ordered) > 3:
        summary += ", ..."
    return summary or None


def format_topic_label(label: str | float | None) -> str:
    """
    Wandelt BERTopic-interne Labels in lesbare Anzeigenamen um.
    Schaut zuerst in data/topic_labels.json; fällt auf automatisch
    generierte Darstellung zurück wenn kein Eintrag vorhanden.
    """
    custom = _load_topic_labels()

    if not isinstance(label, str):
        return custom.get("-1", "Kein Thema zugeordnet")

    text = label.strip()
    if not text:
        return custom.get("-1", "Kein Thema zugeordnet")

    # Numerisches Präfix extrahieren ("3_bundestagswahl_..." → "3")
    parts = text.split(" ", 1)
    prefix = parts[0].split("_")[0]  # auch Unterstrich-Format ("3_bundestagswahl")

    if prefix.lstrip("-").isdigit():
        if prefix in custom:
            return custom[prefix]
        if prefix == "-1":
            return custom.get("-1", "Sonstige / kein Thema")
        remainder = parts[1].strip() if len(parts) > 1 else ""
        # Unterstrich-Format normalisieren
        remainder = remainder.replace("_", " ").strip()
        return f"Topic {prefix}: {remainder}" if remainder else f"Topic {prefix}"

    return text


def prepare_guest_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Create a helper table with party and role summaries per guest."""
    if df.empty:
        return pd.DataFrame(columns=["name", "primary_party", "known_roles"])
    if "name" not in df.columns:
        return pd.DataFrame(columns=["name", "primary_party", "known_roles"])

    records: list[dict[str, str | None]] = []
    for name, group in df.groupby("name", sort=False):
        primary_party = _most_common_value(group["party"]) if "party" in group.columns else None
        known_roles = _summarize_roles_for_guest(group, primary_party)
        records.append(
            {
                "name": name,
                "primary_party": primary_party,
                "known_roles": known_roles,
            }
        )

    return pd.DataFrame(records)


TIMEFRAME_OPTIONS: dict[str, pd.DateOffset | None] = {
    "6 Monate": pd.DateOffset(months=6),
    "Letzte 12 Monate": pd.DateOffset(months=12),
    "2 Jahre": pd.DateOffset(years=2),
    "Gesamter Zeitraum": None,
}


def filter_by_timeframe(
    df: pd.DataFrame | None, timeframe_label: str
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """Return a copy of the data filtered to the chosen timeframe."""
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(), None, None

    date_series = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = date_series.dropna()
    if valid_dates.empty:
        return pd.DataFrame(), None, None

    latest_date = valid_dates.max()
    offset = TIMEFRAME_OPTIONS.get(timeframe_label)

    if offset is None or timeframe_label == "Gesamter Zeitraum":
        filtered = df.copy()
        start_date = valid_dates.min()
    else:
        start_date = latest_date - offset
        filtered = df[date_series >= start_date].copy()

    return filtered, start_date, latest_date


def summarize_show_coverage(df: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize date coverage and episode count per show."""
    if (
        df is None
        or df.empty
        or "show" not in df.columns
        or "date" not in df.columns
    ):
        return pd.DataFrame(columns=["show", "first_date", "last_date", "episodes"])

    working = df.dropna(subset=["show"]).copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working[working["date"].notna()]
    if working.empty:
        return pd.DataFrame(columns=["show", "first_date", "last_date", "episodes"])

    date_summary = (
        working.groupby("show")["date"]
        .agg(["min", "max"])
        .rename(columns={"min": "first_date", "max": "last_date"})
    )

    if "uid" in working.columns:
        episode_counts = working.groupby("show")["uid"].nunique()
    else:
        episode_counts = working.groupby("show").size()

    summary = date_summary.join(episode_counts.rename("episodes"))
    return summary.reset_index()


def summarize_topic_counts(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return episode counts per topic, including unclassified episodes."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Thema", "Episoden"])

    working = df.copy()
    if "topic_label" in working.columns:
        working["topic_display"] = working["topic_label"].apply(format_topic_label)
    else:
        working["topic_display"] = "Kein Thema zugeordnet"

    if "topic" in working.columns:
        working.loc[working["topic"].isna(), "topic_display"] = "Kein Thema zugeordnet"
        working.loc[working["topic"] == -1, "topic_display"] = "Sonstige / kein Thema"

    topic_counts = (
        working["topic_display"].fillna("Kein Thema zugeordnet").value_counts()
    )

    return (
        topic_counts.rename_axis("Thema")
        .reset_index(name="Episoden")
        .sort_values("Episoden", ascending=False)
    )
