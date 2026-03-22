from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REVIEW_PATH = DATA_DIR / "guest_validation_review.xlsx"
OVERRIDES_PATH = DATA_DIR / "guest_validation_overrides.json"
VALID_STATUSES = {"accept", "reject"}


def _normalize_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value or "").split()).strip()


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        normalized_key = _normalize_key(key)
        if not normalized_key:
            continue
        if isinstance(value, str):
            status = value.strip().lower()
            note = ""
        elif isinstance(value, dict):
            status = str(value.get("status", "")).strip().lower()
            note = _normalize_key(value.get("note", ""))
        else:
            continue
        if status not in VALID_STATUSES:
            continue
        normalized[normalized_key] = {"status": status, "note": note}
    return normalized


def extract_override_updates(review_path: Path = REVIEW_PATH) -> dict[str, dict[str, str]]:
    if not review_path.exists():
        return {}

    review_df = pd.read_excel(review_path)
    updates: dict[str, dict[str, str]] = {}

    for _, row in review_df.iterrows():
        manual_status = _normalize_key(row.get("manual_status", "")).lower()
        if manual_status not in VALID_STATUSES:
            continue

        key = _normalize_key(row.get("cleaned_name", "")) or _normalize_key(row.get("raw_name", ""))
        if not key:
            continue

        note = _normalize_key(row.get("manual_note", ""))
        updates[key] = {"status": manual_status, "note": note}

    return updates


def merge_overrides(
    existing: dict[str, dict[str, str]],
    updates: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    merged = {key: dict(value) for key, value in existing.items()}
    for key, value in updates.items():
        merged[_normalize_key(key)] = {
            "status": value["status"],
            "note": _normalize_key(value.get("note", "")),
        }
    return dict(sorted(merged.items(), key=lambda item: item[0].casefold()))


def save_overrides(overrides: dict[str, dict[str, str]], path: Path = OVERRIDES_PATH) -> None:
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sync_guest_validation_overrides(
    review_path: Path = REVIEW_PATH,
    overrides_path: Path = OVERRIDES_PATH,
) -> tuple[int, int]:
    existing = load_overrides(overrides_path)
    updates = extract_override_updates(review_path)
    merged = merge_overrides(existing, updates)
    save_overrides(merged, overrides_path)
    return len(updates), len(merged)


def main() -> None:
    updates_count, total_count = sync_guest_validation_overrides()
    print(f"Applied {updates_count} override updates. Total overrides: {total_count}.")


if __name__ == "__main__":
    main()
