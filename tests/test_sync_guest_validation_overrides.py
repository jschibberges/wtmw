from __future__ import annotations

import json

import pandas as pd

import sync_guest_validation_overrides as sync_mod


def test_extract_override_updates_uses_cleaned_name_and_filters_statuses(tmp_path):
    review_path = tmp_path / "guest_validation_review.xlsx"
    pd.DataFrame(
        [
            {
                "cleaned_name": "Detlef D! Soost",
                "raw_name": "Detlef D! Soost",
                "manual_status": "accept",
                "manual_note": "Künstlername",
            },
            {
                "cleaned_name": "",
                "raw_name": "„Süddeutsche Zeitung“",
                "manual_status": "reject",
                "manual_note": "Organisation",
            },
            {
                "cleaned_name": "Ignore Me",
                "raw_name": "Ignore Me",
                "manual_status": "maybe",
                "manual_note": "invalid",
            },
        ]
    ).to_excel(review_path, index=False)

    updates = sync_mod.extract_override_updates(review_path)

    assert updates == {
        "Detlef D! Soost": {"status": "accept", "note": "Künstlername"},
        "„Süddeutsche Zeitung“": {"status": "reject", "note": "Organisation"},
    }


def test_sync_guest_validation_overrides_merges_and_overwrites_existing_entries(tmp_path):
    overrides_path = tmp_path / "guest_validation_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "Detlef D! Soost": {"status": "reject", "note": "old"},
                "Bestehend": {"status": "accept", "note": "keep"},
            }
        ),
        encoding="utf-8",
    )
    review_path = tmp_path / "guest_validation_review.xlsx"
    pd.DataFrame(
        [
            {
                "cleaned_name": "Detlef D! Soost",
                "raw_name": "Detlef D! Soost",
                "manual_status": "accept",
                "manual_note": "Künstlername",
            },
            {
                "cleaned_name": "Familie Pinzler",
                "raw_name": "Familie Pinzler",
                "manual_status": "reject",
                "manual_note": "Gruppe",
            },
        ]
    ).to_excel(review_path, index=False)

    updates_count, total_count = sync_mod.sync_guest_validation_overrides(
        review_path=review_path,
        overrides_path=overrides_path,
    )

    saved = json.loads(overrides_path.read_text(encoding="utf-8"))
    assert updates_count == 2
    assert total_count == 3
    assert saved == {
        "Bestehend": {"status": "accept", "note": "keep"},
        "Detlef D! Soost": {"status": "accept", "note": "Künstlername"},
        "Familie Pinzler": {"status": "reject", "note": "Gruppe"},
    }
