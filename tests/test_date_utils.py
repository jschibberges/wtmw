from __future__ import annotations

import pandas as pd

from date_utils import (
    ANALYSIS_START_DATE,
    filter_to_analysis_window,
    is_on_or_after_analysis_start,
)


def test_is_on_or_after_analysis_start_respects_2015_cutoff():
    assert not is_on_or_after_analysis_start("31.12.2014")
    assert is_on_or_after_analysis_start("01.01.2015")
    assert is_on_or_after_analysis_start("2024-01-15")


def test_filter_to_analysis_window_drops_older_rows_and_coerces_dates():
    df = pd.DataFrame(
        {
            "uid": ["old", "new"],
            "date": ["31.12.2014", "01.09.24"],
            "value": [1, 2],
        }
    )

    filtered = filter_to_analysis_window(df)

    assert filtered["uid"].tolist() == ["new"]
    assert filtered["date"].tolist() == [pd.Timestamp("2024-09-01")]
    assert ANALYSIS_START_DATE == pd.Timestamp("2015-01-01")
