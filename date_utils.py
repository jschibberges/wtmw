from __future__ import annotations

from datetime import date, datetime

import pandas as pd

ANALYSIS_START_DATE = pd.Timestamp("2015-01-01")

_KNOWN_DATE_FORMATS: tuple[str, ...] = (
    "%d.%m.%Y",
    "%d.%m.%y",
    "%Y-%m-%d",
    "%Y/%m/%d",
)


def parse_mixed_date(value: object) -> pd.Timestamp | pd.NaT:
    """Parse common project date formats into a pandas Timestamp."""
    if value is None or value is pd.NaT:
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, date):
        return pd.Timestamp(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return pd.NaT

    for fmt in _KNOWN_DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed)


def coerce_mixed_date_series(series: pd.Series | None) -> pd.Series:
    """Coerce a pandas Series containing mixed date formats."""
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    return series.apply(parse_mixed_date)


def is_on_or_after_analysis_start(value: object) -> bool:
    """Return whether a date value falls within the supported analysis window."""
    parsed = parse_mixed_date(value)
    return pd.notna(parsed) and parsed >= ANALYSIS_START_DATE


def filter_to_analysis_window(
    df: pd.DataFrame | None, date_col: str = "date"
) -> pd.DataFrame:
    """Filter a DataFrame to the supported analysis window and coerce its date column."""
    if df is None:
        return pd.DataFrame()
    if df.empty or date_col not in df.columns:
        return df.copy()

    working = df.copy()
    parsed_dates = coerce_mixed_date_series(working[date_col])
    mask = parsed_dates.notna() & (parsed_dates >= ANALYSIS_START_DATE)
    filtered = working.loc[mask].copy()
    filtered[date_col] = parsed_dates.loc[mask]
    return filtered


def normalize_date_value(value: object) -> object:
    """Return project dates in canonical dd.mm.YYYY string format when possible."""
    parsed = parse_mixed_date(value)
    if pd.notna(parsed):
        return parsed.strftime("%d.%m.%Y")
    return value
