"""
loader.py
=========
CSV loader for OHLCV cryptocurrency data.

Handles common quirks of real-world crypto CSV files:
- Different timestamp column names (datetime, timestamp, time, date, open_time)
- Unix epoch (seconds OR milliseconds) vs ISO strings
- Different column casing (Open vs open vs OPEN)
- Optional volume column
- Sorting and deduplication
- Gap detection (returns warnings, not errors)

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


REQUIRED_COLS = ["open", "high", "low", "close"]
OPTIONAL_COLS = ["volume"]

# Common aliases for the timestamp column
TIMESTAMP_ALIASES = [
    "datetime", "timestamp", "time", "date",
    "open_time", "opentime", "Open Time", "Open time",
    "openTime", "ts", "t",
]


def _find_timestamp_column(columns: list[str]) -> Optional[str]:
    """Find the timestamp column by case-insensitive alias match."""
    lower_map = {c.lower(): c for c in columns}
    for alias in TIMESTAMP_ALIASES:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names; strip whitespace."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Convert a timestamp series to pandas datetime, handling multiple formats."""
    # Try numeric (unix epoch)
    if pd.api.types.is_numeric_dtype(series):
        # Detect seconds vs milliseconds: timestamps > 10^11 are ms (post-1973 in ms)
        sample = series.dropna().iloc[0] if len(series.dropna()) > 0 else 0
        unit = "ms" if sample > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, utc=True).dt.tz_localize(None)
    # Else assume string
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def load_ohlcv_csv(
    path: Union[str, Path],
    timestamp_col: Optional[str] = None,
    expected_freq: str = "1h",
    warn_on_gaps: bool = True,
) -> pd.DataFrame:
    """
    Load an OHLCV CSV file with robust schema handling.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    timestamp_col : str, optional
        Explicit timestamp column name. If None, auto-detects from common aliases.
    expected_freq : str
        Expected frequency for gap detection ('1h', '15min', '1d', etc).
        Set to None to skip gap checking.
    warn_on_gaps : bool
        Emit a warning if data has gaps relative to expected_freq.

    Returns
    -------
    pd.DataFrame
        Sorted by datetime ascending, with columns:
        datetime (index), open, high, low, close, volume (if present).
        Also exposes a 'datetime' column for compatibility with strategies
        that expect it as a regular column.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If required OHLC columns are missing or unparseable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_column_names(df)

    # Find and parse timestamp
    if timestamp_col is None:
        timestamp_col = _find_timestamp_column(list(df.columns))
        if timestamp_col is None:
            raise ValueError(
                f"Could not detect timestamp column. Columns present: {list(df.columns)}. "
                f"Pass timestamp_col= explicitly."
            )
    timestamp_col = timestamp_col.lower()
    if timestamp_col not in df.columns:
        raise ValueError(f"Specified timestamp_col '{timestamp_col}' not in columns: {list(df.columns)}")

    df["datetime"] = _parse_timestamp(df[timestamp_col])

    # Verify OHLC presence
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}. Found: {list(df.columns)}")

    # Coerce to numeric (handle string "1234.56" cases)
    for c in REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop bars missing essential data
    n_before = len(df)
    df = df.dropna(subset=REQUIRED_COLS + ["datetime"])
    if n_before - len(df) > 0:
        warnings.warn(f"Dropped {n_before - len(df)} bars with NaN OHLC values.")

    # Sort, deduplicate
    df = df.sort_values("datetime").reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)
    if n_before - len(df) > 0:
        warnings.warn(f"Dropped {n_before - len(df)} duplicate timestamps.")

    # Index by datetime but also keep as column
    df = df.set_index("datetime", drop=False)
    df.index.name = "datetime_idx"

    # Gap check
    if warn_on_gaps and expected_freq is not None and len(df) >= 2:
        expected_delta = pd.Timedelta(expected_freq)
        deltas = df["datetime"].diff().dropna()
        gaps = deltas[deltas > expected_delta * 1.5]
        if len(gaps) > 0:
            n_gaps = len(gaps)
            max_gap = gaps.max()
            warnings.warn(
                f"Detected {n_gaps} gaps in data (max gap: {max_gap}). "
                f"Expected freq: {expected_freq}. Strategies assuming "
                f"continuous data may behave unexpectedly."
            )

    # Reorder columns: datetime first, then OHLCV
    keep_cols = ["datetime"] + REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]
    df = df[keep_cols]

    return df


def load_btc_eth_pair(
    btc_path: Union[str, Path],
    eth_path: Union[str, Path],
    align: bool = True,
) -> pd.DataFrame:
    """
    Load BTC and ETH OHLCV CSVs and merge them into a single DataFrame.

    Output column naming follows the ETH strategy's convention:
    'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume',
    'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume', 'datetime'.

    Parameters
    ----------
    btc_path, eth_path : str or Path
        Paths to BTC and ETH OHLCV CSVs.
    align : bool
        If True (default), inner-join on datetime so only bars present in
        both series are kept. If False, full outer join.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame, indexed by datetime, sorted ascending.
    """
    btc = load_ohlcv_csv(btc_path)
    eth = load_ohlcv_csv(eth_path)

    btc_pref = btc.rename(columns={
        c: f"btc_{c}" for c in REQUIRED_COLS + OPTIONAL_COLS if c in btc.columns
    })
    eth_pref = eth.rename(columns={
        c: f"eth_{c}" for c in REQUIRED_COLS + OPTIONAL_COLS if c in eth.columns
    })

    how = "inner" if align else "outer"
    merged = btc_pref.merge(
        eth_pref.drop(columns=["datetime"]),
        left_index=True, right_index=True, how=how,
    )

    n_btc, n_eth, n_merged = len(btc), len(eth), len(merged)
    if align and (n_btc != n_merged or n_eth != n_merged):
        warnings.warn(
            f"Misaligned bars: BTC={n_btc}, ETH={n_eth}, after inner-join={n_merged}. "
            f"Dropped {max(n_btc, n_eth) - n_merged} non-overlapping bars."
        )

    return merged.sort_index()


def slice_by_date(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Slice a DataFrame to a date range (inclusive on both ends)."""
    out = df
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out
