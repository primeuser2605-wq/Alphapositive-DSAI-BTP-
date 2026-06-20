"""
binance_puller.py
=================
Pull historical hourly OHLCV data from Binance's public REST API.

No API key required. Uses the public klines endpoint, which has a hard
limit of 1000 bars per request (~41 days at 1h frequency). This script
paginates automatically.

Usage:
    python -m data_io.binance_puller \\
        --symbol BTCUSDT --interval 1h \\
        --start 2024-01-01 --end 2024-12-31 \\
        --output data/BTC_USDT_1h_2024.csv

Or from Python:
    from data_io.binance_puller import pull_klines
    df = pull_klines("BTCUSDT", "1h", "2024-01-01", "2024-12-31")
    df.to_csv("data/BTC_USDT_1h_2024.csv")

Rate limits: Binance allows ~1200 requests/minute on this endpoint.
We add a small delay between requests to stay polite.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import argparse
import time
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd


# Binance public klines endpoint. Spot API (no auth needed for klines).
KLINES_URL = "https://api.binance.com/api/v3/klines"
ALT_URL = "https://data-api.binance.vision/api/v3/klines"  # fallback

MAX_LIMIT = 1000  # Binance hard limit


def _ts_ms(dt) -> int:
    """Convert any date-like to milliseconds since unix epoch."""
    return int(pd.Timestamp(dt).timestamp() * 1000)


def _interval_to_ms(interval: str) -> int:
    """Convert Binance interval string to milliseconds."""
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    n = int(interval[:-1])
    unit = interval[-1]
    return n * units[unit]


def _fetch_chunk(
    symbol: str, interval: str,
    start_ms: int, end_ms: int,
    use_alt: bool = False,
) -> list:
    """Fetch one chunk of up to MAX_LIMIT klines."""
    import requests  # lazy import — only required when actually pulling
    url = ALT_URL if use_alt else KLINES_URL
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_LIMIT,
    }
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 451 or resp.status_code == 403:
        # Geographic block; try the alt endpoint
        if not use_alt:
            return _fetch_chunk(symbol, interval, start_ms, end_ms, use_alt=True)
        raise RuntimeError(
            f"Binance returned {resp.status_code} — likely geographic restriction. "
            f"Try running from a different location or use a proxy."
        )
    resp.raise_for_status()
    return resp.json()


def pull_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    delay_seconds: float = 0.1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pull historical klines, paginating as needed.

    Parameters
    ----------
    symbol : str
        Trading pair like 'BTCUSDT' or 'ETHUSDT'.
    interval : str
        Bar frequency: '1m', '5m', '15m', '1h', '4h', '1d', etc.
    start, end : str
        Date strings (ISO format preferred): '2024-01-01' or '2024-01-01 00:00:00'.
    delay_seconds : float
        Pause between requests to avoid rate-limit warnings.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: datetime (index), open, high, low, close, volume.
    """
    start_ms = _ts_ms(start)
    end_ms = _ts_ms(end)
    bar_ms = _interval_to_ms(interval)
    chunk_ms = bar_ms * MAX_LIMIT

    all_rows: list = []
    cursor = start_ms

    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms - 1, end_ms)
        if verbose:
            cur_iso = pd.Timestamp(cursor, unit="ms").strftime("%Y-%m-%d")
            print(f"  Fetching {symbol} {interval} from {cur_iso} ...")
        rows = _fetch_chunk(symbol, interval, cursor, chunk_end)
        if not rows:
            # No data returned — advance the cursor anyway
            cursor += chunk_ms
            continue
        all_rows.extend(rows)
        # Advance cursor past the last bar we got
        last_open_ms = rows[-1][0]
        cursor = last_open_ms + bar_ms
        time.sleep(delay_seconds)

    if not all_rows:
        raise RuntimeError(f"No data returned for {symbol} {interval} {start} to {end}")

    # Binance klines schema (12 fields):
    # [open_time, open, high, low, close, volume, close_time, quote_vol,
    #  n_trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime", drop=False)
    df.index.name = "datetime_idx"

    if verbose:
        print(f"  Fetched {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}.")

    return df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Pull Binance OHLCV data.")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--interval", default="1h", help="e.g. 1h, 15m, 1d")
    parser.add_argument("--start", required=True, help="e.g. 2024-01-01")
    parser.add_argument("--end", required=True, help="e.g. 2024-12-31")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between requests")
    args = parser.parse_args()

    print(f"Pulling {args.symbol} {args.interval} from {args.start} to {args.end}...")
    df = pull_klines(args.symbol, args.interval, args.start, args.end,
                     delay_seconds=args.delay, verbose=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} bars to {out}")


if __name__ == "__main__":
    main()
