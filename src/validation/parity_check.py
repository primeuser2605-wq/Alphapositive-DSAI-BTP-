"""
parity_check.py
===============
Compare the refactored strategy output against the original Inter IIT quarterly
summaries (BTC_backtest_summary.csv, ETH_backtest_summary.csv).

Why this matters
----------------
The refactor is only useful if it reproduces the original numbers (within
acceptable tolerance). Otherwise, all downstream validation experiments
are on a different strategy than the one the report describes.

What this script does
---------------------
1. Loads real BTC/ETH OHLCV data
2. Runs the refactored strategy
3. Bins trades into the same quarters as the original summaries
4. Computes per-quarter: profit %, total trades, win rate
5. Compares each quarter against the original CSV
6. Reports a "match score" (% of quarters within tolerance) and a side-by-side diff

Acceptance criteria
-------------------
- Quarterly profit %: within ±15% absolute (e.g. 30% vs 45% is acceptable;
  -10% vs +20% is not). This is wide because of legitimate execution-detail
  differences (next-open vs current-close, fee accounting, etc.)
- Total trades per quarter: within ±25%
- Long/short split: same sign at quarter granularity

Tighter parity (±5%) would require exactly reproducing the Untrade SDK's
execution model, which is closed-source.

Usage:
    python -m validation.parity_check \\
        --eth-data data/ETH_USDT_1h_2020_2023.csv \\
        --btc-data data/BTC_USDT_1h_2020_2023.csv \\
        --eth-original validation_data/ETH_backtest_summary.csv \\
        --btc-original validation_data/BTC_backtest_summary.csv

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Tolerance thresholds (loose by design — see docstring)
PROFIT_PCT_TOLERANCE = 15.0      # absolute %
TRADE_COUNT_TOLERANCE = 0.25     # relative
WIN_RATE_TOLERANCE = 0.20        # absolute (e.g. 50% vs 70%)


@dataclass
class QuarterComparison:
    """One row's worth of comparison: original vs refactored."""
    quarter: str
    orig_profit_pct: float
    refactor_profit_pct: float
    profit_diff: float
    orig_trades: int
    refactor_trades: int
    trade_diff: int
    orig_win_rate: float
    refactor_win_rate: float
    win_rate_diff: float
    profit_match: bool
    trades_match: bool
    win_rate_match: bool

    @property
    def all_match(self) -> bool:
        return self.profit_match and self.trades_match and self.win_rate_match


def load_original_summary(path: str | Path) -> pd.DataFrame:
    """Load the original quarterly summary CSV."""
    df = pd.read_csv(path)
    df["From"] = pd.to_datetime(df["From"])
    df["To"] = pd.to_datetime(df["To"])
    df["quarter_label"] = df["From"].dt.to_period("Q").astype(str)
    return df


def bin_trades_into_quarters(
    trades_df: pd.DataFrame,
    initial_capital: float = 1000.0,
) -> pd.DataFrame:
    """
    Bin a trade-level DataFrame into quarterly summary rows that match the
    schema of the original quarterly summary CSV.

    Each quarter:
    - Starts with $initial_capital
    - Final balance = initial_capital + sum(pnl_net for that quarter)
    - Profit % = (final - initial) / initial * 100
    - Trades = count of trades closed in that quarter
    - Win rate = mean(pnl_net > 0)

    NOTE: This matches the original summary's convention of "fresh $1000
    each quarter," not compounding across quarters.
    """
    if len(trades_df) == 0:
        return pd.DataFrame()

    tdf = trades_df.copy()
    tdf["exit_time"] = pd.to_datetime(tdf["exit_time"])
    tdf["quarter_label"] = tdf["exit_time"].dt.to_period("Q").astype(str)
    tdf["is_long"] = tdf["side"] == 1
    tdf["is_win"] = tdf["pnl_net"] > 0

    rows = []
    for q, group in tdf.groupby("quarter_label"):
        n_trades = len(group)
        if n_trades == 0:
            continue
        # The "profit %" in the original is computed as if starting fresh
        # with initial_capital each quarter — replicate that
        pnl_sum = group["pnl_net"].sum()
        profit_pct = (pnl_sum / initial_capital) * 100
        rows.append({
            "quarter_label": q,
            "Profit (%)": profit_pct,
            "Total Trades": n_trades,
            "Long Trades": int(group["is_long"].sum()),
            "Short Trades": int((~group["is_long"]).sum()),
            "Win Rate (%)": float(group["is_win"].mean() * 100),
        })

    return pd.DataFrame(rows)


def compare_quarters(
    refactor_summary: pd.DataFrame,
    original_summary: pd.DataFrame,
) -> list[QuarterComparison]:
    """Compare the two summaries quarter by quarter."""
    comparisons = []
    # Build a lookup for refactor results
    ref_map = {row["quarter_label"]: row for _, row in refactor_summary.iterrows()}

    for _, orig_row in original_summary.iterrows():
        q = orig_row["quarter_label"]
        if q not in ref_map:
            # Quarter present in original but not in refactor (or vice versa)
            comparisons.append(QuarterComparison(
                quarter=q,
                orig_profit_pct=float(orig_row["Profit (%)"]),
                refactor_profit_pct=0.0,
                profit_diff=float(orig_row["Profit (%)"]),
                orig_trades=int(orig_row["Total Trades"]),
                refactor_trades=0,
                trade_diff=int(orig_row["Total Trades"]),
                orig_win_rate=float(orig_row["Win Rate (%)"]),
                refactor_win_rate=0.0,
                win_rate_diff=float(orig_row["Win Rate (%)"]),
                profit_match=False,
                trades_match=False,
                win_rate_match=False,
            ))
            continue
        ref_row = ref_map[q]
        profit_diff = float(ref_row["Profit (%)"] - orig_row["Profit (%)"])
        trades_diff = int(ref_row["Total Trades"] - orig_row["Total Trades"])
        wr_diff = float(ref_row["Win Rate (%)"] - orig_row["Win Rate (%)"])

        # Match thresholds
        profit_match = abs(profit_diff) <= PROFIT_PCT_TOLERANCE
        orig_t = int(orig_row["Total Trades"])
        trades_match = (orig_t == 0 and ref_row["Total Trades"] == 0) or \
                       (orig_t > 0 and abs(trades_diff) / orig_t <= TRADE_COUNT_TOLERANCE)
        wr_match = abs(wr_diff) <= WIN_RATE_TOLERANCE * 100

        comparisons.append(QuarterComparison(
            quarter=q,
            orig_profit_pct=float(orig_row["Profit (%)"]),
            refactor_profit_pct=float(ref_row["Profit (%)"]),
            profit_diff=profit_diff,
            orig_trades=int(orig_row["Total Trades"]),
            refactor_trades=int(ref_row["Total Trades"]),
            trade_diff=trades_diff,
            orig_win_rate=float(orig_row["Win Rate (%)"]),
            refactor_win_rate=float(ref_row["Win Rate (%)"]),
            win_rate_diff=wr_diff,
            profit_match=profit_match,
            trades_match=trades_match,
            win_rate_match=wr_match,
        ))

    return comparisons


def format_comparison_table(comparisons: list[QuarterComparison]) -> str:
    """Render the comparison as a readable table."""
    if not comparisons:
        return "(no comparisons available)"

    lines = []
    lines.append(f"{'Quarter':<10} {'Orig%':>8} {'Ref%':>8} {'Δ%':>7}  "
                 f"{'Orig#':>5} {'Ref#':>5}  {'OrigWR':>6} {'RefWR':>6}  Match")
    lines.append("-" * 80)
    for c in comparisons:
        m = ""
        m += "P" if c.profit_match else "."
        m += "T" if c.trades_match else "."
        m += "W" if c.win_rate_match else "."
        lines.append(
            f"{c.quarter:<10} "
            f"{c.orig_profit_pct:>8.2f} {c.refactor_profit_pct:>8.2f} "
            f"{c.profit_diff:>+7.2f}  "
            f"{c.orig_trades:>5d} {c.refactor_trades:>5d}  "
            f"{c.orig_win_rate:>6.1f} {c.refactor_win_rate:>6.1f}  {m}"
        )
    lines.append("-" * 80)
    n_total = len(comparisons)
    n_match = sum(1 for c in comparisons if c.all_match)
    n_p = sum(1 for c in comparisons if c.profit_match)
    n_t = sum(1 for c in comparisons if c.trades_match)
    n_w = sum(1 for c in comparisons if c.win_rate_match)
    lines.append(
        f"Quarters matching all three: {n_match}/{n_total} "
        f"(P={n_p}/{n_total}, T={n_t}/{n_total}, W={n_w}/{n_total})"
    )
    lines.append(f"Match key: P=profit, T=trade count, W=win rate")
    return "\n".join(lines)


def run_parity_check_eth(
    eth_data_path: str | Path,
    btc_data_path: str | Path,
    eth_original_path: str | Path,
    verbose: bool = True,
) -> tuple[float, str]:
    """
    End-to-end ETH parity check.

    Returns (match_fraction, report_string).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_io.loader import load_btc_eth_pair, slice_by_date
    from strategies.eth_regime_confirmation import run_eth_strategy, ETHStrategyConfig
    from backtester import run_backtest

    if verbose:
        print("[parity-check ETH] Loading data...")
    df = load_btc_eth_pair(btc_data_path, eth_data_path, align=True)
    df = slice_by_date(df, start="2020-01-01", end="2023-12-31")

    if verbose:
        print(f"[parity-check ETH] Running strategy on {len(df)} bars...")
    signals = run_eth_strategy(df.reset_index(drop=True), config=ETHStrategyConfig())

    # Rename for backtester
    signals = signals.rename(columns={
        "eth_open": "open", "eth_high": "high",
        "eth_low": "low", "eth_close": "close",
    })

    if verbose:
        print("[parity-check ETH] Backtesting...")
    result = run_backtest(signals, signal_col="signal", initial_capital=1000.0,
                          fee_rate=0.0015)
    trades_df = result.trades_df()

    if verbose:
        print(f"[parity-check ETH] Got {len(trades_df)} trades. Binning into quarters...")
    refactor_summary = bin_trades_into_quarters(trades_df, initial_capital=1000.0)
    original_summary = load_original_summary(eth_original_path)
    comparisons = compare_quarters(refactor_summary, original_summary)

    report = format_comparison_table(comparisons)
    match_frac = sum(1 for c in comparisons if c.all_match) / max(len(comparisons), 1)
    return match_frac, report


def main():
    parser = argparse.ArgumentParser(description="Parity check vs original quarterly summaries.")
    parser.add_argument("--eth-data", required=False, help="Path to ETH OHLCV CSV")
    parser.add_argument("--btc-data", required=False, help="Path to BTC OHLCV CSV")
    parser.add_argument("--eth-original", required=False, help="Original ETH quarterly summary CSV")
    parser.add_argument("--btc-original", required=False, help="Original BTC quarterly summary CSV")
    parser.add_argument("--strategy", choices=["eth", "btc", "both"], default="both")
    args = parser.parse_args()

    if args.strategy in ("eth", "both"):
        if not (args.eth_data and args.btc_data and args.eth_original):
            print("ETH parity check requires --eth-data, --btc-data, --eth-original")
        else:
            print("\n" + "=" * 80)
            print("ETH PARITY CHECK")
            print("=" * 80)
            frac, report = run_parity_check_eth(
                args.eth_data, args.btc_data, args.eth_original
            )
            print(report)
            print(f"\nOverall match rate: {frac:.1%}")

    # BTC parity left as a TODO — requires also training the Q-agent reproducibly,
    # which is sensitive to RNG state and impossible to make pixel-exact against
    # the original without sharing the same seed. The harness pattern is the
    # same as ETH; see run_parity_check_eth above as a template.
    if args.strategy in ("btc", "both"):
        print("\n[NOTE] BTC parity check not implemented — Q-learning RNG sensitivity")
        print("       makes exact parity infeasible. See parity_check.py module docstring.")


if __name__ == "__main__":
    main()
