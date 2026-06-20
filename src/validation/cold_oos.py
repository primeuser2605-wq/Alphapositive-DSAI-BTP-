"""
cold_oos.py
===========
Cold out-of-sample evaluation for both ETH and BTC strategies.

The methodological purpose
--------------------------
The original report's L3 limitation: "the full 2020-2023 window was visible
during development for both strategies. A genuinely cold test requires data
not available at design time."

This module IS that test. It:
1. Pulls hourly OHLCV data for BTC and ETH for a date range that POST-DATES
   strategy development (default: 2024-01-01 onward).
2. Runs both strategies with FROZEN parameters — no tuning, no hyperparameter
   exposure in the CLI. The parameters are whatever ETHStrategyConfig() and
   BTCQLearningConfig() default to.
3. Backtests each on the cold data using the in-house backtester.
4. Computes Politis-Romano stationary bootstrap CIs on per-trade returns.
5. Writes a structured report comparing cold-OOS metrics to in-sample
   numbers, with explicit caveats about what comparison means.

The cold-OOS guarantee
----------------------
The frozen-parameter requirement is enforced by NOT exposing any strategy
config to the CLI. To run a different parameter set, the user must edit
the source code — which leaves a paper trail and makes the change visible
in version control. This is the only honest way to do OOS.

Network requirement
-------------------
Pulling data from Binance requires network access and that the running
machine is not geographically blocked from the Binance API. If running
from a blocked region, use the binance-public-data archive on GitHub
instead (see README).

Usage
-----
    # Default: 2024-01-01 to today
    python -m src.validation.cold_oos --output-dir results/cold_oos_2024/

    # Custom range
    python -m src.validation.cold_oos \\
        --start 2024-01-01 --end 2024-06-30 \\
        --output-dir results/cold_oos_2024_h1/

    # Use existing CSVs instead of pulling fresh data
    python -m src.validation.cold_oos \\
        --btc-csv data/BTC_USDT_1h_2024.csv \\
        --eth-csv data/ETH_USDT_1h_2024.csv \\
        --output-dir results/cold_oos_2024/

Output
------
For each strategy, writes:
- {strategy}_trades.csv      — per-trade log
- {strategy}_equity.csv       — equity curve
- {strategy}_metrics.json     — full backtest metrics + bootstrap CIs
- {strategy}_report.txt       — human-readable summary

Plus a top-level cold_oos_summary.txt comparing both strategies.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd

from backtester import run_backtest, BacktestResult
from data_io.loader import load_btc_eth_pair, load_ohlcv_csv
from strategies.eth_regime_confirmation import ETHStrategyConfig, run_eth_strategy
from strategies.btc_qlearning import (
    BTCQLearningConfig, run_btc_qlearning_strategy,
)
from validation.bootstrap import bootstrap_all_metrics, summarize_bootstrap


# =====================================================================
# Data acquisition
# =====================================================================
def acquire_data(
    btc_csv: Optional[str],
    eth_csv: Optional[str],
    start: str,
    end: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load BTC/ETH data either from provided CSVs or by pulling from Binance.

    Parameters
    ----------
    btc_csv, eth_csv : str or None
        Paths to existing OHLCV CSVs. If None, data is pulled fresh.
    start, end : str
        Date range (ISO format).
    cache_dir : str or None
        If pulling fresh data, save the CSVs here for future reuse.

    Returns
    -------
    pd.DataFrame
        BTC/ETH OHLCV merged, indexed by datetime.
    """
    if btc_csv and eth_csv:
        print(f"[cold-oos] Loading BTC from {btc_csv}...")
        print(f"[cold-oos] Loading ETH from {eth_csv}...")
        df = load_btc_eth_pair(btc_csv, eth_csv, align=True)
        # Apply date filter
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return df

    # Pull fresh from Binance
    print(f"[cold-oos] Pulling BTCUSDT hourly from Binance for {start} to {end}...")
    print(f"[cold-oos] (this requires network access and a non-geo-blocked location)")
    from data_io.binance_puller import pull_klines

    btc_df = pull_klines("BTCUSDT", "1h", start, end, verbose=True)
    eth_df = pull_klines("ETHUSDT", "1h", start, end, verbose=True)

    if cache_dir:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        btc_path = cache / f"BTCUSDT_1h_{start}_{end}.csv"
        eth_path = cache / f"ETHUSDT_1h_{start}_{end}.csv"
        btc_df.to_csv(btc_path, index=False)
        eth_df.to_csv(eth_path, index=False)
        print(f"[cold-oos] Cached: {btc_path}, {eth_path}")
        # Re-load via load_btc_eth_pair for consistent column naming
        return load_btc_eth_pair(btc_path, eth_path, align=True)

    # No cache: build the merged frame directly
    btc_pref = btc_df.rename(columns={
        c: f"btc_{c}" for c in ("open", "high", "low", "close", "volume")
    })
    eth_pref = eth_df.rename(columns={
        c: f"eth_{c}" for c in ("open", "high", "low", "close", "volume")
    })
    merged = btc_pref.merge(
        eth_pref.drop(columns=["datetime"]),
        left_index=True, right_index=True, how="inner",
    )
    return merged.sort_index()


# =====================================================================
# Strategy runners
# =====================================================================
def run_eth_on_cold(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run the ETH strategy with FROZEN default config.
    """
    print(f"[cold-oos] Running ETH strategy with frozen parameters...")
    config = ETHStrategyConfig()  # frozen defaults

    signals_df = run_eth_strategy(df.reset_index(drop=True), config=config)
    signals_df = signals_df.rename(columns={
        "eth_open": "open", "eth_high": "high",
        "eth_low": "low", "eth_close": "close",
    })
    result = run_backtest(signals_df, signal_col="signal",
                          initial_capital=1000.0, fee_rate=0.0015)

    # Bootstrap CIs on trade returns
    trades_df = result.trades_df()
    if len(trades_df) >= 10:
        trade_returns = trades_df["return_pct"].values
        cis = bootstrap_all_metrics(
            trade_returns, n_resamples=5000,
            mean_block_length=10.0, confidence_level=0.95, seed=42,
        )
    else:
        cis = None

    # Save artifacts
    trades_df.to_csv(output_dir / "eth_trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "eth_equity.csv")

    payload = {
        "strategy": "eth_regime_confirmation",
        "n_bars": len(df),
        "config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in config.__dict__.items()},
        "metrics": result.metrics,
        "final_equity": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) else 0.0,
        "bootstrap_cis": {
            k: {"lower": ci.lower, "upper": ci.upper,
                "point_estimate": ci.point_estimate}
            for k, ci in cis.items()
        } if cis else None,
    }
    with open(output_dir / "eth_metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(output_dir / "eth_report.txt", "w") as f:
        f.write(result.summary())
        if cis:
            f.write("\n\n")
            f.write(summarize_bootstrap(cis))

    return payload


def run_btc_on_cold(
    df: pd.DataFrame,
    output_dir: Path,
    q_table_path: Optional[str] = None,
) -> dict:
    """
    Run the BTC Q-learning strategy on cold OOS data.

    Two modes:

    1. **True cold OOS** (preferred): pass `q_table_path` pointing to a
       Q-table trained on the original 2020-2023 data. The policy is
       applied to the entire cold window. No retraining occurs on the
       cold data; this is the L3-correct test.

    2. **Fallback retrain mode**: if `q_table_path` is None, the agent is
       retrained on the cold window's first 70% and evaluated on the last
       30%. This is a degraded approximation — it's NOT a true cold test
       because parameters are still being fit on data inside the OOS window.
       The output documents this caveat explicitly.

    To create a Q-table for the preferred mode, train on 2020-2023 once and
    save with `save_q_table(Q, config, 'data/q_table_2020_2023.pkl')`.
    """
    print(f"[cold-oos] Running BTC Q-learning strategy with frozen parameters...")
    config = BTCQLearningConfig()  # frozen defaults

    # Build a 'BTC-only' frame for the strategy
    btc_df = df.rename(columns={
        "btc_open": "open", "btc_high": "high",
        "btc_low": "low", "btc_close": "close",
        "btc_volume": "volume",
    })[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    caveat = None
    q_table_metadata = None

    if q_table_path is not None and Path(q_table_path).exists():
        # ===== TRUE COLD OOS MODE =====
        print(f"[cold-oos] Loading pre-trained Q-table from {q_table_path}")
        from strategies.btc_qlearning import load_q_table, compute_features, policy_to_signals
        Q, loaded_config, q_table_metadata = load_q_table(
            q_table_path, expected_config=config, strict=False,
        )
        print(f"[cold-oos] Q-table loaded: shape={Q.shape}, trained on "
              f"{q_table_metadata.get('saved_at', 'unknown date')}")
        print(f"[cold-oos] Applying frozen policy to entire cold window (TRUE OOS).")

        # Apply the policy to the cold data
        feat = compute_features(btc_df, config)
        feat_clean = feat.dropna(
            subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"]
        ).reset_index(drop=True)
        signals = policy_to_signals(feat_clean, Q, config)
        df_with_signals = feat_clean.copy()
        df_with_signals["signal"] = signals

        # Backtest on the entire cold window
        result = run_backtest(df_with_signals, signal_col="signal",
                               initial_capital=1000.0, fee_rate=0.0015)
        n_train = 0
        n_test = len(df_with_signals)
        mode = "true_cold_oos"
    else:
        # ===== FALLBACK RETRAIN MODE =====
        if q_table_path is not None:
            print(f"[cold-oos] WARNING: Q-table at {q_table_path} not found.")
        print(f"[cold-oos] No pre-trained Q-table provided. Falling back to retrain mode.")
        print(f"[cold-oos] CAVEAT: agent will be retrained on the cold window's first 70%.")
        print(f"[cold-oos]         This is NOT a true cold test. To get a true cold result:")
        print(f"[cold-oos]         (1) train on 2020-2023 once, save with save_q_table()")
        print(f"[cold-oos]         (2) re-run cold_oos with --btc-q-table=path/to/qtable.pkl")

        train_end_idx = int(0.7 * len(btc_df))
        train_end_dt = str(btc_df["datetime"].iloc[train_end_idx])
        df_with_signals, Q = run_btc_qlearning_strategy(
            btc_df, train_end_date=train_end_dt, config=config, verbose=False,
        )
        # Backtest only on the test-portion
        test_df = df_with_signals.iloc[train_end_idx:].reset_index(drop=True)
        result = run_backtest(test_df, signal_col="signal",
                               initial_capital=1000.0, fee_rate=0.0015)
        n_train = train_end_idx
        n_test = len(btc_df) - train_end_idx
        mode = "fallback_retrain"
        caveat = ("BTC agent retrained on cold-window's first 70% (no pre-trained Q-table "
                  "supplied). True OOS would require --btc-q-table=path/to/qtable.pkl.")

    # Bootstrap CIs
    trades_df = result.trades_df()
    if len(trades_df) >= 10:
        trade_returns = trades_df["return_pct"].values
        cis = bootstrap_all_metrics(
            trade_returns, n_resamples=5000,
            mean_block_length=10.0, confidence_level=0.95, seed=42,
        )
    else:
        cis = None

    trades_df.to_csv(output_dir / "btc_trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "btc_equity.csv")

    payload = {
        "strategy": "btc_qlearning",
        "mode": mode,
        "n_bars_total": len(btc_df),
        "n_bars_train": n_train,
        "n_bars_test": n_test,
        "config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in config.__dict__.items()},
        "metrics": result.metrics,
        "final_equity": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) else 0.0,
        "bootstrap_cis": {
            k: {"lower": ci.lower, "upper": ci.upper,
                "point_estimate": ci.point_estimate}
            for k, ci in cis.items()
        } if cis else None,
        "q_table_metadata": q_table_metadata,
        "caveat": caveat,
    }
    with open(output_dir / "btc_metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(output_dir / "btc_report.txt", "w") as f:
        f.write(result.summary())
        if cis:
            f.write("\n\n")
            f.write(summarize_bootstrap(cis))
        f.write(f"\n\nMode: {mode}\n")
        if caveat:
            f.write(f"CAVEAT: {caveat}\n")
        else:
            f.write("This is a true cold OOS evaluation: Q-table was trained "
                    "outside the cold window.\n")

    return payload


# =====================================================================
# Summary writer
# =====================================================================
def write_summary(
    output_dir: Path,
    eth_payload: dict,
    btc_payload: dict,
    start: str,
    end: str,
) -> None:
    """Write the top-level summary comparing cold-OOS to in-sample numbers."""
    lines = []
    lines.append("=" * 88)
    lines.append("COLD OUT-OF-SAMPLE EVALUATION")
    lines.append("=" * 88)
    lines.append(f"Date range: {start} to {end}")
    lines.append(f"Run on: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("This evaluation uses FROZEN strategy parameters (defaults from")
    lines.append("ETHStrategyConfig and BTCQLearningConfig). No tuning was performed.")
    lines.append("")

    # In-sample reference numbers from the original report
    lines.append("REFERENCE (from original 2020-2023 in-sample numbers in report):")
    lines.append("  ETH Sharpe (in-sample, 2020-2023): 5.97")
    lines.append("  ETH Max DD (in-sample, 2020-2023): 17.14%")
    lines.append("  BTC Sharpe (in-sample, 2023 test): 9.15")
    lines.append("  BTC Max DD (in-sample, 2023 test): 13.50%")
    lines.append("")

    # ETH cold-OOS
    lines.append("ETH/USDT (cold OOS):")
    m = eth_payload["metrics"]
    lines.append(f"  Total trades:    {m.get('total_trades', 0)}")
    lines.append(f"  Sharpe ratio:    {m.get('sharpe', 0):+.3f}")
    lines.append(f"  Total return:    {m.get('total_return', 0) * 100:+.2f}%")
    lines.append(f"  Max drawdown:    {m.get('max_drawdown', 0) * 100:+.2f}%")
    lines.append(f"  Win rate:        {m.get('win_rate', 0) * 100:.1f}%")
    if eth_payload.get("bootstrap_cis"):
        ci = eth_payload["bootstrap_cis"].get("sharpe")
        if ci:
            lines.append(f"  Sharpe 95% CI:   [{ci['lower']:+.3f}, {ci['upper']:+.3f}]")
    lines.append("")

    # BTC cold-OOS
    lines.append("BTC/USDT (cold OOS):")
    m = btc_payload["metrics"]
    lines.append(f"  Total trades:    {m.get('total_trades', 0)}")
    lines.append(f"  Sharpe ratio:    {m.get('sharpe', 0):+.3f}")
    lines.append(f"  Total return:    {m.get('total_return', 0) * 100:+.2f}%")
    lines.append(f"  Max drawdown:    {m.get('max_drawdown', 0) * 100:+.2f}%")
    lines.append(f"  Win rate:        {m.get('win_rate', 0) * 100:.1f}%")
    if btc_payload.get("bootstrap_cis"):
        ci = btc_payload["bootstrap_cis"].get("sharpe")
        if ci:
            lines.append(f"  Sharpe 95% CI:   [{ci['lower']:+.3f}, {ci['upper']:+.3f}]")
    btc_mode = btc_payload.get("mode", "unknown")
    if btc_mode == "true_cold_oos":
        lines.append(f"  Mode: TRUE COLD OOS (pre-trained Q-table used)")
    else:
        caveat = btc_payload.get("caveat") or ""
        if caveat:
            lines.append(f"  CAVEAT: {caveat}")
    lines.append("")

    # Interpretation
    lines.append("INTERPRETATION GUIDE:")
    lines.append("  - If cold-OOS Sharpe LB > 0:  evidence of real edge OOS")
    lines.append("  - If cold-OOS Sharpe ≈ 0 or CI brackets 0:  in-sample edge was overfit")
    lines.append("  - If cold-OOS Sharpe < 0:    strategy actively loses money OOS")
    lines.append("")
    lines.append("Both strategies should be considered 'validated' only if their")
    lines.append("cold-OOS Sharpe lower bound (bootstrap CI) exceeds zero.")
    lines.append("=" * 88)

    out_path = output_dir / "cold_oos_summary.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print()
    print("\n".join(lines))
    print(f"\nFull summary saved to: {out_path}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Cold out-of-sample evaluation for BTC and ETH strategies.",
        epilog="No strategy parameters are exposed by design (parameter freeze).",
    )
    parser.add_argument("--start", default="2024-01-01",
                        help="Start date (ISO). Default: 2024-01-01")
    parser.add_argument("--end", default=date.today().isoformat(),
                        help="End date (ISO). Default: today")
    parser.add_argument("--btc-csv", default=None,
                        help="Path to existing BTC OHLCV CSV (skip Binance pull)")
    parser.add_argument("--eth-csv", default=None,
                        help="Path to existing ETH OHLCV CSV (skip Binance pull)")
    parser.add_argument("--output-dir", default="results/cold_oos/",
                        help="Where to write trade logs, equity curves, JSON, and summary.")
    parser.add_argument("--cache-dir", default=None,
                        help="If pulling fresh data, save CSVs here for reuse.")
    parser.add_argument("--strategy", choices=["eth", "btc", "both"], default="both",
                        help="Which strategy/strategies to evaluate.")
    parser.add_argument("--btc-q-table", default=None,
                        help="Path to a pre-trained Q-table (saved via save_q_table). "
                             "If provided, the BTC agent runs in true-cold-OOS mode "
                             "(no retraining on cold data). Otherwise falls back to "
                             "internal 70/30 split with a documented caveat.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cold-oos] Cold OOS evaluation")
    print(f"[cold-oos] Date range: {args.start} to {args.end}")
    print(f"[cold-oos] Output: {output_dir}")
    print()

    df = acquire_data(args.btc_csv, args.eth_csv, args.start, args.end,
                       cache_dir=args.cache_dir)
    print(f"[cold-oos] Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print()

    eth_payload = None
    btc_payload = None
    if args.strategy in ("eth", "both"):
        eth_payload = run_eth_on_cold(df, output_dir)
    if args.strategy in ("btc", "both"):
        btc_payload = run_btc_on_cold(df, output_dir, q_table_path=args.btc_q_table)

    if eth_payload and btc_payload:
        write_summary(output_dir, eth_payload, btc_payload, args.start, args.end)


if __name__ == "__main__":
    main()
