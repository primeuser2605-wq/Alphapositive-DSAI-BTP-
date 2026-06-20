"""
hurst_ablation.py
=================
Ablation experiment over Hurst exponent computation: window length and method.

The methodological question
---------------------------
The original report's L4 limitation states:
  "Hurst exponent computed on only 120 observations is short relative to the
   n > 1000 typically recommended for R/S stability."

This is testable. The ablation runs the strategy over a grid of (window, method)
combinations:
- Windows: 120 (original), 250, 500, 1000
- Methods: R/S (original) vs DFA (the proposed replacement)

For each combination, two things are measured:
1. Stability of the Hurst estimate itself (mean, std, NaN fraction)
2. Downstream impact on strategy metrics (Sharpe, return, MDD, trade count)

The two measurements answer different questions:
- Hurst stability: is the estimator producing sensible values, or is it noisy?
- Downstream impact: does the strategy actually care?

The downstream answer may be small if the Hurst gate is non-binding (which
the gate ablation experiment suggested for synthetic data). In that case,
the stability comparison is the real finding.

Output
------
HurstAblationResult: per-config metrics + per-config Hurst series statistics.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from itertools import product

import numpy as np
import pandas as pd

from backtester import run_backtest
from strategies.eth_regime_confirmation import (
    ETHStrategyConfig, run_eth_strategy,
)
from indicators import rolling_hurst, rolling_dfa_hurst


# =====================================================================
# Result containers
# =====================================================================
@dataclass
class HurstConfigResult:
    """Results for a single (window, method) configuration."""
    window: int
    method: str  # 'rs' or 'dfa'

    # Hurst-series statistics (computed on ETH close prices)
    hurst_mean: float
    hurst_std: float
    hurst_min: float
    hurst_max: float
    hurst_nan_fraction: float
    hurst_above_05_fraction: float  # fraction of bars where gate would pass

    # Strategy metrics
    total_trades: int
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    win_rate: float


@dataclass
class HurstAblationResult:
    """Full ablation experiment output."""
    results: List[HurstConfigResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame for analysis."""
        rows = []
        for r in self.results:
            rows.append({
                "window": r.window,
                "method": r.method,
                "hurst_mean": r.hurst_mean,
                "hurst_std": r.hurst_std,
                "hurst_min": r.hurst_min,
                "hurst_max": r.hurst_max,
                "hurst_nan_frac": r.hurst_nan_fraction,
                "hurst_above_05_frac": r.hurst_above_05_fraction,
                "trades": r.total_trades,
                "sharpe": r.sharpe,
                "sortino": r.sortino,
                "return": r.total_return,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
            })
        return pd.DataFrame(rows)

    def baseline(self) -> HurstConfigResult:
        """Return the (120, R/S) configuration that matches the original strategy."""
        for r in self.results:
            if r.window == 120 and r.method == "rs":
                return r
        # Fallback to first
        return self.results[0] if self.results else None

    def report(self) -> str:
        lines = []
        lines.append("=" * 96)
        lines.append("HURST ABLATION: WINDOW × METHOD")
        lines.append("=" * 96)
        lines.append("")
        lines.append("HURST ESTIMATOR STATISTICS (on the underlying ETH close series):")
        lines.append(
            f"{'Window':>7} {'Method':>7}  {'Mean':>8} {'Std':>8} {'Min':>8} "
            f"{'Max':>8} {'NaN%':>7} {'>0.5%':>7}"
        )
        lines.append("-" * 96)
        for r in self.results:
            lines.append(
                f"{r.window:>7d} {r.method:>7s}  "
                f"{r.hurst_mean:>+8.3f} {r.hurst_std:>8.3f} "
                f"{r.hurst_min:>+8.3f} {r.hurst_max:>+8.3f} "
                f"{r.hurst_nan_fraction * 100:>6.1f}% "
                f"{r.hurst_above_05_fraction * 100:>6.1f}%"
            )
        lines.append("")
        lines.append("DOWNSTREAM STRATEGY METRICS:")
        lines.append(
            f"{'Window':>7} {'Method':>7}  {'Trades':>6} {'Sharpe':>8} "
            f"{'Return':>9} {'MaxDD':>9} {'WinR':>6}"
        )
        lines.append("-" * 96)
        for r in self.results:
            lines.append(
                f"{r.window:>7d} {r.method:>7s}  "
                f"{r.total_trades:>6d} {r.sharpe:>+8.3f} "
                f"{r.total_return * 100:>+8.2f}% "
                f"{r.max_drawdown * 100:>+8.2f}% "
                f"{r.win_rate * 100:>5.1f}%"
            )
        lines.append("")

        # Interpretation block
        baseline = self.baseline()
        if baseline is not None:
            lines.append("DELTAS vs BASELINE (window=120, method=R/S):")
            lines.append(
                f"{'Window':>7} {'Method':>7}  {'ΔSharpe':>10} {'ΔReturn':>10} "
                f"{'ΔTrades':>9} {'ΔHurst_std':>12}"
            )
            lines.append("-" * 96)
            for r in self.results:
                if r is baseline:
                    continue
                lines.append(
                    f"{r.window:>7d} {r.method:>7s}  "
                    f"{r.sharpe - baseline.sharpe:>+10.3f} "
                    f"{(r.total_return - baseline.total_return) * 100:>+9.2f}% "
                    f"{r.total_trades - baseline.total_trades:>+9d} "
                    f"{r.hurst_std - baseline.hurst_std:>+12.4f}"
                )

        # Verdict heuristic
        lines.append("")
        lines.append("=" * 96)
        df = self.to_dataframe()
        sharpe_range = float(df["sharpe"].max() - df["sharpe"].min())
        hurst_std_max = float(df["hurst_std"].max())
        hurst_std_min = float(df["hurst_std"].min())

        if sharpe_range < 0.15:
            verdict = (
                f"VERDICT: Downstream impact is MINIMAL (Sharpe range = {sharpe_range:.3f}). "
                f"The Hurst gate appears non-binding on this data — gate ablation "
                f"corroborates this. Hurst window/method choice is not a load-bearing decision."
            )
        else:
            verdict = (
                f"VERDICT: Downstream impact is meaningful (Sharpe range = {sharpe_range:.3f}). "
                f"Hurst configuration affects strategy performance."
            )
        lines.append(verdict)
        if hurst_std_max > 0:
            ratio = hurst_std_min / hurst_std_max
            lines.append(
                f"Hurst estimator stability: std ranges {hurst_std_min:.3f}–{hurst_std_max:.3f} "
                f"({ratio*100:.0f}% of max). Larger windows or DFA typically reduce std."
            )
        lines.append("=" * 96)
        return "\n".join(lines)


# =====================================================================
# Hurst-series statistics helper
# =====================================================================
def _hurst_series_stats(prices: pd.Series, window: int, method: str
                         ) -> Dict[str, float]:
    """Compute Hurst series and summary statistics."""
    if method == "rs":
        h = rolling_hurst(prices, window=window)
    elif method == "dfa":
        h = rolling_dfa_hurst(prices, window=window)
    else:
        raise ValueError(f"Unknown method: {method}")

    nan_frac = float(h.isna().mean())
    h_clean = h.dropna()
    if len(h_clean) == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "nan_fraction": nan_frac, "above_05_fraction": 0.0,
        }
    return {
        "mean": float(h_clean.mean()),
        "std": float(h_clean.std()),
        "min": float(h_clean.min()),
        "max": float(h_clean.max()),
        "nan_fraction": nan_frac,
        "above_05_fraction": float((h_clean > 0.5).mean()),
    }


# =====================================================================
# Experiment runner
# =====================================================================
def run_hurst_ablation(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = (120, 250, 500, 1000),
    methods: Tuple[str, ...] = ("rs", "dfa"),
    base_config: ETHStrategyConfig = None,
    initial_capital: float = 1000.0,
    fee_rate: float = 0.0015,
    verbose: bool = True,
) -> HurstAblationResult:
    """
    Run the Hurst (window, method) ablation experiment.

    Parameters
    ----------
    df : DataFrame
        BTC/ETH OHLCV data. Must contain btc_open, btc_high, btc_low,
        btc_close, eth_open, eth_high, eth_low, eth_close, datetime.
    windows : tuple of int
        Window lengths to test. Default (120, 250, 500, 1000).
    methods : tuple of str
        Methods to test. Default ('rs', 'dfa').
    base_config : ETHStrategyConfig
        Strategy config. If None, defaults are used (other than Hurst).
    initial_capital, fee_rate : float
        Backtester parameters.
    verbose : bool
        Print progress.

    Returns
    -------
    HurstAblationResult
    """
    from dataclasses import replace

    if base_config is None:
        base_config = ETHStrategyConfig()

    results: List[HurstConfigResult] = []
    n_configs = len(windows) * len(methods)
    i = 0

    for window, method in product(windows, methods):
        i += 1
        if verbose:
            print(f"[hurst-ablation] ({i}/{n_configs}) Running window={window}, method={method}...")

        # Compute Hurst series statistics on the underlying ETH close
        hurst_stats = _hurst_series_stats(df["eth_close"], window=window, method=method)

        # Run the strategy with this Hurst config
        cfg = replace(base_config, hurst_window=window, hurst_method=method)
        signals_df = run_eth_strategy(df, config=cfg)
        signals_df = signals_df.rename(columns={
            "eth_open": "open", "eth_high": "high",
            "eth_low": "low", "eth_close": "close",
        })
        bt = run_backtest(signals_df, signal_col="signal",
                          initial_capital=initial_capital,
                          fee_rate=fee_rate)
        m = bt.metrics

        results.append(HurstConfigResult(
            window=window,
            method=method,
            hurst_mean=hurst_stats["mean"],
            hurst_std=hurst_stats["std"],
            hurst_min=hurst_stats["min"],
            hurst_max=hurst_stats["max"],
            hurst_nan_fraction=hurst_stats["nan_fraction"],
            hurst_above_05_fraction=hurst_stats["above_05_fraction"],
            total_trades=int(m.get("total_trades", 0)),
            sharpe=float(m.get("sharpe", 0.0)),
            sortino=float(m.get("sortino", 0.0)),
            total_return=float(m.get("total_return", 0.0)),
            max_drawdown=float(m.get("max_drawdown", 0.0)),
            win_rate=float(m.get("win_rate", 0.0)),
        ))

    return HurstAblationResult(results=results)
