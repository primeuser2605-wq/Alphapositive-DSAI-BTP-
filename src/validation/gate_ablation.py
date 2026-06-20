"""
gate_ablation.py
================
Pre-condition gate ablation experiment for the ETH strategy.

The methodological question
---------------------------
The original report's N3 novelty claim is the "hierarchical gate-then-signal
architecture": three pre-condition filters (Hurst, BTC-ETH correlation, BTC
ATR) must all be satisfied before any entry signal is evaluated. The claim
is that this two-stage architecture reduces false signals taken in the
wrong regime.

The claim is testable. This module tests it.

Procedure
---------
Run the ETH strategy five times on the same data:
1. All three gates active (baseline = original strategy)
2. Hurst gate disabled (other two active)
3. Correlation gate disabled (other two active)
4. ATR gate disabled (other two active)
5. All three gates disabled

For each configuration, report total trades, Sharpe, MDD, total return,
win rate. Rank gates by the magnitude of degradation when removed.

Interpreting the result
-----------------------
A gate that "matters" should produce one or more of the following when removed:
- More trades (gate was filtering some out) AND lower Sharpe (filtered trades
  were better-than-average to remove)
- Larger MDD (gate was filtering catastrophe trades)
- Lower win rate (filtered trades were lower-quality on average)

A gate that doesn't matter shows up as: removing it doesn't change anything.
That outcome would refute the N3 novelty claim for that gate specifically.

The "all gates off" configuration is the most permissive — it lets the
CUSUM+signal layer fire unconstrained. The comparison "all off" vs
"all on" measures the total contribution of the gating architecture.

Output
------
GateAblationResult containing per-configuration metrics and a ranked
attribution: which gate's removal hurts most?

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dataclasses import dataclass, field, replace
from typing import Dict

import numpy as np
import pandas as pd

from backtester import run_backtest, BacktestResult
from strategies.eth_regime_confirmation import (
    ETHStrategyConfig, run_eth_strategy,
)


# =====================================================================
# Result containers
# =====================================================================
@dataclass
class GateConfigResult:
    """Metrics for one gate configuration."""
    name: str  # "all_on", "hurst_off", "corr_off", "atr_off", "all_off"
    enable_hurst: bool
    enable_corr: bool
    enable_atr: bool
    total_trades: int
    long_trades: int
    short_trades: int
    win_rate: float
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float


@dataclass
class GateAblationResult:
    """Full ablation experiment output."""
    configs: Dict[str, GateConfigResult]

    def baseline(self) -> GateConfigResult:
        return self.configs["all_on"]

    def attribution(self) -> Dict[str, Dict[str, float]]:
        """
        Per-gate attribution: how much does removing each gate hurt the
        baseline's metrics?

        For each gate, compute the *delta* of (gate_off - all_on) across
        Sharpe, return, MDD, trades. A gate that "matters" will show
        a meaningful negative Sharpe delta when removed (or a more-negative
        MDD).
        """
        base = self.baseline()
        out: Dict[str, Dict[str, float]] = {}
        for gate_name, cfg_key in [("hurst", "hurst_off"), ("correlation", "corr_off"), ("atr", "atr_off")]:
            if cfg_key not in self.configs:
                continue
            ablated = self.configs[cfg_key]
            out[gate_name] = {
                "delta_sharpe": ablated.sharpe - base.sharpe,
                "delta_return": ablated.total_return - base.total_return,
                "delta_mdd": ablated.max_drawdown - base.max_drawdown,
                "delta_trades": ablated.total_trades - base.total_trades,
                "delta_win_rate": ablated.win_rate - base.win_rate,
            }
        return out

    def report(self) -> str:
        """Render a human-readable table + attribution analysis."""
        lines = []
        lines.append("=" * 88)
        lines.append("ETH STRATEGY — PRE-CONDITION GATE ABLATION")
        lines.append("=" * 88)
        lines.append(f"{'Configuration':<14} {'Hurst':>5} {'Corr':>5} {'ATR':>5}  "
                     f"{'Trades':>6} {'Sharpe':>8} {'Return':>9} {'MaxDD':>8} {'WinR':>6}")
        lines.append("-" * 88)
        order = ["all_on", "hurst_off", "corr_off", "atr_off", "all_off"]
        for key in order:
            if key not in self.configs:
                continue
            c = self.configs[key]
            h_mark = "Y" if c.enable_hurst else "."
            r_mark = "Y" if c.enable_corr else "."
            a_mark = "Y" if c.enable_atr else "."
            lines.append(
                f"{c.name:<14} {h_mark:>5} {r_mark:>5} {a_mark:>5}  "
                f"{c.total_trades:>6d} {c.sharpe:>+8.3f} "
                f"{c.total_return * 100:>+8.2f}% {c.max_drawdown * 100:>+7.2f}% "
                f"{c.win_rate * 100:>5.1f}%"
            )
        lines.append("-" * 88)
        lines.append("")
        lines.append("PER-GATE ATTRIBUTION (delta vs all-on baseline):")
        lines.append(f"{'Gate':<14} {'ΔSharpe':>10} {'ΔReturn':>10} {'ΔMDD':>10} "
                     f"{'ΔTrades':>9} {'ΔWinR':>9}")
        lines.append("-" * 88)
        attr = self.attribution()
        # Rank by absolute Sharpe delta (most-impactful gate first)
        ranked = sorted(attr.items(), key=lambda kv: -abs(kv[1]["delta_sharpe"]))
        for gate_name, deltas in ranked:
            lines.append(
                f"{gate_name:<14} "
                f"{deltas['delta_sharpe']:>+10.3f} "
                f"{deltas['delta_return'] * 100:>+9.2f}% "
                f"{deltas['delta_mdd'] * 100:>+9.2f}% "
                f"{int(deltas['delta_trades']):>+9d} "
                f"{deltas['delta_win_rate'] * 100:>+8.2f}%"
            )
        lines.append("-" * 88)

        # Interpretation
        if ranked:
            most_impactful, most_deltas = ranked[0]
            sharpe_drop = -most_deltas["delta_sharpe"]
            if sharpe_drop > 0.1:
                lines.append(
                    f"VERDICT: The '{most_impactful}' gate matters most — removing it "
                    f"drops Sharpe by {sharpe_drop:.3f}."
                )
            elif sharpe_drop < -0.1:
                # Bizarre: removing the gate HELPS
                lines.append(
                    f"VERDICT: The '{most_impactful}' gate appears to HURT performance — "
                    f"removing it improves Sharpe by {abs(sharpe_drop):.3f}. This is "
                    f"a signal that the gate is mis-tuned for this data."
                )
            else:
                lines.append(
                    f"VERDICT: No gate has a meaningful impact (max ΔSharpe = "
                    f"{abs(sharpe_drop):.3f}). The hierarchical architecture (N3 claim) "
                    f"is not validated on this data."
                )
        lines.append("=" * 88)
        return "\n".join(lines)


# =====================================================================
# Experiment runner
# =====================================================================
def run_gate_ablation(
    df: pd.DataFrame,
    base_config: ETHStrategyConfig = None,
    initial_capital: float = 1000.0,
    fee_rate: float = 0.0015,
    verbose: bool = True,
) -> GateAblationResult:
    """
    Run the gate ablation experiment.

    Parameters
    ----------
    df : DataFrame
        Must contain BTC/ETH OHLCV columns: btc_open, btc_high, btc_low,
        btc_close, eth_open, eth_high, eth_low, eth_close, datetime.
    base_config : ETHStrategyConfig, optional
        Strategy config. If None, defaults are used.
    initial_capital, fee_rate : float
        Backtester parameters.
    verbose : bool
        Print progress.

    Returns
    -------
    GateAblationResult
    """
    if base_config is None:
        base_config = ETHStrategyConfig()

    configurations = [
        ("all_on",     True,  True,  True),
        ("hurst_off",  False, True,  True),
        ("corr_off",   True,  False, True),
        ("atr_off",    True,  True,  False),
        ("all_off",    False, False, False),
    ]

    results: Dict[str, GateConfigResult] = {}

    for name, h, c, a in configurations:
        if verbose:
            print(f"[ablation] Running config '{name}' "
                  f"(hurst={h}, corr={c}, atr={a})...")
        cfg = replace(base_config,
                       enable_hurst_gate=h,
                       enable_correlation_gate=c,
                       enable_atr_gate=a)
        signals_df = run_eth_strategy(df, config=cfg)
        # Rename for the backtester's expected column names
        signals_df = signals_df.rename(columns={
            "eth_open": "open", "eth_high": "high",
            "eth_low": "low", "eth_close": "close",
        })
        bt = run_backtest(signals_df, signal_col="signal",
                          initial_capital=initial_capital,
                          fee_rate=fee_rate)
        m = bt.metrics
        results[name] = GateConfigResult(
            name=name,
            enable_hurst=h, enable_corr=c, enable_atr=a,
            total_trades=int(m.get("total_trades", 0)),
            long_trades=int(m.get("long_trades", 0)),
            short_trades=int(m.get("short_trades", 0)),
            win_rate=float(m.get("win_rate", 0.0)),
            sharpe=float(m.get("sharpe", 0.0)),
            sortino=float(m.get("sortino", 0.0)),
            total_return=float(m.get("total_return", 0.0)),
            max_drawdown=float(m.get("max_drawdown", 0.0)),
        )

    return GateAblationResult(configs=results)
