"""
reward_comparison.py
====================
Reward function comparison experiment for the BTC Q-learning agent.

The methodological question
---------------------------
The report's L5 limitation:
  "The current reward mixes unrealized P&L, realized P&L, commissions, a
   flat-position penalty, and large terminal bankruptcy rewards at
   incommensurable scales. The 30/1 long/short split in 2023 is consistent
   with the agent learning 'always hold a position' rather than time long vs
   short. Proposed solution: log-utility differential reward per Moody &
   Saffell (2001)."

The fix is implemented in btc_qlearning.py as `reward_type='log_utility'`.
This module tests whether it actually fixes the long/short imbalance.

Procedure
---------
For each reward type ('original', 'log_utility'):
  - Train the agent N_SEEDS times (different random seeds)
  - Evaluate on the test slice
  - Record: Sharpe, Sortino, total return, MDD, win rate, # long, # short

Compare the two distributions on:
  - Long/short balance (the original L5 symptom)
  - Sharpe (does the new reward help or hurt aggregate edge?)
  - Variance across seeds (which reward learns more consistently?)

Output: a side-by-side comparison + a verdict on whether log_utility
fixes the asymmetry without sacrificing performance.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dataclasses import dataclass, field, replace
from typing import Dict, List

import numpy as np
import pandas as pd

from backtester import run_backtest
from strategies.btc_qlearning import (
    BTCQLearningConfig, compute_features, TradingEnvironment,
    train_q_agent, policy_to_signals,
)


# =====================================================================
# Result containers
# =====================================================================
@dataclass
class RewardSeedResult:
    """Per-seed result for one reward function."""
    reward_type: str
    seed: int
    total_trades: int
    long_trades: int
    short_trades: int
    win_rate: float
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float

    @property
    def long_fraction(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.long_trades / self.total_trades


@dataclass
class RewardComparisonResult:
    """Full experiment output."""
    by_seed: Dict[str, List[RewardSeedResult]]  # reward_type → list of seed results

    def aggregate(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Per-reward × per-metric × {mean, std, min, max}."""
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for reward_type, results in self.by_seed.items():
            if not results:
                continue
            out[reward_type] = {}
            for metric in ("sharpe", "sortino", "total_return", "max_drawdown",
                           "win_rate", "total_trades", "long_fraction"):
                vals = [getattr(r, metric) for r in results]
                out[reward_type][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        return out

    def report(self) -> str:
        """Render the comparison."""
        agg = self.aggregate()
        lines = []
        lines.append("=" * 96)
        lines.append("BTC Q-LEARNING — REWARD FUNCTION COMPARISON (tests L5 fix)")
        lines.append("=" * 96)
        if not agg:
            return "\n".join(lines + ["(no results)"])

        n_seeds = len(next(iter(self.by_seed.values())))
        lines.append(f"Across {n_seeds} seeds per reward type:")
        lines.append("")

        # Per-metric table
        rewards = sorted(agg.keys())
        metrics_to_show = [
            ("sharpe",       "Sharpe",        "{:>+9.3f}", False),
            ("sortino",      "Sortino",       "{:>+9.3f}", False),
            ("total_return", "Return",        "{:>+8.2f}%", True),
            ("max_drawdown", "MaxDD",         "{:>+8.2f}%", True),
            ("win_rate",     "WinRate",       "{:>+8.2f}%", True),
            ("total_trades", "Trades",        "{:>9.1f}",   False),
            ("long_fraction", "LongFrac",     "{:>8.2f}",   False),
        ]

        # Header
        header_parts = [f"{'Metric':<14}"]
        for r in rewards:
            header_parts.append(f"{r + ' (mean)':>16}")
            header_parts.append(f"{r + ' (std)':>16}")
        lines.append("  ".join(header_parts))
        lines.append("-" * 96)

        for metric, label, fmt, is_pct in metrics_to_show:
            row = [f"{label:<14}"]
            for r in rewards:
                stats = agg[r][metric]
                m = stats["mean"]
                s = stats["std"]
                if is_pct:
                    m, s = m * 100, s * 100
                row.append(f"{fmt.format(m):>16}")
                row.append(f"{fmt.format(s):>16}")
            lines.append("  ".join(row))

        lines.append("-" * 96)
        lines.append("")

        # Verdict block: focus on L5's actual symptom (long/short imbalance)
        if "original" in agg and "log_utility" in agg:
            orig = agg["original"]
            log_u = agg["log_utility"]
            orig_long_frac = orig["long_fraction"]["mean"]
            log_long_frac = log_u["long_fraction"]["mean"]
            orig_sharpe = orig["sharpe"]["mean"]
            log_sharpe = log_u["sharpe"]["mean"]

            lines.append("L5 SYMPTOM (long/short asymmetry):")
            lines.append(f"  Original reward:   {orig_long_frac*100:5.1f}% long")
            lines.append(f"  Log-utility reward: {log_long_frac*100:5.1f}% long")
            balance_shift = abs(orig_long_frac - 0.5) - abs(log_long_frac - 0.5)
            if balance_shift > 0.05:
                lines.append(f"  → Log-utility BALANCES trades better "
                             f"(closer to 50/50 by {balance_shift*100:.1f}pp)")
            elif balance_shift < -0.05:
                lines.append(f"  → Log-utility UNBALANCES trades MORE — not the L5 fix")
            else:
                lines.append(f"  → No meaningful change in trade balance")
            lines.append("")

            # Aggregate performance comparison
            lines.append("PERFORMANCE COMPARISON:")
            lines.append(f"  Original:    Sharpe mean = {orig_sharpe:+.3f} "
                         f"(std {orig['sharpe']['std']:.3f})")
            lines.append(f"  Log-utility: Sharpe mean = {log_sharpe:+.3f} "
                         f"(std {log_u['sharpe']['std']:.3f})")
            gap = log_sharpe - orig_sharpe
            if abs(gap) < 0.05:
                lines.append(f"  → Performance is ~equivalent (ΔSharpe = {gap:+.3f})")
            elif gap > 0:
                lines.append(f"  → Log-utility outperforms by ΔSharpe = {gap:+.3f}")
            else:
                lines.append(f"  → Log-utility underperforms by ΔSharpe = {gap:+.3f}")
            lines.append("")

            # Final verdict
            lines.append("VERDICT:")
            fix_works = balance_shift > 0.05
            performance_ok = gap > -0.1
            log_trades = log_u["total_trades"]["mean"]
            orig_trades = orig["total_trades"]["mean"]
            # Special case: log_utility produces ~zero trades
            if log_trades < 0.5 and orig_trades >= 1.0:
                lines.append("  The L5 fix SUPPRESSES TRADING on this data: with the")
                lines.append("  flat-position penalty removed, the agent learns 'do nothing'.")
                lines.append("  This is technically correct behavior — under log-utility,")
                lines.append("  staying flat IS the rational policy when no edge exists.")
                lines.append("  Whether this is a net win depends on whether real data")
                lines.append("  presents enough genuine edge for the agent to selectively trade.")
                lines.append("  On synthetic data here: original reward forces (mostly bad)")
                lines.append("  trades; log-utility correctly stays out. Neither is wrong;")
                lines.append("  they answer different questions.")
            elif fix_works and performance_ok:
                lines.append("  The L5 fix WORKS. Log-utility reward shifts the agent toward")
                lines.append("  more balanced long/short trading without sacrificing performance.")
            elif fix_works and not performance_ok:
                lines.append("  The L5 fix PARTIALLY works. Trade balance is improved, but at")
                lines.append("  the cost of Sharpe. Trade-off; pick based on risk preferences.")
            elif not fix_works and performance_ok:
                lines.append("  The L5 fix is INEFFECTIVE on this data. Trade asymmetry persists.")
                lines.append("  The root cause may be in data regime, not reward design.")
            else:
                lines.append("  The L5 fix DOESN'T HELP on this data. Neither balance nor Sharpe")
                lines.append("  improves. Reconsider the proposed fix.")

        lines.append("=" * 96)
        return "\n".join(lines)


# =====================================================================
# Experiment runner
# =====================================================================
def run_reward_comparison(
    df: pd.DataFrame,
    n_seeds: int = 5,
    base_config: BTCQLearningConfig = None,
    train_frac: float = 0.7,
    initial_capital: float = 1000.0,
    fee_rate: float = 0.0015,
    verbose: bool = True,
) -> RewardComparisonResult:
    """
    Run the reward-function comparison experiment.

    Parameters
    ----------
    df : DataFrame
        BTC OHLCV. Must have columns datetime, open, high, low, close, volume.
    n_seeds : int
        Number of seeds per reward type. Default 5.
    base_config : BTCQLearningConfig, optional
        Base config (reward_type is overridden per run).
    train_frac : float
        Train/test split fraction. Default 0.7.
    initial_capital, fee_rate : float
        Backtester parameters.
    verbose : bool
        Print progress.

    Returns
    -------
    RewardComparisonResult
    """
    if base_config is None:
        base_config = BTCQLearningConfig()

    reward_types = ("original", "log_utility")
    by_seed: Dict[str, List[RewardSeedResult]] = {r: [] for r in reward_types}

    # Pre-compute features once (deterministic given data)
    feat = compute_features(df, base_config).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"]
    ).reset_index(drop=True)
    n_train = int(train_frac * len(feat))
    train_feat = feat.iloc[:n_train].reset_index(drop=True)
    test_feat = feat.iloc[n_train:].reset_index(drop=True)

    if verbose:
        print(f"[reward-comparison] train_feat={len(train_feat)}, test_feat={len(test_feat)}")

    for reward_type in reward_types:
        for seed in range(n_seeds):
            if verbose:
                print(f"[reward-comparison] reward={reward_type}, seed={seed}...")
            cfg = replace(base_config, reward_type=reward_type, seed=seed)
            env = TradingEnvironment(
                train_feat["close"].values,
                {
                    "rsi_signal": train_feat["rsi_signal"].values,
                    "ema_signal": train_feat["ema_signal"].values,
                    "aroon_signal": train_feat["aroon_signal"].values,
                    "pct_change": train_feat["pct_change"].values,
                },
                cfg,
            )
            Q = train_q_agent(env, cfg, verbose=False)
            signals = policy_to_signals(test_feat, Q, cfg)
            test_with_signals = test_feat.copy()
            test_with_signals["signal"] = signals
            bt = run_backtest(test_with_signals, signal_col="signal",
                              initial_capital=initial_capital, fee_rate=fee_rate)
            m = bt.metrics
            by_seed[reward_type].append(RewardSeedResult(
                reward_type=reward_type,
                seed=seed,
                total_trades=int(m.get("total_trades", 0)),
                long_trades=int(m.get("long_trades", 0)),
                short_trades=int(m.get("short_trades", 0)),
                win_rate=float(m.get("win_rate", 0.0)),
                sharpe=float(m.get("sharpe", 0.0)),
                sortino=float(m.get("sortino", 0.0)),
                total_return=float(m.get("total_return", 0.0)),
                max_drawdown=float(m.get("max_drawdown", 0.0)),
            ))

    return RewardComparisonResult(by_seed=by_seed)
