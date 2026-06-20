"""
rl_vs_rule_experiment.py
========================
The methodological experiment for the project's N4 (interpretability-first
tabular Q-learning) novelty claim.

Procedure
---------
1. Compute features once.
2. Split data into train (70%) and test (30%).
3. For each baseline rule: apply on TEST set, backtest, record metrics.
4. For each RL seed (default 5 seeds): train on TRAIN, apply on TEST,
   backtest, record metrics. Distribution over seeds quantifies RL's
   variance.
5. Report:
   - Each baseline's deterministic metrics
   - RL's mean ± std over seeds for each metric
   - Verdict: does RL's mean exceed the best rule by more than RL's std?

Verdict thresholds (intentionally conservative)
-----------------------------------------------
- "RL adds signal" requires: RL_mean_sharpe > best_rule_sharpe + RL_std_sharpe
  i.e. one standard deviation of separation.

- "RL ties with rules" if: |RL_mean - best_rule| < RL_std

- "RL underperforms" if: RL_mean < best_rule - RL_std

A negative finding here is scientifically valuable. Don't engineer the
experiment until RL wins.

Usage:
    from validation.rl_vs_rule_experiment import run_comparison_experiment
    result = run_comparison_experiment(df, n_seeds=5, train_frac=0.7)
    print(result.report())

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtester import run_backtest, BacktestResult
from strategies.btc_qlearning import (
    BTCQLearningConfig, compute_features, TradingEnvironment,
    train_q_agent, policy_to_signals,
)
from strategies.btc_rule_based import BASELINES, apply_rule


# =====================================================================
# Result container
# =====================================================================
@dataclass
class StrategyMetrics:
    """Minimal metrics extracted from a BacktestResult."""
    name: str
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    long_trades: int
    short_trades: int

    @classmethod
    def from_result(cls, name: str, result: BacktestResult) -> "StrategyMetrics":
        m = result.metrics
        return cls(
            name=name,
            sharpe=m.get("sharpe", 0.0),
            sortino=m.get("sortino", 0.0),
            total_return=m.get("total_return", 0.0),
            max_drawdown=m.get("max_drawdown", 0.0),
            total_trades=m.get("total_trades", 0),
            win_rate=m.get("win_rate", 0.0),
            long_trades=m.get("long_trades", 0),
            short_trades=m.get("short_trades", 0),
        )


@dataclass
class ComparisonResult:
    """Full experiment result."""
    rule_metrics: Dict[str, StrategyMetrics]
    rl_metrics_by_seed: Dict[int, StrategyMetrics]  # one entry per seed
    config_summary: dict

    def rl_summary(self) -> Dict[str, Dict[str, float]]:
        """Mean/std/min/max for each RL metric across seeds."""
        if not self.rl_metrics_by_seed:
            return {}
        agg = {}
        for field_name in ("sharpe", "sortino", "total_return",
                           "max_drawdown", "total_trades", "win_rate"):
            vals = [getattr(m, field_name) for m in self.rl_metrics_by_seed.values()]
            agg[field_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return agg

    def verdict(self) -> str:
        """Render the methodological verdict."""
        if not self.rl_metrics_by_seed:
            return "VERDICT: no RL runs available — cannot compare."

        rl_agg = self.rl_summary()
        rl_mean_sharpe = rl_agg["sharpe"]["mean"]
        rl_std_sharpe = rl_agg["sharpe"]["std"]

        # Best rule by Sharpe, excluding rules that took no trades (degenerate)
        active_rules = {k: v for k, v in self.rule_metrics.items() if v.total_trades > 0}
        if not active_rules:
            return ("VERDICT: no rule baseline produced any trades on this data. "
                    "Cannot make a meaningful comparison. Likely a synthetic-data "
                    "limitation; try real data or relax rule conditions.")
        best_rule_name = max(active_rules, key=lambda k: active_rules[k].sharpe)
        best_rule_sharpe = active_rules[best_rule_name].sharpe

        gap = rl_mean_sharpe - best_rule_sharpe
        # If all RL seeds gave the same answer (std=0), we can't talk about σ separation
        if rl_std_sharpe < 1e-9:
            std_note = " (all RL seeds converged to identical policy — std=0)"
            if gap > 0.1:
                verdict = f"RL OUTPERFORMS (Sharpe gap = {gap:+.3f} vs '{best_rule_name}'){std_note}"
            elif gap < -0.1:
                verdict = f"RL UNDERPERFORMS (Sharpe gap = {gap:+.3f} vs '{best_rule_name}'){std_note}"
            else:
                verdict = f"RL ROUGHLY TIES (Sharpe gap = {gap:+.3f} vs '{best_rule_name}'){std_note}"
            return verdict

        std_units = gap / rl_std_sharpe

        if gap > rl_std_sharpe:
            verdict = (
                f"RL ADDS SIGNAL (Sharpe gap = {gap:+.3f} vs best rule "
                f"'{best_rule_name}'; {std_units:+.2f}σ separation)"
            )
        elif abs(gap) < rl_std_sharpe:
            verdict = (
                f"RL TIES WITH RULES (Sharpe gap = {gap:+.3f} vs best rule "
                f"'{best_rule_name}'; within ±1σ = {rl_std_sharpe:.3f})"
            )
        else:
            verdict = (
                f"RL UNDERPERFORMS (Sharpe gap = {gap:+.3f} vs best rule "
                f"'{best_rule_name}'; {std_units:+.2f}σ — RL costs more than it gives)"
            )

        return verdict

    def report(self) -> str:
        """Pretty-print full results."""
        lines = []
        lines.append("=" * 78)
        lines.append("RL VS RULE-BASED BASELINE EXPERIMENT")
        lines.append("=" * 78)
        cfg = self.config_summary
        lines.append(f"Train bars: {cfg.get('n_train', '?')} | "
                     f"Test bars: {cfg.get('n_test', '?')} | "
                     f"RL seeds: {cfg.get('n_seeds', '?')} | "
                     f"Episodes/seed: {cfg.get('n_episodes', '?')}")
        lines.append("")

        # Rule baselines
        lines.append("RULE-BASED BASELINES (deterministic on test set):")
        lines.append(f"{'Rule':<30} {'Sharpe':>8} {'Return':>8} {'MaxDD':>7} "
                     f"{'Trades':>7} {'WinR':>6}")
        lines.append("-" * 78)
        for name, m in self.rule_metrics.items():
            lines.append(
                f"{name:<30} {m.sharpe:>8.3f} {m.total_return*100:>7.2f}% "
                f"{m.max_drawdown*100:>6.2f}% {m.total_trades:>7d} "
                f"{m.win_rate*100:>5.1f}%"
            )
        lines.append("")

        # RL across seeds
        if self.rl_metrics_by_seed:
            lines.append(f"TABULAR Q-LEARNING (across {len(self.rl_metrics_by_seed)} seeds):")
            agg = self.rl_summary()
            lines.append(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            lines.append("-" * 78)
            for metric in ("sharpe", "sortino", "total_return",
                           "max_drawdown", "win_rate"):
                a = agg[metric]
                fmt = (f"{a['mean']*100:>9.2f}%" if metric in
                       ("total_return", "max_drawdown", "win_rate")
                       else f"{a['mean']:>10.3f}")
                fmt_std = (f"{a['std']*100:>9.2f}%" if metric in
                           ("total_return", "max_drawdown", "win_rate")
                           else f"{a['std']:>10.3f}")
                fmt_min = (f"{a['min']*100:>9.2f}%" if metric in
                           ("total_return", "max_drawdown", "win_rate")
                           else f"{a['min']:>10.3f}")
                fmt_max = (f"{a['max']*100:>9.2f}%" if metric in
                           ("total_return", "max_drawdown", "win_rate")
                           else f"{a['max']:>10.3f}")
                lines.append(f"{metric:<20} {fmt} {fmt_std} {fmt_min} {fmt_max}")
            lines.append("")

        # Verdict
        lines.append("-" * 78)
        lines.append(self.verdict())
        lines.append("=" * 78)
        return "\n".join(lines)


# =====================================================================
# Experiment runner
# =====================================================================
def _backtest_with_signals(signals_df: pd.DataFrame,
                            initial_capital: float = 1000.0,
                            fee_rate: float = 0.0015) -> BacktestResult:
    """Helper: run the backtester with the standard config."""
    # Normalize column names that the backtester expects
    df = signals_df.copy()
    return run_backtest(df, signal_col="signal", price_col="close",
                        open_col="open", initial_capital=initial_capital,
                        fee_rate=fee_rate)


def run_comparison_experiment(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    n_seeds: int = 5,
    config: Optional[BTCQLearningConfig] = None,
    rule_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> ComparisonResult:
    """
    Run the full RL-vs-rule experiment.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns datetime, open, high, low, close, [volume].
    train_frac : float
        Fraction of data used for RL training. Rules don't train.
    n_seeds : int
        Number of independent RL training runs (each with a different seed).
    config : BTCQLearningConfig
        RL hyperparameters. If None, default config is used. The config's
        `n_episodes` controls training length.
    rule_names : list of str
        Which rules from `BASELINES` to evaluate. If None, all are evaluated.
    verbose : bool
        Print progress.

    Returns
    -------
    ComparisonResult
    """
    if config is None:
        config = BTCQLearningConfig()
    if rule_names is None:
        rule_names = list(BASELINES.keys())

    # --- Split data ---
    n = len(df)
    n_train = int(n * train_frac)
    test_df = df.iloc[n_train:].reset_index(drop=True)
    if verbose:
        print(f"[experiment] Train bars: {n_train} | Test bars: {len(test_df)}")

    # --- Rule baselines (deterministic) ---
    rule_metrics: Dict[str, StrategyMetrics] = {}
    for rule_name in rule_names:
        if verbose:
            print(f"[experiment] Evaluating rule: {rule_name}")
        # apply_rule computes features and signals on the full df, then we take test slice
        signals_full = apply_rule(df, rule_name, config)
        # Take the same test slice (preserving the index)
        signals_test = signals_full.iloc[n_train:].reset_index(drop=True)
        result = _backtest_with_signals(signals_test, fee_rate=config.commission_rate)
        rule_metrics[rule_name] = StrategyMetrics.from_result(rule_name, result)

    # --- RL across seeds ---
    rl_metrics_by_seed: Dict[int, StrategyMetrics] = {}
    # Pre-compute features once (deterministic)
    feat_full = compute_features(df, config)
    feat_full = feat_full.dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"]
    ).reset_index(drop=True)

    # Find where the train/test split lands in the post-dropna data
    train_end_idx = min(n_train, len(feat_full))
    train_feat = feat_full.iloc[:train_end_idx]
    test_feat = feat_full.iloc[train_end_idx:].reset_index(drop=True)

    if verbose:
        print(f"[experiment] After feature dropna: "
              f"train_feat={len(train_feat)}, test_feat={len(test_feat)}")

    for seed in range(n_seeds):
        if verbose:
            print(f"[experiment] RL training seed {seed + 1}/{n_seeds}...")
        # Build a seed-specific config (the strategy reads config.seed internally)
        from dataclasses import replace
        seeded_config = replace(config, seed=seed)
        env_train = TradingEnvironment(
            train_feat["close"].values,
            {
                "rsi_signal": train_feat["rsi_signal"].values,
                "ema_signal": train_feat["ema_signal"].values,
                "aroon_signal": train_feat["aroon_signal"].values,
                "pct_change": train_feat["pct_change"].values,
            },
            seeded_config,
        )
        Q = train_q_agent(env_train, seeded_config, verbose=False)

        # Apply greedy policy to TEST
        test_signals = policy_to_signals(test_feat, Q, seeded_config)
        test_signals_df = test_feat.copy()
        test_signals_df["signal"] = test_signals
        result = _backtest_with_signals(test_signals_df,
                                         fee_rate=seeded_config.commission_rate)
        rl_metrics_by_seed[seed] = StrategyMetrics.from_result(
            f"RL seed={seed}", result
        )

    return ComparisonResult(
        rule_metrics=rule_metrics,
        rl_metrics_by_seed=rl_metrics_by_seed,
        config_summary={
            "n_train": n_train,
            "n_test": n - n_train,
            "n_seeds": n_seeds,
            "n_episodes": config.n_episodes,
        },
    )
