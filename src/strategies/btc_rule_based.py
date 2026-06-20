"""
btc_rule_based.py
=================
Rule-based baseline strategies for BTC, using the SAME features as the
tabular Q-learning agent in `btc_qlearning.py`.

The point of this module
------------------------
Validate (or refute) the claim that tabular Q-learning adds signal beyond
what a competent hand-coded rule on the same features would provide. This
is the core test for the project's "interpretability-first RL" novelty
argument (N4 in the report).

If a simple rule on (RSI, EMA, Aroon, position) features matches or exceeds
the Q-learning agent's performance:
  -> RL adds nothing on this feature set; the agent is rediscovering rules
  -> Honest negative result; the right next step is richer features
     (volatility regime, time-of-day, volume), not deeper RL

If RL meaningfully outperforms every reasonable rule:
  -> The agent is finding patterns hand-coding missed
  -> The interpretability-first design's value is demonstrated

Either outcome is informative. The unscientific outcome would be: not running
this comparison at all.

The rules implemented here are deliberately NOT tuned. Tuning a rule until
it matches RL would defeat the purpose. The contract is: write the rule
the way a thoughtful trader would, before seeing RL's results, and compare.

Rules implemented
-----------------
1. `rule_momentum_confluence`: long if all three signals agree bullish (RSI≥0,
   EMA bullish, Aroon bullish); short on the mirror; flat otherwise.

2. `rule_majority_vote`: ternary sum of the three signals; long if ≥+2,
   short if ≤-2, else flat.

3. `rule_mean_reversion`: RSI in the classical mean-reversion sense
   (long when RSI<35, short when RSI>75). Included as a deliberate
   strawman — momentum-style features used mean-reversion-style; should
   underperform.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from strategies.btc_qlearning import compute_features, BTCQLearningConfig


# =====================================================================
# Helper
# =====================================================================
def _signal_from_features(feat: pd.DataFrame, decide_fn: Callable) -> pd.Series:
    """
    Apply a per-row decision function to features and return a signal series.

    Parameters
    ----------
    feat : pd.DataFrame
        Must contain columns: rsi_signal, ema_signal, aroon_signal, pct_change, rsi.
    decide_fn : Callable
        Takes a dict of feature values, returns -1, 0, or +1.

    Returns
    -------
    pd.Series of -1/0/+1 with the same index as feat.
    """
    signals = []
    for _, row in feat.iterrows():
        s = decide_fn({
            "rsi_signal": row.get("rsi_signal", 0),
            "ema_signal": row.get("ema_signal", 0),
            "aroon_signal": row.get("aroon_signal", 0),
            "rsi": row.get("rsi", 50),  # raw RSI for mean-reversion rule
            "pct_change": row.get("pct_change", 0),
        })
        signals.append(s)
    return pd.Series(signals, index=feat.index, dtype=int)


# =====================================================================
# Rule 1: Momentum Confluence
# =====================================================================
def rule_momentum_confluence(df: pd.DataFrame, config: BTCQLearningConfig = None
                              ) -> pd.DataFrame:
    """
    Long when RSI_signal ≥ 0, EMA_signal = +1, Aroon_signal = +1.
    Short on the mirror (RSI_signal ≤ 0, EMA = -1, Aroon = -1).
    Flat otherwise.

    Rationale: this is the obvious "all signals agree" interpretation of
    the three ternary features. A competent trader given these features
    would write something like this.
    """
    if config is None:
        config = BTCQLearningConfig()
    out = compute_features(df, config)

    def decide(f):
        if f["ema_signal"] == 1 and f["aroon_signal"] == 1 and f["rsi_signal"] >= 0:
            return 1
        if f["ema_signal"] == -1 and f["aroon_signal"] == -1 and f["rsi_signal"] <= 0:
            return -1
        return 0

    out["signal"] = _signal_from_features(out, decide)
    return out


# =====================================================================
# Rule 2: Majority Vote
# =====================================================================
def rule_majority_vote(df: pd.DataFrame, config: BTCQLearningConfig = None
                       ) -> pd.DataFrame:
    """
    Long if sum of the three ternary signals is ≥ +2.
    Short if sum ≤ -2.
    Flat otherwise.

    Rationale: more permissive than momentum confluence. Two out of three
    signals agreeing is enough to take a position.
    """
    if config is None:
        config = BTCQLearningConfig()
    out = compute_features(df, config)

    def decide(f):
        total = f["rsi_signal"] + f["ema_signal"] + f["aroon_signal"]
        if total >= 2:
            return 1
        if total <= -2:
            return -1
        return 0

    out["signal"] = _signal_from_features(out, decide)
    return out


# =====================================================================
# Rule 3: Mean Reversion (strawman)
# =====================================================================
def rule_mean_reversion(df: pd.DataFrame, config: BTCQLearningConfig = None
                        ) -> pd.DataFrame:
    """
    Long when raw RSI < 35; short when raw RSI > 75. Flat otherwise.

    Rationale: this is the classical mean-reversion reading of RSI ('buy
    the dip, sell the rip'). It contradicts the momentum-style framing of
    the other features. Included as a deliberate strawman to bracket the
    space: a competent rule should beat this, and so should RL.
    """
    if config is None:
        config = BTCQLearningConfig()
    out = compute_features(df, config)

    def decide(f):
        rsi = f["rsi"]
        if pd.isna(rsi):
            return 0
        if rsi < 35:
            return 1
        if rsi > 75:
            return -1
        return 0

    out["signal"] = _signal_from_features(out, decide)
    return out


# =====================================================================
# Registry & info
# =====================================================================
@dataclass
class RuleInfo:
    """Metadata about a baseline rule."""
    name: str
    short_description: str
    expected_role: str  # "strong baseline", "moderate baseline", "strawman"
    fn: Callable


BASELINES = {
    "momentum_confluence": RuleInfo(
        name="Momentum Confluence",
        short_description="Long if RSI≥0 AND EMA bull AND Aroon bull",
        expected_role="strong baseline",
        fn=rule_momentum_confluence,
    ),
    "majority_vote": RuleInfo(
        name="Majority Vote",
        short_description="Long if sum of 3 ternary signals ≥ +2",
        expected_role="moderate baseline",
        fn=rule_majority_vote,
    ),
    "mean_reversion": RuleInfo(
        name="Mean Reversion (RSI)",
        short_description="Long if RSI<35, short if RSI>75",
        expected_role="strawman",
        fn=rule_mean_reversion,
    ),
}


def apply_rule(df: pd.DataFrame, rule_name: str,
               config: BTCQLearningConfig = None) -> pd.DataFrame:
    """Apply a named rule from the registry."""
    if rule_name not in BASELINES:
        raise KeyError(f"Unknown rule '{rule_name}'. Available: {list(BASELINES.keys())}")
    return BASELINES[rule_name].fn(df, config)
