"""
demo_rl_vs_rule.py
==================
Run the RL-vs-rule-based-baseline experiment and report the verdict.

This is the methodological experiment that tests the project's N4 novelty
claim ("interpretability-first tabular Q-learning adds value beyond what
rule-based systems on the same features provide").

On synthetic data, this gives a controlled environment for testing the
experiment harness. On real data, this gives the actual answer.

Usage:
    python demo_rl_vs_rule.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src", "strategies"))
sys.path.insert(0, os.path.join(HERE, "src", "validation"))

import numpy as np
import pandas as pd

from btc_qlearning import BTCQLearningConfig
from rl_vs_rule_experiment import run_comparison_experiment


def make_synthetic_btc(n=3000, seed=42):
    """Synthetic BTC OHLCV with three regimes (bull / bear / chop)."""
    np.random.seed(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="1h")
    rets = np.zeros(n)
    a, b = n // 3, 2 * n // 3
    rets[:a] = np.random.randn(a) * 0.005 + 0.001       # bull
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001  # bear
    rets[b:] = np.random.randn(n - b) * 0.003           # chop
    close = 20000 * np.cumprod(1 + rets)
    open_p = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(close, open_p) * (1 + np.abs(np.random.randn(n) * 0.001))
    low = np.minimum(close, open_p) * (1 - np.abs(np.random.randn(n) * 0.001))
    return pd.DataFrame({
        "datetime": times,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": np.abs(np.random.randn(n)) * 100,
    })


def main():
    print("Generating synthetic BTC data (3000 bars, three regimes)...")
    df = make_synthetic_btc(n=3000)

    print("Running comparison experiment (5 RL seeds × 30 episodes each)...")
    print("This will take 1-2 minutes.\n")
    config = BTCQLearningConfig(n_episodes=30)  # short but not trivial
    result = run_comparison_experiment(
        df, train_frac=0.7, n_seeds=5, config=config, verbose=True,
    )

    print()
    print(result.report())

    # Save the result for the README / report
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rl_vs_rule_synthetic.txt"), "w") as f:
        f.write(result.report())
    print(f"\nReport saved to results/rl_vs_rule_synthetic.txt")


if __name__ == "__main__":
    main()
