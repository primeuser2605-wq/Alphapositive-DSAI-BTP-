"""
demo_reward_comparison.py
=========================
Run the BTC reward function comparison experiment.

Tests the report's L5 fix: does the Moody-Saffell log-utility reward fix
the 30:1 long/short asymmetry that the original reward produced?

Usage:
    python demo_reward_comparison.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src", "strategies"))
sys.path.insert(0, os.path.join(HERE, "src", "validation"))

import numpy as np
import pandas as pd

from btc_qlearning import BTCQLearningConfig
from reward_comparison import run_reward_comparison


def make_synthetic_btc(n=3000, seed=42):
    """Synthetic BTC OHLCV with regime structure."""
    np.random.seed(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="1h")
    rets = np.zeros(n)
    a, b = n // 3, 2 * n // 3
    rets[:a] = np.random.randn(a) * 0.005 + 0.001   # bull
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001  # bear
    rets[b:] = np.random.randn(n - b) * 0.003       # chop
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
    print("=" * 96)
    print("BTC REWARD FUNCTION COMPARISON")
    print("=" * 96)
    print()
    print("Tests L5: does Moody-Saffell log-utility reward fix the")
    print("original reward function's long/short asymmetry?")
    print()

    df = make_synthetic_btc(n=3000)
    print(f"Generated synthetic BTC data: {len(df)} bars, three regimes.")
    print()

    cfg = BTCQLearningConfig(n_episodes=50)
    print(f"Running comparison: 2 rewards × 5 seeds × {cfg.n_episodes} episodes = 10 training runs.")
    print("This will take a few minutes.")
    print()

    result = run_reward_comparison(
        df, n_seeds=5, base_config=cfg, train_frac=0.7, verbose=True,
    )
    print()
    print(result.report())

    # Save
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "reward_comparison_report.txt"), "w") as f:
        f.write(result.report())
        f.write("\n")
    print()
    print(f"Report saved to results/reward_comparison_report.txt")


if __name__ == "__main__":
    main()
