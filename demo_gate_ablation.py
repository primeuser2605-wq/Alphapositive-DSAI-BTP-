"""
demo_gate_ablation.py
=====================
Run the ETH gate ablation experiment.

Tests the report's N3 novelty claim ('hierarchical gate-then-signal
architecture'). Each pre-condition gate is removed in turn; the impact
on Sharpe, return, MDD, trade count, and win rate is measured.

Usage:
    python demo_gate_ablation.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src", "strategies"))
sys.path.insert(0, os.path.join(HERE, "src", "validation"))

import numpy as np
import pandas as pd

from gate_ablation import run_gate_ablation


def make_synthetic_btc_eth(n=2500, seed=42):
    """Synthetic correlated BTC/ETH OHLCV with regime structure."""
    np.random.seed(seed)
    times = pd.date_range("2023-01-01", periods=n, freq="1h")
    rets = np.zeros(n)
    a, b = n // 3, 2 * n // 3
    rets[:a] = np.random.randn(a) * 0.005 + 0.001       # bull
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001  # bear
    rets[b:] = np.random.randn(n - b) * 0.003           # chop
    btc_close = 20000 * np.cumprod(1 + rets)
    eth_rets = 0.85 * rets + np.random.randn(n) * 0.003
    eth_close = 1500 * np.cumprod(1 + eth_rets)
    def ohlcv(close):
        open_p = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(close, open_p) * (1 + np.abs(np.random.randn(n) * 0.001))
        low = np.minimum(close, open_p) * (1 - np.abs(np.random.randn(n) * 0.001))
        return open_p, high, low
    btc_o, btc_h, btc_l = ohlcv(btc_close)
    eth_o, eth_h, eth_l = ohlcv(eth_close)
    return pd.DataFrame({
        "datetime": times,
        "btc_open": btc_o, "btc_high": btc_h, "btc_low": btc_l, "btc_close": btc_close,
        "eth_open": eth_o, "eth_high": eth_h, "eth_low": eth_l, "eth_close": eth_close,
    })


def main():
    print("=" * 88)
    print("ETH GATE ABLATION EXPERIMENT")
    print("=" * 88)
    print()
    print("Generating synthetic BTC/ETH data (2500 hourly bars, three regimes)...")
    df = make_synthetic_btc_eth(n=2500)
    print(f"Data range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print()
    print("Running ablation (5 strategy configurations)...")
    print()

    result = run_gate_ablation(df, verbose=True)
    print()
    print(result.report())

    # Save
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gate_ablation_report.txt"), "w") as f:
        f.write(result.report())
        f.write("\n")
    print()
    print(f"Report saved to results/gate_ablation_report.txt")


if __name__ == "__main__":
    main()
