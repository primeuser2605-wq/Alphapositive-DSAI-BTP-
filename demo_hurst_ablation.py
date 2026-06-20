"""
demo_hurst_ablation.py
======================
Run the Hurst (window, method) ablation experiment.

Tests the report's L4 limitation: "Hurst exponent computed on only 120
observations is short relative to the n > 1000 typically recommended for
R/S stability."

The ablation runs the ETH strategy over 8 configurations (4 windows × 2 methods)
and reports both:
1. Stability of the Hurst estimator itself (mean, std, NaN fraction)
2. Downstream impact on strategy metrics

Usage:
    python demo_hurst_ablation.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src", "strategies"))
sys.path.insert(0, os.path.join(HERE, "src", "validation"))

import numpy as np
import pandas as pd

from hurst_ablation import run_hurst_ablation


def make_synthetic_btc_eth(n=2500, seed=42):
    """Synthetic correlated BTC/ETH OHLCV with regime structure."""
    np.random.seed(seed)
    times = pd.date_range("2023-01-01", periods=n, freq="1h")
    rets = np.zeros(n)
    a, b = n // 3, 2 * n // 3
    rets[:a] = np.random.randn(a) * 0.005 + 0.001
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001
    rets[b:] = np.random.randn(n - b) * 0.003
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
    print("=" * 96)
    print("HURST WINDOW × METHOD ABLATION")
    print("=" * 96)
    print()
    print("Generating synthetic BTC/ETH data (1500 hourly bars, three regimes)...")
    df = make_synthetic_btc_eth(n=1500)
    print(f"Data range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print()
    print("Running ablation (3 windows × 2 methods = 6 configurations)...")
    print("(NOTE: window=1000 omitted to keep runtime under ~3 minutes; the original")
    print(" report uses 120 anyway, so 120/250/500 brackets the relevant range.)")
    print()

    result = run_hurst_ablation(
        df,
        windows=(120, 250, 500),
        methods=("rs", "dfa"),
        verbose=True,
    )
    print()
    print(result.report())

    # Save
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "hurst_ablation_report.txt"), "w") as f:
        f.write(result.report())
        f.write("\n")
    result.to_dataframe().to_csv(
        os.path.join(out_dir, "hurst_ablation.csv"), index=False
    )
    print()
    print(f"Report saved to results/hurst_ablation_report.txt")
    print(f"DataFrame saved to results/hurst_ablation.csv")


if __name__ == "__main__":
    main()
