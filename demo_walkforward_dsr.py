"""
demo_walkforward_dsr.py
========================
Demonstrate walk-forward validation + Deflated Sharpe Ratio end-to-end
on synthetic data.

The pipeline:
1. Generate synthetic BTC data with three regimes
2. Run walk-forward with 8 folds (rolling window)
3. Compute DSR from the fold results
4. Print the report and verdict

This is what the rigor section of a portfolio looks like — measure
performance across multiple disjoint OOS windows, then correct for
multiple-testing bias.

Usage:
    python demo_walkforward_dsr.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src", "strategies"))
sys.path.insert(0, os.path.join(HERE, "src", "validation"))

import numpy as np
import pandas as pd

from walkforward import run_walkforward
from deflated_sharpe import deflated_sharpe_from_walkforward
from btc_qlearning import BTCQLearningConfig
from btc_rule_based import rule_momentum_confluence


def make_synthetic_btc(n=4000, seed=42):
    """Synthetic BTC OHLCV with regime structure."""
    np.random.seed(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="1h")
    rets = np.zeros(n)
    a, b, c = n // 4, n // 2, 3 * n // 4
    rets[:a] = np.random.randn(a) * 0.005 + 0.001
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001
    rets[b:c] = np.random.randn(c - b) * 0.003 + 0.0005
    rets[c:] = np.random.randn(n - c) * 0.006 - 0.0005
    close = 20000 * np.cumprod(1 + rets)
    open_p = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(close, open_p) * (1 + np.abs(np.random.randn(n) * 0.001))
    low = np.minimum(close, open_p) * (1 - np.abs(np.random.randn(n) * 0.001))
    return pd.DataFrame({
        "datetime": times,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": np.abs(np.random.randn(n)) * 100,
    })


def momentum_strategy_adapter(train_df, test_df):
    """Wrap rule_momentum_confluence in the walk-forward strategy_fn interface."""
    # The rule needs the full DataFrame to compute features causally,
    # so we concatenate train and test, apply the rule, then return just the test slice.
    combined = pd.concat([train_df, test_df], ignore_index=True)
    out = rule_momentum_confluence(combined, BTCQLearningConfig())
    # Take only the test portion
    return out.iloc[len(train_df):].reset_index(drop=True)


def main():
    print("=" * 78)
    print("WALK-FORWARD + DEFLATED SHARPE RATIO DEMO")
    print("=" * 78)
    print()
    print("Generating synthetic BTC data (4000 hourly bars, four regimes)...")
    df = make_synthetic_btc(n=4000)
    print(f"Data range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print()

    print("Running walk-forward (rolling window, 600 train / 200 test, "
          "embargo=24, purge=10)...")
    print()
    wf_result = run_walkforward(
        df,
        strategy_fn=momentum_strategy_adapter,
        train_bars=600,
        test_bars=200,
        step_bars=200,        # non-overlapping test slices
        mode="rolling",
        purge_bars=10,
        embargo_bars=24,
        initial_capital=1000.0,
        fee_rate=0.0015,
        verbose=True,
    )
    print()
    print(wf_result.report())
    print()

    if len(wf_result.folds) < 2:
        print("Need ≥ 2 folds for DSR; got {len(wf_result.folds)}. Skipping.")
        return

    print()
    print("Computing Deflated Sharpe Ratio from walk-forward result...")
    dsr = deflated_sharpe_from_walkforward(wf_result)
    print()
    print(dsr)
    print()
    print("VERDICT:")
    print(f"  {dsr.verdict()}")

    # Save artifacts
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    wf_result.folds_df().to_csv(
        os.path.join(out_dir, "walkforward_folds.csv"), index=False
    )
    with open(os.path.join(out_dir, "walkforward_dsr_report.txt"), "w") as f:
        f.write(wf_result.report())
        f.write("\n\n")
        f.write(str(dsr))
        f.write("\n\nVerdict:\n")
        f.write(dsr.verdict())
        f.write("\n")
    print()
    print(f"Results saved to results/walkforward_folds.csv and "
          f"results/walkforward_dsr_report.txt")
    print("=" * 78)


if __name__ == "__main__":
    main()
