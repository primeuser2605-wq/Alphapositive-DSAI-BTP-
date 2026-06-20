"""
demo_full_pipeline.py
=====================
End-to-end demonstration of the validation infrastructure on synthetic data.

This script demonstrates, on synthetic BTC/ETH data:
1. Computing features and signals via the refactored ETH strategy
2. Running the in-house backtester
3. Computing Politis-Romano bootstrap confidence intervals on Sharpe, Sortino, etc.
4. Producing a complete report

To run on real 2020-2023 data, replace `make_synthetic_btc_eth()` with a
loader for your CSVs. The rest of the pipeline is unchanged.

Usage:
    python demo_full_pipeline.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src", "strategies"))
sys.path.insert(0, os.path.join(HERE, "src", "validation"))

import numpy as np
import pandas as pd

from eth_regime_confirmation import run_eth_strategy, ETHStrategyConfig
from backtester import run_backtest
from bootstrap import bootstrap_all_metrics, summarize_bootstrap


def make_synthetic_btc_eth(n=3000, seed=42):
    """
    Build synthetic correlated BTC/ETH data with bull-bear-flat regime
    structure so the strategy can actually trigger trades.
    """
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
    print("=" * 70)
    print("END-TO-END VALIDATION PIPELINE DEMO")
    print("=" * 70)
    print()

    # --- 1. Load data ---
    print("[1/4] Loading data (synthetic, n=3000 hourly bars)...")
    df = make_synthetic_btc_eth(n=3000)
    print(f"      Data range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print()

    # --- 2. Run the strategy ---
    print("[2/4] Running ETH regime-confirmation strategy...")
    config = ETHStrategyConfig()
    signals_df = run_eth_strategy(df, config=config)
    n_long = (signals_df["signal"] == 1).sum()
    n_short = (signals_df["signal"] == -1).sum()
    n_flat = (signals_df["signal"] == 0).sum()
    print(f"      Signal counts: long={n_long}, short={n_short}, flat={n_flat}")
    print()

    # Rename ETH columns for the backtester convention
    signals_df = signals_df.rename(columns={
        "eth_open": "open", "eth_high": "high",
        "eth_low": "low", "eth_close": "close",
    })

    # --- 3. Run the backtester ---
    print("[3/4] Running in-house backtester...")
    result = run_backtest(
        signals_df, signal_col="signal",
        initial_capital=1000.0,
        fee_rate=0.0015,
        leverage=1.0,
    )
    print()
    print(result.summary())
    print()

    # --- 4. Bootstrap confidence intervals ---
    trades_df = result.trades_df()
    if len(trades_df) < 10:
        print(f"[4/4] Skipping bootstrap: only {len(trades_df)} trades "
              f"(need >= 10 for meaningful CIs).")
        return

    print(f"[4/4] Computing Politis-Romano bootstrap CIs from {len(trades_df)} trades...")
    trade_returns = trades_df["return_pct"].values
    cis = bootstrap_all_metrics(
        trade_returns,
        n_resamples=5000,
        mean_block_length=10.0,
        confidence_level=0.95,
        seed=42,
    )
    print()
    print(summarize_bootstrap(cis))
    print()

    # --- Save results ---
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    trades_df.to_csv(os.path.join(out_dir, "demo_trades.csv"), index=False)
    result.equity_curve.to_csv(os.path.join(out_dir, "demo_equity.csv"))
    result.per_bar.to_csv(os.path.join(out_dir, "demo_per_bar.csv"))
    print(f"Results saved to {out_dir}/")
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
