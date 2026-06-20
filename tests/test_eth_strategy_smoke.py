"""
test_eth_strategy_smoke.py
==========================
Smoke test: the refactored ETH strategy runs end-to-end on synthetic data
and produces backtest results without errors.

This is NOT a correctness test against the original strategy's reported
numbers — that requires the actual 2020-2023 data, which is in /data/.
It tests the contract: feed in OHLCV, get out signals, run backtester, get
a valid BacktestResult.
"""
import sys, os
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')

import numpy as np
import pandas as pd

from eth_regime_confirmation import run_eth_strategy, ETHStrategyConfig
from backtester import run_backtest


def make_synthetic_btc_eth(n=2000, seed=42):
    """
    Build synthetic correlated BTC/ETH OHLCV data with a clear
    bull-bear-flat regime structure, so the strategy can detect something.
    """
    np.random.seed(seed)
    times = pd.date_range("2023-01-01", periods=n, freq="1h")

    # Construct returns with regime structure
    rets = np.zeros(n)
    a, b = n // 3, 2 * n // 3
    rets[:a] = np.random.randn(a) * 0.005 + 0.001       # bull drift
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001  # bear drift
    rets[b:] = np.random.randn(n - b) * 0.003           # flat/quiet

    btc_close = 20000 * np.cumprod(1 + rets)
    # ETH highly correlated with BTC plus its own noise
    eth_rets = 0.85 * rets + np.random.randn(n) * 0.003
    eth_close = 1500 * np.cumprod(1 + eth_rets)

    def ohlcv(close):
        # Simple synthetic OHLC: open ≈ prev close, high/low add noise
        open_p = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(close, open_p) * (1 + np.abs(np.random.randn(n) * 0.001))
        low = np.minimum(close, open_p) * (1 - np.abs(np.random.randn(n) * 0.001))
        return open_p, high, low

    btc_open, btc_high, btc_low = ohlcv(btc_close)
    eth_open, eth_high, eth_low = ohlcv(eth_close)

    df = pd.DataFrame({
        "datetime": times,
        "btc_open": btc_open, "btc_high": btc_high, "btc_low": btc_low, "btc_close": btc_close,
        "eth_open": eth_open, "eth_high": eth_high, "eth_low": eth_low, "eth_close": eth_close,
    })
    return df


def test_strategy_runs_to_completion():
    """Strategy + backtester pipeline should run without errors."""
    df = make_synthetic_btc_eth(n=2000)
    signals = run_eth_strategy(df)

    # Signals must be in {-1, 0, +1}
    assert signals["signal"].isin([-1, 0, 1]).all()

    # Some trades should happen (or be deliberately zero — both valid)
    # We just check the type & shape
    assert len(signals) == 2000
    assert "signal_reason" in signals.columns


def test_strategy_passes_to_backtester():
    """Signals from the strategy should be a valid input to the backtester."""
    df = make_synthetic_btc_eth(n=2000)
    signals = run_eth_strategy(df)

    # Rename ETH columns to standard names for backtester
    signals = signals.rename(columns={
        "eth_open": "open", "eth_high": "high",
        "eth_low": "low", "eth_close": "close",
    })

    result = run_backtest(signals, signal_col="signal", initial_capital=1000.0)

    # Result must have all standard fields
    assert result is not None
    assert "sharpe" in result.metrics
    assert "max_drawdown" in result.metrics
    assert "total_trades" in result.metrics

    print(f"Synthetic test result: {result.metrics['total_trades']} trades, "
          f"Sharpe={result.metrics['sharpe']:.3f}, "
          f"MDD={result.metrics['max_drawdown']*100:.2f}%")


def test_config_overrides_work():
    """Custom config should affect strategy behavior."""
    df = make_synthetic_btc_eth(n=2000)

    # Strict config: high Hurst threshold → fewer gates open → fewer trades
    strict = ETHStrategyConfig(hurst_threshold=0.9)
    sig_strict = run_eth_strategy(df, config=strict)

    # Permissive config: low Hurst threshold
    permissive = ETHStrategyConfig(hurst_threshold=0.0)
    sig_perm = run_eth_strategy(df, config=permissive)

    n_strict = (sig_strict["signal"] != 0).sum()
    n_perm = (sig_perm["signal"] != 0).sum()

    # Strict should produce fewer or equal active signals
    assert n_strict <= n_perm, f"Strict={n_strict}, Permissive={n_perm}"


def test_ewma_sigma_option():
    """EWMA-based CUSUM sigma should produce different signals than rolling."""
    df = make_synthetic_btc_eth(n=2000)

    cfg_rolling = ETHStrategyConfig(cusum_sigma_method="rolling")
    cfg_ewma = ETHStrategyConfig(cusum_sigma_method="ewma")

    sig_r = run_eth_strategy(df, config=cfg_rolling)
    sig_e = run_eth_strategy(df, config=cfg_ewma)

    # They should produce DIFFERENT regime detections
    diff_count = (sig_r["btc_regime"] != sig_e["btc_regime"]).sum()
    assert diff_count > 0, "EWMA sigma produced identical results to rolling"


def test_dfa_hurst_option():
    """DFA-based Hurst should produce non-NaN values where R/S also does."""
    df = make_synthetic_btc_eth(n=2000)

    cfg_dfa = ETHStrategyConfig(hurst_method="dfa")
    sig = run_eth_strategy(df, config=cfg_dfa)

    # DFA Hurst should be computed (not all NaN after warmup)
    valid = sig["eth_hurst"].dropna()
    assert len(valid) > 100, f"DFA Hurst only produced {len(valid)} valid values"


if __name__ == "__main__":
    import traceback
    tests = [
        test_strategy_runs_to_completion,
        test_strategy_passes_to_backtester,
        test_config_overrides_work,
        test_ewma_sigma_option,
        test_dfa_hurst_option,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
