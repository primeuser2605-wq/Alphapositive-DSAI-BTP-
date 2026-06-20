"""
test_hurst_ablation.py
======================
Tests for the Hurst (window, method) ablation experiment.

Data sizes are kept small because R/S Hurst is slow at large windows.
"""
import sys
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')

import numpy as np
import pandas as pd

from hurst_ablation import (
    run_hurst_ablation, HurstAblationResult, HurstConfigResult,
    _hurst_series_stats,
)


def make_synthetic_btc_eth(n=800, seed=42):
    """Small synthetic BTC/ETH for fast tests."""
    np.random.seed(seed)
    times = pd.date_range("2023-01-01", periods=n, freq="1h")
    rets = np.random.randn(n) * 0.005
    btc_close = 20000 * np.cumprod(1 + rets)
    eth_close = 1500 * np.cumprod(1 + 0.85 * rets + np.random.randn(n) * 0.003)
    def ohlcv(c):
        o = np.concatenate([[c[0]], c[:-1]])
        return o, np.maximum(c, o) * 1.001, np.minimum(c, o) * 0.999
    bo, bh, bl = ohlcv(btc_close)
    eo, eh, el = ohlcv(eth_close)
    return pd.DataFrame({
        "datetime": times,
        "btc_open": bo, "btc_high": bh, "btc_low": bl, "btc_close": btc_close,
        "eth_open": eo, "eth_high": eh, "eth_low": el, "eth_close": eth_close,
    })


# =====================================================================
# Hurst series statistics
# =====================================================================
def test_hurst_series_stats_returns_all_keys():
    """The stats dict must have all required keys."""
    df = make_synthetic_btc_eth(n=400)
    stats = _hurst_series_stats(df["eth_close"], window=120, method="rs")
    required = {"mean", "std", "min", "max", "nan_fraction", "above_05_fraction"}
    assert set(stats.keys()) == required


def test_hurst_series_stats_nan_fraction_bounded():
    """NaN fraction should be in [0, 1]."""
    df = make_synthetic_btc_eth(n=400)
    stats = _hurst_series_stats(df["eth_close"], window=120, method="rs")
    assert 0.0 <= stats["nan_fraction"] <= 1.0


def test_hurst_series_stats_dfa_works():
    """DFA method should produce sensible values."""
    df = make_synthetic_btc_eth(n=400)
    stats = _hurst_series_stats(df["eth_close"], window=120, method="dfa")
    # On a random series, both R/S and DFA should hover near 0.5 to 1.0
    assert 0.0 < stats["mean"] < 2.0


def test_hurst_unknown_method_raises():
    """Unknown method should raise."""
    df = make_synthetic_btc_eth(n=400)
    try:
        _hurst_series_stats(df["eth_close"], window=120, method="bogus")
        assert False, "Should have raised"
    except ValueError:
        pass


# =====================================================================
# Ablation experiment
# =====================================================================
def test_run_ablation_smoke():
    """Smoke test: ablation runs end-to-end with minimal grid."""
    df = make_synthetic_btc_eth(n=400)
    result = run_hurst_ablation(
        df, windows=(120,), methods=("rs", "dfa"), verbose=False,
    )
    assert isinstance(result, HurstAblationResult)
    assert len(result.results) == 2


def test_run_ablation_full_grid():
    """Full grid: 2 windows × 2 methods = 4 configurations."""
    df = make_synthetic_btc_eth(n=600)
    result = run_hurst_ablation(
        df, windows=(120, 250), methods=("rs", "dfa"), verbose=False,
    )
    assert len(result.results) == 4
    # All configs should be unique
    config_keys = {(r.window, r.method) for r in result.results}
    assert len(config_keys) == 4


def test_baseline_is_120_rs():
    """baseline() returns the (120, R/S) configuration when present."""
    df = make_synthetic_btc_eth(n=400)
    result = run_hurst_ablation(
        df, windows=(120, 250), methods=("rs", "dfa"), verbose=False,
    )
    baseline = result.baseline()
    assert baseline is not None
    assert baseline.window == 120
    assert baseline.method == "rs"


def test_to_dataframe_works():
    """to_dataframe produces a usable DataFrame."""
    df = make_synthetic_btc_eth(n=400)
    result = run_hurst_ablation(
        df, windows=(120,), methods=("rs", "dfa"), verbose=False,
    )
    out_df = result.to_dataframe()
    assert isinstance(out_df, pd.DataFrame)
    assert len(out_df) == 2
    for col in ("window", "method", "sharpe", "trades", "hurst_std"):
        assert col in out_df.columns


def test_report_renders():
    """report() returns a non-empty string with key sections."""
    df = make_synthetic_btc_eth(n=400)
    result = run_hurst_ablation(
        df, windows=(120,), methods=("rs", "dfa"), verbose=False,
    )
    rep = result.report()
    assert "HURST ABLATION" in rep
    assert "ESTIMATOR STATISTICS" in rep
    assert "VERDICT" in rep


def test_nan_fraction_decreases_with_size():
    """For a given window, NaN fraction should be roughly window/n."""
    df = make_synthetic_btc_eth(n=400)
    stats_120 = _hurst_series_stats(df["eth_close"], window=120, method="rs")
    # With 400 bars and window 120, NaN frac should be ~ 120/400 = 0.30
    assert 0.20 < stats_120["nan_fraction"] < 0.40


def test_strategy_metrics_present_in_results():
    """Every config result should have strategy metrics populated."""
    df = make_synthetic_btc_eth(n=600)
    result = run_hurst_ablation(
        df, windows=(120,), methods=("rs", "dfa"), verbose=False,
    )
    for r in result.results:
        assert isinstance(r.total_trades, int)
        assert isinstance(r.sharpe, float)
        assert isinstance(r.max_drawdown, float)


if __name__ == "__main__":
    import traceback
    tests = [
        test_hurst_series_stats_returns_all_keys,
        test_hurst_series_stats_nan_fraction_bounded,
        test_hurst_series_stats_dfa_works,
        test_hurst_unknown_method_raises,
        test_run_ablation_smoke,
        test_run_ablation_full_grid,
        test_baseline_is_120_rs,
        test_to_dataframe_works,
        test_report_renders,
        test_nan_fraction_decreases_with_size,
        test_strategy_metrics_present_in_results,
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
