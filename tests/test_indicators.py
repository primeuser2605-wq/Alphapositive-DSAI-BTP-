"""
test_indicators.py
==================
Tests for the indicators module. Validates correctness against simple cases
where the expected output is known analytically.
"""
import sys
sys.path.insert(0, '/home/claude/quant_portfolio/src')

import numpy as np
import pandas as pd

from indicators import (
    rsi, atr, ema, bollinger_bands, aroon, supertrend,
    rolling_hurst, rolling_dfa_hurst, kalman_filter_1d, cusum_regime,
    rolling_correlation,
)


def make_series(values, freq="1h"):
    """Helper to build a small Series."""
    idx = pd.date_range("2023-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def test_rsi_constant_price():
    """RSI of a constant series should be NaN (no gains, no losses)."""
    s = make_series([100.0] * 30)
    r = rsi(s, window=14)
    # Last values: avg_gain=0, avg_loss=0, RS=0/0=NaN → RSI=NaN
    assert pd.isna(r.iloc[-1]) or r.iloc[-1] == 50.0


def test_rsi_monotonic_rise():
    """RSI of a monotonic uptrend → 100 (all gains, no losses)."""
    s = make_series([100 + i for i in range(30)])
    r = rsi(s, window=14)
    assert r.iloc[-1] == 100.0


def test_rsi_monotonic_fall():
    """RSI of monotonic downtrend → 0."""
    s = make_series([100 - i for i in range(30)])
    r = rsi(s, window=14)
    assert r.iloc[-1] == 0.0


def test_atr_constant_range():
    """ATR for constant H-L should equal H-L."""
    n = 30
    high = make_series([102.0] * n)
    low = make_series([100.0] * n)
    close = make_series([101.0] * n)
    a = atr(high, low, close, window=14)
    # After warmup, ATR should be 2.0
    assert abs(a.iloc[-1] - 2.0) < 1e-6


def test_ema_recent_weighted_more():
    """After a sustained price level change, EMA converges to the new level."""
    s = make_series([100.0] * 20 + [110.0] * 30)
    e = ema(s, span=5)
    # After 30 bars at 110, EMA should be very close to 110
    assert abs(e.iloc[-1] - 110.0) < 0.5
    # And during the transition, EMA is between 100 and 110
    assert 100 < e.iloc[22] < 110


def test_bollinger_constant():
    """Constant price: middle = constant, upper = lower = constant (std=0)."""
    s = make_series([100.0] * 30)
    m, u, l = bollinger_bands(s, window=20, num_std=2)
    assert abs(m.iloc[-1] - 100.0) < 1e-6
    assert abs(u.iloc[-1] - m.iloc[-1]) < 1e-6
    assert abs(l.iloc[-1] - m.iloc[-1]) < 1e-6


def test_kalman_smoothing():
    """Kalman should reduce noise variance."""
    np.random.seed(42)
    true_signal = np.linspace(100, 110, 200)
    noise = np.random.randn(200) * 2
    observed = pd.Series(true_signal + noise)
    
    filtered = kalman_filter_1d(observed, observation_covariance=1.0, transition_covariance=0.01)
    
    # Filtered should be closer to true signal than raw observations
    raw_mse = np.mean((observed.values - true_signal) ** 2)
    filtered_mse = np.mean((filtered.values - true_signal) ** 2)
    assert filtered_mse < raw_mse, \
        f"Filter didn't reduce MSE: raw={raw_mse:.3f}, filtered={filtered_mse:.3f}"


def test_kalman_causal():
    """Kalman value at t should not change when adding future data."""
    s = make_series([100.0 + i * 0.5 + np.random.randn() for i in range(100)])
    f_full = kalman_filter_1d(s)
    f_partial = kalman_filter_1d(s.iloc[:50])
    
    # At index 49, both should give identical values (causal property)
    assert abs(f_full.iloc[49] - f_partial.iloc[49]) < 1e-9, \
        "Kalman filter is not causal!"


def test_cusum_no_trend_quiet():
    """CUSUM on constant prices (no trend) should mostly stay in neutral regime."""
    n = 200
    s = make_series([100.0] * n)
    ref = s.copy()
    _, _, regime = cusum_regime(s, ref)
    # Mostly neutral
    counts = regime.value_counts()
    assert counts.get("neutral", 0) >= 0.9 * n


def test_cusum_detects_uptrend():
    """CUSUM on a clear uptrend with lagging reference should detect bullish regime."""
    n = 200
    np.random.seed(0)
    trend = np.cumsum(np.random.randn(n) * 0.1 + 0.1)  # drift up
    prices = pd.Series(100 + trend, index=pd.date_range("2023-01-01", periods=n, freq="1h"))
    # Use Kalman as reference
    ref = kalman_filter_1d(prices, observation_covariance=0.5, transition_covariance=0.001)
    _, _, regime = cusum_regime(prices, ref, sigma_window=5)
    # Should detect SOME bullish bars (not asking all)
    assert (regime == "bullish").sum() > 5


def test_hurst_random_walk_returns():
    """Hurst of random walk RETURNS should be approximately 0.5."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(2000))
    h = rolling_hurst(returns, window=500).dropna()
    # With 500-bar window on returns, R/S gives H near 0.5
    assert 0.35 < h.mean() < 0.65, f"Hurst of random walk returns: {h.mean():.3f}"


def test_dfa_hurst_random_walk_returns():
    """DFA Hurst of random walk RETURNS should be approximately 0.5."""
    np.random.seed(42)
    # Use 800 pts, window 300: ~500 windows, takes ~20s.
    # Full 2000/500 like the R/S test would take ~100s — DFA has Python overhead per window.
    returns = pd.Series(np.random.randn(800))
    h = rolling_dfa_hurst(returns, window=300).dropna()
    assert 0.35 < h.mean() < 0.65, f"DFA Hurst of random walk returns: {h.mean():.3f}"


def test_supertrend_direction_in_uptrend():
    """In a clean uptrend, supertrend direction should be predominantly +1."""
    n = 100
    high = make_series([100 + i for i in range(n)])
    low = make_series([99 + i for i in range(n)])
    close = make_series([99.5 + i for i in range(n)])
    _, direction = supertrend(high, low, close, period=12, multiplier=2.5)
    # After warmup, direction should be mostly +1
    assert (direction.iloc[20:] == 1).sum() > 0.8 * (n - 20)


def test_rolling_correlation_perfect():
    """Perfectly correlated series should have correlation 1."""
    a = make_series([100 + i for i in range(50)])
    b = a.copy() * 2 + 5  # linear transformation
    c = rolling_correlation(a, b, window=20)
    # Should be very close to 1
    assert abs(c.iloc[-1] - 1.0) < 1e-6


if __name__ == "__main__":
    import traceback
    tests = [
        test_rsi_constant_price, test_rsi_monotonic_rise, test_rsi_monotonic_fall,
        test_atr_constant_range, test_ema_recent_weighted_more,
        test_bollinger_constant, test_kalman_smoothing, test_kalman_causal,
        test_cusum_no_trend_quiet, test_cusum_detects_uptrend,
        test_hurst_random_walk_returns, test_dfa_hurst_random_walk_returns,
        test_supertrend_direction_in_uptrend, test_rolling_correlation_perfect,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
