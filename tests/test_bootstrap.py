"""
test_bootstrap.py
=================
Tests for the Politis-Romano stationary bootstrap.
"""
import sys
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')

import numpy as np
import pandas as pd

from bootstrap import (
    stationary_bootstrap_indices, bootstrap_metric_ci, bootstrap_all_metrics,
    summarize_bootstrap, BootstrapCI,
)


def test_indices_correct_length():
    """Resample indices should have the requested length."""
    idx = stationary_bootstrap_indices(n=100, mean_block_length=10, seed=42)
    assert len(idx) == 100


def test_indices_in_range():
    """All indices should be in [0, n)."""
    idx = stationary_bootstrap_indices(n=100, mean_block_length=10, seed=42)
    assert idx.min() >= 0
    assert idx.max() < 100


def test_indices_reproducible():
    """Same seed → same indices."""
    idx1 = stationary_bootstrap_indices(n=100, mean_block_length=10, seed=42)
    idx2 = stationary_bootstrap_indices(n=100, mean_block_length=10, seed=42)
    assert np.array_equal(idx1, idx2)


def test_indices_different_seeds_differ():
    """Different seeds should produce different resamples."""
    idx1 = stationary_bootstrap_indices(n=100, mean_block_length=10, seed=1)
    idx2 = stationary_bootstrap_indices(n=100, mean_block_length=10, seed=2)
    assert not np.array_equal(idx1, idx2)


def test_block_length_short_means_iid():
    """Block length 1 should be effectively IID resampling."""
    # With mean_block_length=1 (p=1), blocks are almost always length 1
    idx = stationary_bootstrap_indices(n=1000, mean_block_length=1.0, seed=42)
    # Check: large variance in consecutive differences (since each is independent)
    diffs = np.diff(idx)
    # In IID resampling, consecutive indices are roughly uniform → diff std ≈ n/sqrt(12) ≈ 290
    assert np.std(diffs) > 100, f"Expected high variance in IID, got std={np.std(diffs):.1f}"


def test_block_length_long_preserves_blocks():
    """Long mean block length should preserve consecutive-index blocks."""
    # With mean_block_length=100 on n=1000, we expect about 10 blocks
    idx = stationary_bootstrap_indices(n=1000, mean_block_length=100.0, seed=42)
    diffs = np.diff(idx)
    # In block-mode, many consecutive diffs should be exactly 1
    fraction_one = (diffs == 1).mean()
    assert fraction_one > 0.5, f"Expected many +1 diffs in block mode, got {fraction_one:.2f}"


def test_sharpe_ci_contains_point_estimate():
    """The bootstrap CI should bracket the point estimate (in expectation)."""
    np.random.seed(0)
    rets = np.random.randn(100) * 0.02 + 0.001
    ci = bootstrap_metric_ci(rets, metric="sharpe", n_resamples=2000,
                              annualization=252, seed=42)
    assert ci.lower <= ci.point_estimate <= ci.upper


def test_sharpe_ci_narrows_with_more_data():
    """CI should be tighter with more data."""
    np.random.seed(0)
    rets_short = np.random.randn(50) * 0.02 + 0.001
    rets_long = np.random.randn(500) * 0.02 + 0.001
    ci_short = bootstrap_metric_ci(rets_short, metric="sharpe",
                                     n_resamples=2000, annualization=252, seed=42)
    ci_long = bootstrap_metric_ci(rets_long, metric="sharpe",
                                    n_resamples=2000, annualization=252, seed=42)
    width_short = ci_short.upper - ci_short.lower
    width_long = ci_long.upper - ci_long.lower
    assert width_long < width_short, \
        f"Long CI ({width_long:.3f}) should be narrower than short CI ({width_short:.3f})"


def test_all_metrics_returned():
    """bootstrap_all_metrics should return all standard metrics."""
    np.random.seed(0)
    rets = np.random.randn(100) * 0.02 + 0.001
    cis = bootstrap_all_metrics(rets, n_resamples=1000, annualization=252, seed=42)
    expected = {"sharpe", "sortino", "total_return", "max_drawdown", "win_rate"}
    assert set(cis.keys()) == expected
    for name, ci in cis.items():
        assert isinstance(ci, BootstrapCI)


def test_summarize_produces_string():
    """summarize_bootstrap returns a non-empty multiline string."""
    np.random.seed(0)
    rets = np.random.randn(100) * 0.02
    cis = bootstrap_all_metrics(rets, n_resamples=500, annualization=252, seed=42)
    s = summarize_bootstrap(cis)
    assert isinstance(s, str)
    assert "sharpe" in s.lower()


def test_reproducibility():
    """Same seed → same CI."""
    rets = np.random.RandomState(42).randn(100) * 0.02
    ci1 = bootstrap_metric_ci(rets, metric="sharpe", n_resamples=1000, seed=42)
    ci2 = bootstrap_metric_ci(rets, metric="sharpe", n_resamples=1000, seed=42)
    assert ci1.lower == ci2.lower
    assert ci1.upper == ci2.upper


def test_too_few_returns_raises():
    """Should raise on insufficient data."""
    try:
        bootstrap_metric_ci(np.array([0.01]), metric="sharpe", n_resamples=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import traceback
    tests = [
        test_indices_correct_length,
        test_indices_in_range,
        test_indices_reproducible,
        test_indices_different_seeds_differ,
        test_block_length_short_means_iid,
        test_block_length_long_preserves_blocks,
        test_sharpe_ci_contains_point_estimate,
        test_sharpe_ci_narrows_with_more_data,
        test_all_metrics_returned,
        test_summarize_produces_string,
        test_reproducibility,
        test_too_few_returns_raises,
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
