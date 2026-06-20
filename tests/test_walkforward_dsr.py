"""
test_walkforward_dsr.py
=======================
Tests for walk-forward harness and Deflated Sharpe Ratio.
"""
import sys
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')

import numpy as np
import pandas as pd

from walkforward import (
    generate_folds, run_walkforward,
    WalkForwardResult, WalkForwardFold,
)
from deflated_sharpe import (
    deflated_sharpe, deflated_sharpe_from_walkforward,
    DSRResult, _expected_max_sharpe,
)


# ---------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------
def make_synthetic(n=1000, seed=42):
    np.random.seed(seed)
    times = pd.date_range("2023-01-01", periods=n, freq="1h")
    rets = np.random.randn(n) * 0.005
    close = 100 * np.cumprod(1 + rets)
    open_p = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(close, open_p) * 1.001
    low = np.minimum(close, open_p) * 0.999
    return pd.DataFrame({
        "datetime": times,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": np.abs(np.random.randn(n)) * 10,
    })


def simple_buy_strategy(train_df, test_df):
    """A trivial strategy that buys and holds on the test window."""
    out = test_df.copy()
    out["signal"] = 1
    return out


def random_strategy_factory(seed):
    """Return a strategy_fn that uses random signals (for noise-level testing)."""
    def strat(train_df, test_df):
        rng = np.random.default_rng(seed)
        out = test_df.copy()
        out["signal"] = rng.choice([-1, 0, 1], size=len(out))
        return out
    return strat


# =====================================================================
# Fold generation tests
# =====================================================================
def test_generate_folds_rolling_basic():
    """Rolling mode: fixed train size, non-overlapping test slices."""
    df = make_synthetic(n=1000)
    folds = generate_folds(df, train_bars=400, test_bars=100, mode="rolling")
    # First fold: train=[0:400], test=[400:500]
    assert folds[0] == (0, 400, 400, 500)
    # Test slices are non-overlapping: fold 2 test = [500:600]
    assert folds[1][2] == 500 and folds[1][3] == 600
    # All folds should have same train size in rolling mode
    train_sizes = [te - ts for (ts, te, _, _) in folds]
    assert all(sz == 400 for sz in train_sizes)


def test_generate_folds_expanding_grows():
    """Expanding mode: train_start fixed at 0; train grows over time."""
    df = make_synthetic(n=1000)
    folds = generate_folds(df, train_bars=400, test_bars=100, mode="expanding")
    # All train windows start at 0
    assert all(ts == 0 for (ts, _, _, _) in folds)
    # Train end grows with each fold
    train_ends = [te for (_, te, _, _) in folds]
    assert all(train_ends[i] < train_ends[i + 1] for i in range(len(train_ends) - 1))


def test_generate_folds_no_overlap_train_test():
    """In every fold, train ends before test starts."""
    df = make_synthetic(n=1500)
    folds = generate_folds(df, train_bars=300, test_bars=80, embargo_bars=10, mode="rolling")
    for ts, te, t2s, t2e in folds:
        assert te <= t2s, f"Train end {te} overlaps test start {t2s}"
        assert (t2s - te) >= 10, f"Embargo too small: {t2s - te} < 10"


def test_generate_folds_purge_applied():
    """Purge should reduce train end by purge_bars."""
    df = make_synthetic(n=1000)
    folds_no_purge = generate_folds(df, train_bars=400, test_bars=100, mode="rolling", purge_bars=0)
    folds_purged = generate_folds(df, train_bars=400, test_bars=100, mode="rolling", purge_bars=20)
    # Train ends should be exactly 20 earlier in the purged version
    for (_, te1, _, _), (_, te2, _, _) in zip(folds_no_purge, folds_purged):
        assert te1 - te2 == 20


def test_generate_folds_invalid_mode():
    """Invalid mode should raise."""
    df = make_synthetic(n=100)
    try:
        generate_folds(df, train_bars=50, test_bars=10, mode="bogus")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_generate_folds_too_little_data():
    """Insufficient data → fewer or no folds."""
    df = make_synthetic(n=100)
    folds = generate_folds(df, train_bars=200, test_bars=100, mode="rolling")
    assert len(folds) == 0


# =====================================================================
# Walk-forward runner tests
# =====================================================================
def test_run_walkforward_smoke():
    """Smoke test: walk-forward runs end-to-end."""
    df = make_synthetic(n=1000)
    result = run_walkforward(
        df, strategy_fn=simple_buy_strategy,
        train_bars=300, test_bars=100,
        mode="rolling", verbose=False,
    )
    assert isinstance(result, WalkForwardResult)
    assert len(result.folds) > 0
    assert len(result.pooled_returns) > 0


def test_run_walkforward_metrics_in_folds():
    """Each fold should have Sharpe, return, MDD, trades, win rate."""
    df = make_synthetic(n=1000)
    result = run_walkforward(
        df, strategy_fn=simple_buy_strategy,
        train_bars=300, test_bars=100,
        mode="rolling", verbose=False,
    )
    for fold in result.folds:
        assert isinstance(fold, WalkForwardFold)
        # Each metric should be a float / int
        assert isinstance(fold.sharpe, (int, float, np.floating))
        assert fold.total_trades >= 0


def test_walkforward_summary_stats():
    """summary_stats produces all expected keys."""
    df = make_synthetic(n=1000)
    result = run_walkforward(
        df, strategy_fn=simple_buy_strategy,
        train_bars=300, test_bars=100,
        mode="rolling", verbose=False,
    )
    s = result.summary_stats()
    for k in ("n_folds", "mean_sharpe", "std_sharpe", "min_sharpe",
              "max_sharpe", "mean_return", "positive_fold_rate"):
        assert k in s


def test_walkforward_report_renders():
    """report() returns a non-empty string."""
    df = make_synthetic(n=1000)
    result = run_walkforward(
        df, strategy_fn=simple_buy_strategy,
        train_bars=300, test_bars=100,
        mode="rolling", verbose=False,
    )
    rep = result.report()
    assert "WALK-FORWARD VALIDATION RESULT" in rep
    assert "Sharpe across folds" in rep


# =====================================================================
# DSR tests
# =====================================================================
def test_expected_max_sharpe_increases_with_n():
    """E[max SR] should increase with number of trials (selection bias grows)."""
    sm_2 = _expected_max_sharpe(2, sharpe_variance=0.1)
    sm_10 = _expected_max_sharpe(10, sharpe_variance=0.1)
    sm_100 = _expected_max_sharpe(100, sharpe_variance=0.1)
    assert sm_2 < sm_10 < sm_100, f"Got {sm_2}, {sm_10}, {sm_100}"


def test_expected_max_sharpe_zero_variance():
    """If all trials had identical Sharpe (variance=0), expected max = 0."""
    assert _expected_max_sharpe(10, sharpe_variance=0.0) == 0.0


def test_dsr_zero_returns_no_edge():
    """For zero-mean random returns from many random trials, DSR should be low."""
    np.random.seed(0)
    returns = np.random.randn(500) * 0.01  # zero mean
    trial_sharpes = np.random.randn(20)    # null-distributed sharpes
    result = deflated_sharpe(returns, trial_sharpes,
                              observed_sharpe=float(np.max(trial_sharpes)))
    # If our observed equals the max of random trials, DSR should be ~0.5 or lower
    assert 0.0 <= result.deflated_sharpe <= 1.0
    assert result.deflated_sharpe < 0.7, (
        f"DSR={result.deflated_sharpe} too high for the best of pure noise"
    )


def test_dsr_strong_edge_high_dsr():
    """A genuinely strong edge — observed Sharpe far above null — should have high DSR."""
    np.random.seed(0)
    # Returns with clear positive drift
    returns = np.random.randn(2000) * 0.005 + 0.002
    mean_r, std_r = float(np.mean(returns)), float(np.std(returns, ddof=1))
    sr_obs = (mean_r / std_r) * np.sqrt(8760)
    # Five "trials" all with low Sharpe — observed is clearly the outlier
    trial_sharpes = [0.1, 0.2, 0.15, 0.05, sr_obs]
    result = deflated_sharpe(returns, trial_sharpes, observed_sharpe=sr_obs)
    assert result.deflated_sharpe > 0.95, (
        f"DSR={result.deflated_sharpe} too low for a strong genuine edge"
    )


def test_dsr_result_str_renders():
    """DSRResult.__str__ should produce something readable."""
    np.random.seed(0)
    returns = np.random.randn(500) * 0.01
    trial_sharpes = np.random.randn(10)
    result = deflated_sharpe(returns, trial_sharpes,
                              observed_sharpe=float(np.max(trial_sharpes)))
    s = str(result)
    assert "Observed Sharpe" in s
    v = result.verdict()
    assert isinstance(v, str) and len(v) > 0


def test_dsr_too_few_trials_raises():
    """Need ≥ 2 trials."""
    try:
        deflated_sharpe(np.random.randn(100), trial_sharpes=[1.5],
                        observed_sharpe=1.5)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_dsr_too_few_returns_raises():
    """Need ≥ 4 returns for kurtosis."""
    try:
        deflated_sharpe(np.array([0.01, 0.02]),
                        trial_sharpes=[1.5, 1.0, 0.5],
                        observed_sharpe=1.5)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_dsr_from_walkforward():
    """End-to-end: walk-forward result feeds DSR."""
    df = make_synthetic(n=1500)
    wf = run_walkforward(
        df, strategy_fn=simple_buy_strategy,
        train_bars=300, test_bars=100,
        mode="rolling", verbose=False,
    )
    if len(wf.folds) >= 2:
        dsr = deflated_sharpe_from_walkforward(wf)
        assert isinstance(dsr, DSRResult)
        assert 0.0 <= dsr.deflated_sharpe <= 1.0


# =====================================================================
# Test runner
# =====================================================================
if __name__ == "__main__":
    import traceback
    tests = [
        test_generate_folds_rolling_basic,
        test_generate_folds_expanding_grows,
        test_generate_folds_no_overlap_train_test,
        test_generate_folds_purge_applied,
        test_generate_folds_invalid_mode,
        test_generate_folds_too_little_data,
        test_run_walkforward_smoke,
        test_run_walkforward_metrics_in_folds,
        test_walkforward_summary_stats,
        test_walkforward_report_renders,
        test_expected_max_sharpe_increases_with_n,
        test_expected_max_sharpe_zero_variance,
        test_dsr_zero_returns_no_edge,
        test_dsr_strong_edge_high_dsr,
        test_dsr_result_str_renders,
        test_dsr_too_few_trials_raises,
        test_dsr_too_few_returns_raises,
        test_dsr_from_walkforward,
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
