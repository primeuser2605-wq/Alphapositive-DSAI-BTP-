"""
test_reward_comparison.py
=========================
Tests for the BTC reward function comparison experiment.
"""
import sys
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')

import numpy as np
import pandas as pd

from btc_qlearning import BTCQLearningConfig
from reward_comparison import (
    run_reward_comparison,
    RewardComparisonResult,
    RewardSeedResult,
)


def make_synthetic_btc(n=1500, seed=42):
    """Small synthetic BTC OHLCV for fast tests."""
    np.random.seed(seed)
    times = pd.date_range("2023-01-01", periods=n, freq="1h")
    rets = np.zeros(n)
    a, b = n // 3, 2 * n // 3
    rets[:a] = np.random.randn(a) * 0.005 + 0.001
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001
    rets[b:] = np.random.randn(n - b) * 0.003
    close = 20000 * np.cumprod(1 + rets)
    open_p = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(close, open_p) * (1 + np.abs(np.random.randn(n) * 0.001))
    low = np.minimum(close, open_p) * (1 - np.abs(np.random.randn(n) * 0.001))
    return pd.DataFrame({
        "datetime": times,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": np.abs(np.random.randn(n)) * 100,
    })


# =====================================================================
# Container behavior
# =====================================================================
def test_long_fraction_zero_trades():
    """RewardSeedResult.long_fraction should be 0 when there are no trades."""
    r = RewardSeedResult(
        reward_type="original", seed=0,
        total_trades=0, long_trades=0, short_trades=0,
        win_rate=0.0, sharpe=0.0, sortino=0.0,
        total_return=0.0, max_drawdown=0.0,
    )
    assert r.long_fraction == 0.0


def test_long_fraction_basic():
    """long_fraction = long_trades / total_trades."""
    r = RewardSeedResult(
        reward_type="original", seed=0,
        total_trades=10, long_trades=7, short_trades=3,
        win_rate=0.6, sharpe=1.0, sortino=1.2,
        total_return=0.1, max_drawdown=-0.1,
    )
    assert abs(r.long_fraction - 0.7) < 1e-9


# =====================================================================
# Experiment behavior
# =====================================================================
def test_run_smoke():
    """Smoke test: experiment runs end-to-end with small parameters."""
    df = make_synthetic_btc(n=600)
    cfg = BTCQLearningConfig(n_episodes=3)
    result = run_reward_comparison(
        df, n_seeds=2, base_config=cfg, verbose=False,
    )
    assert isinstance(result, RewardComparisonResult)
    assert set(result.by_seed.keys()) == {"original", "log_utility"}
    assert len(result.by_seed["original"]) == 2
    assert len(result.by_seed["log_utility"]) == 2


def test_aggregate_has_all_metrics():
    """aggregate() returns mean/std/min/max for each metric."""
    df = make_synthetic_btc(n=600)
    cfg = BTCQLearningConfig(n_episodes=3)
    result = run_reward_comparison(
        df, n_seeds=2, base_config=cfg, verbose=False,
    )
    agg = result.aggregate()
    for reward in ("original", "log_utility"):
        for metric in ("sharpe", "sortino", "total_return", "max_drawdown",
                       "win_rate", "total_trades", "long_fraction"):
            assert metric in agg[reward]
            for stat in ("mean", "std", "min", "max"):
                assert stat in agg[reward][metric]


def test_report_renders():
    """report() returns a non-empty string with key sections."""
    df = make_synthetic_btc(n=600)
    cfg = BTCQLearningConfig(n_episodes=3)
    result = run_reward_comparison(
        df, n_seeds=2, base_config=cfg, verbose=False,
    )
    rep = result.report()
    assert "REWARD FUNCTION COMPARISON" in rep
    assert "L5 SYMPTOM" in rep
    assert "VERDICT" in rep


def test_seeds_use_per_seed_config():
    """
    Different seeds must propagate to different Q-tables.

    We don't test metric divergence because at this scale of synthetic data,
    different Q-tables can still produce the same greedy policy (and thus
    the same metrics) — for instance when all action values are negative and
    the argmax just picks the 'least bad' action. The seed propagation
    invariant is that the *Q-tables* differ.
    """
    from dataclasses import replace
    from btc_qlearning import (
        BTCQLearningConfig, compute_features,
        TradingEnvironment, train_q_agent,
    )
    df = make_synthetic_btc(n=1500)
    cfg = BTCQLearningConfig(n_episodes=30, reward_type="original")
    feat = compute_features(df, cfg).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"]
    ).reset_index(drop=True)
    train_feat = feat.iloc[:int(0.7 * len(feat))]

    Qs = []
    for seed in (0, 1, 2):
        cseed = replace(cfg, seed=seed)
        env = TradingEnvironment(
            train_feat["close"].values,
            {k: train_feat[k].values for k in
             ("rsi_signal", "ema_signal", "aroon_signal", "pct_change")},
            cseed,
        )
        Qs.append(train_q_agent(env, cseed, verbose=False))

    assert not np.array_equal(Qs[0], Qs[1]), \
        "Seed 0 and seed 1 produced identical Q-tables — seed propagation regressed."


def test_results_have_long_short_breakdown():
    """Each seed result should record long and short counts separately."""
    df = make_synthetic_btc(n=600)
    cfg = BTCQLearningConfig(n_episodes=3)
    result = run_reward_comparison(
        df, n_seeds=2, base_config=cfg, verbose=False,
    )
    for reward_results in result.by_seed.values():
        for r in reward_results:
            assert r.long_trades + r.short_trades == r.total_trades


if __name__ == "__main__":
    import traceback
    tests = [
        test_long_fraction_zero_trades,
        test_long_fraction_basic,
        test_run_smoke,
        test_aggregate_has_all_metrics,
        test_report_renders,
        test_seeds_use_per_seed_config,
        test_results_have_long_short_breakdown,
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
