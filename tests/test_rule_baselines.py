"""
test_rule_baselines.py
======================
Tests for the rule-based BTC baselines and the RL-vs-rule experiment.
"""
import sys, os
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')

import numpy as np
import pandas as pd

from btc_rule_based import (
    rule_momentum_confluence,
    rule_majority_vote,
    rule_mean_reversion,
    apply_rule,
    BASELINES,
)
from rl_vs_rule_experiment import (
    run_comparison_experiment,
    StrategyMetrics,
    ComparisonResult,
)
from btc_qlearning import BTCQLearningConfig


def make_synthetic_btc(n=2000, seed=42):
    """Same synthetic-data helper used in test_btc_strategy_smoke."""
    np.random.seed(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="1h")
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
# Rule contract tests
# =====================================================================
def test_momentum_confluence_produces_valid_signals():
    """Rule should produce a signal column in {-1, 0, +1}."""
    df = make_synthetic_btc(n=500)
    out = rule_momentum_confluence(df)
    assert "signal" in out.columns
    assert out["signal"].isin([-1, 0, 1]).all()
    assert len(out) == len(df)


def test_majority_vote_produces_valid_signals():
    df = make_synthetic_btc(n=500)
    out = rule_majority_vote(df)
    assert "signal" in out.columns
    assert out["signal"].isin([-1, 0, 1]).all()


def test_mean_reversion_produces_valid_signals():
    df = make_synthetic_btc(n=500)
    out = rule_mean_reversion(df)
    assert "signal" in out.columns
    assert out["signal"].isin([-1, 0, 1]).all()


def test_apply_rule_dispatches_correctly():
    """apply_rule('momentum_confluence') should produce same output as direct call."""
    df = make_synthetic_btc(n=500)
    direct = rule_momentum_confluence(df)
    via_dispatch = apply_rule(df, "momentum_confluence")
    assert (direct["signal"] == via_dispatch["signal"]).all()


def test_apply_rule_unknown_raises():
    """Unknown rule name should raise."""
    df = make_synthetic_btc(n=100)
    try:
        apply_rule(df, "nonexistent_rule")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_rules_disagree_on_some_bars():
    """Different rules should make different decisions on at least some bars."""
    df = make_synthetic_btc(n=1000)
    momentum = rule_momentum_confluence(df)
    mean_rev = rule_mean_reversion(df)
    # These are opposite philosophies; they MUST disagree somewhere
    disagreements = (momentum["signal"] != mean_rev["signal"]).sum()
    assert disagreements > 10, f"Rules agree everywhere ({disagreements} disagreements)"


def test_baseline_registry_has_entries():
    """The BASELINES registry should be populated."""
    assert len(BASELINES) >= 3
    for name, info in BASELINES.items():
        assert hasattr(info, "fn") and callable(info.fn)


# =====================================================================
# Experiment harness tests
# =====================================================================
def test_strategy_metrics_from_result():
    """StrategyMetrics.from_result should produce a complete object."""
    from backtester import run_backtest
    df = make_synthetic_btc(n=500)
    out = rule_majority_vote(df)
    result = run_backtest(out, signal_col="signal", initial_capital=1000.0)
    m = StrategyMetrics.from_result("test", result)
    assert m.name == "test"
    assert isinstance(m.sharpe, float)
    assert isinstance(m.total_trades, int)


def test_run_comparison_experiment_smoke():
    """Smoke test: experiment runs end-to-end with small parameters."""
    df = make_synthetic_btc(n=1500)
    config = BTCQLearningConfig(n_episodes=3)  # very short for speed
    result = run_comparison_experiment(
        df,
        train_frac=0.7,
        n_seeds=2,  # minimal
        config=config,
        verbose=False,
    )
    assert isinstance(result, ComparisonResult)
    assert len(result.rule_metrics) == 3  # all three baselines
    assert len(result.rl_metrics_by_seed) == 2  # both seeds


def test_comparison_result_verdict_is_string():
    """verdict() should produce a non-empty string under all branches."""
    df = make_synthetic_btc(n=1200)
    config = BTCQLearningConfig(n_episodes=2)
    result = run_comparison_experiment(df, train_frac=0.7, n_seeds=2,
                                        config=config, verbose=False)
    v = result.verdict()
    assert isinstance(v, str) and len(v) > 0
    # Verdict should be one of the three categories
    assert any(kw in v for kw in ["ADDS SIGNAL", "TIES WITH RULES", "UNDERPERFORMS"])


def test_comparison_report_renders():
    """report() should render a multi-line string with key sections."""
    df = make_synthetic_btc(n=1200)
    config = BTCQLearningConfig(n_episodes=2)
    result = run_comparison_experiment(df, train_frac=0.7, n_seeds=2,
                                        config=config, verbose=False)
    rep = result.report()
    assert "RULE-BASED BASELINES" in rep
    assert "TABULAR Q-LEARNING" in rep


def test_rl_summary_has_all_metrics():
    """rl_summary() should return mean/std/min/max for each metric."""
    df = make_synthetic_btc(n=1200)
    config = BTCQLearningConfig(n_episodes=2)
    result = run_comparison_experiment(df, train_frac=0.7, n_seeds=3,
                                        config=config, verbose=False)
    s = result.rl_summary()
    for metric in ("sharpe", "sortino", "total_return", "max_drawdown",
                   "total_trades", "win_rate"):
        assert metric in s
        for stat in ("mean", "std", "min", "max"):
            assert stat in s[metric]


def test_seeds_produce_different_q_tables():
    """
    Regression test: different RL seeds must train to different Q-tables.
    
    Earlier bug: train_q_agent reads config.seed internally, but the
    experiment harness only set np.random.seed(seed) externally. The
    internal call overrode our seed, so all seeds gave identical Q-tables.
    
    We test Q-table difference rather than metric difference because on
    some data distributions, different Q-tables can produce the same
    greedy policy (and thus same metrics) even though learning was
    legitimately seed-varying.
    
    This test guards against regression of the seed-propagation bug.
    """
    from dataclasses import replace
    from btc_qlearning import (
        BTCQLearningConfig, compute_features,
        TradingEnvironment, train_q_agent,
    )

    df = make_synthetic_btc(n=2000)
    config = BTCQLearningConfig(n_episodes=30)
    feat = compute_features(df, config).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"]
    ).reset_index(drop=True)
    train_feat = feat.iloc[:int(len(df) * 0.7)]

    Qs = []
    for seed in (0, 1, 2):
        seeded = replace(config, seed=seed)
        env = TradingEnvironment(
            train_feat["close"].values,
            {
                "rsi_signal": train_feat["rsi_signal"].values,
                "ema_signal": train_feat["ema_signal"].values,
                "aroon_signal": train_feat["aroon_signal"].values,
                "pct_change": train_feat["pct_change"].values,
            },
            seeded,
        )
        Qs.append(train_q_agent(env, seeded, verbose=False))

    # Q-tables MUST differ — verify on at least one pair
    assert not np.array_equal(Qs[0], Qs[1]), (
        "Seed 0 and seed 1 produced identical Q-tables. "
        "Seed propagation bug regressed."
    )


if __name__ == "__main__":
    import traceback
    tests = [
        test_momentum_confluence_produces_valid_signals,
        test_majority_vote_produces_valid_signals,
        test_mean_reversion_produces_valid_signals,
        test_apply_rule_dispatches_correctly,
        test_apply_rule_unknown_raises,
        test_rules_disagree_on_some_bars,
        test_baseline_registry_has_entries,
        test_strategy_metrics_from_result,
        test_run_comparison_experiment_smoke,
        test_comparison_result_verdict_is_string,
        test_comparison_report_renders,
        test_rl_summary_has_all_metrics,
        test_seeds_produce_different_q_tables,
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
