"""
test_btc_strategy_smoke.py
==========================
Smoke test: the refactored BTC Q-learning strategy trains and emits signals.

These are NOT correctness tests against the original strategy's numbers ---
that requires the real 2020-2023 data. They test the contract:
- Training completes without error
- Q-table has correct shape
- Signals output is well-formed and consumable by the backtester
- The alternative log-utility reward function works
"""
import sys, os
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')

import numpy as np
import pandas as pd

from btc_qlearning import (
    BTCQLearningConfig, compute_features, get_state_index, state_size,
    TradingEnvironment, train_q_agent, policy_to_signals,
    run_btc_qlearning_strategy,
    N_ACTIONS, ACTION_HOLD, ACTION_LONG, ACTION_EXIT_LONG,
    ACTION_SHORT, ACTION_EXIT_SHORT,
)
from backtester import run_backtest


def make_synthetic_btc(n=2000, seed=42):
    """Build synthetic BTC OHLCV data with regime structure."""
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


def test_state_size_correct():
    """
    State space is n_pct (20) × n_position (3) × n_signal_bins^3 (27) = 1620.
    
    NOTE: The original report claims 540 states, which is an arithmetic error.
    The actual implementation (both original and refactored) has 1620 states.
    The discrepancy: 540 = 20 * 3 * 3 * 3 = only 4 dimensions, but the state
    space has 5 dimensions (pct, position, RSI, EMA, Aroon).
    """
    cfg = BTCQLearningConfig()
    expected = 20 * 3 * 3 * 3 * 3  # 1620
    ss = state_size(cfg)
    assert ss == expected, f"Expected {expected} states, got {ss}"


def test_state_index_in_range():
    """get_state_index should always return a value within [0, state_size)."""
    cfg = BTCQLearningConfig()
    ss = state_size(cfg)
    # Probe many extreme combinations
    for rsi in (-1, 0, 1):
        for ema in (-1, 0, 1):
            for aroon in (-1, 0, 1):
                for pos in (-1, 0, 1):
                    for pct in (-100, -5, 0, 5, 100, np.nan):
                        idx = get_state_index(rsi, ema, aroon, pos, pct, cfg)
                        assert 0 <= idx < ss, f"State {idx} out of range [0, {ss})"


def test_features_computed():
    """compute_features should add the four signal columns."""
    df = make_synthetic_btc(n=500)
    cfg = BTCQLearningConfig()
    out = compute_features(df, cfg)
    for col in ("rsi_signal", "ema_signal", "aroon_signal", "pct_change"):
        assert col in out.columns
        # Signals should be in {-1, 0, +1}
        if col != "pct_change":
            assert out[col].dropna().isin([-1, 0, 1]).all()


def test_environment_step_returns_correct_tuple():
    """TradingEnvironment.step should return (obs, reward, done)."""
    df = make_synthetic_btc(n=500)
    cfg = BTCQLearningConfig()
    feat = compute_features(df, cfg)
    feat = feat.dropna(subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"])

    env = TradingEnvironment(
        feat["close"].values,
        {
            "rsi_signal": feat["rsi_signal"].values,
            "ema_signal": feat["ema_signal"].values,
            "aroon_signal": feat["aroon_signal"].values,
            "pct_change": feat["pct_change"].values,
        },
        cfg,
    )
    obs = env.reset()
    assert len(obs) == 5  # aroon, rsi, ema, position, pct
    next_obs, reward, done = env.step(ACTION_HOLD)
    assert len(next_obs) == 5
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(done, (bool, np.bool_))


def test_short_training_runs():
    """A very short training run should complete without errors."""
    df = make_synthetic_btc(n=500)
    cfg = BTCQLearningConfig(n_episodes=3)  # short
    feat = compute_features(df, cfg).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"])

    env = TradingEnvironment(
        feat["close"].values,
        {
            "rsi_signal": feat["rsi_signal"].values,
            "ema_signal": feat["ema_signal"].values,
            "aroon_signal": feat["aroon_signal"].values,
            "pct_change": feat["pct_change"].values,
        },
        cfg,
    )
    Q = train_q_agent(env, cfg, verbose=False)
    assert Q.shape == (state_size(cfg), N_ACTIONS)
    # Q should not be all zeros after training (something was learned)
    assert (np.abs(Q) > 0).sum() > 0


def test_policy_to_signals_produces_valid_signals():
    """The greedy policy applied to features should produce a valid signal column."""
    df = make_synthetic_btc(n=500)
    cfg = BTCQLearningConfig(n_episodes=5)
    feat = compute_features(df, cfg).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"])
    
    env = TradingEnvironment(
        feat["close"].values,
        {
            "rsi_signal": feat["rsi_signal"].values,
            "ema_signal": feat["ema_signal"].values,
            "aroon_signal": feat["aroon_signal"].values,
            "pct_change": feat["pct_change"].values,
        },
        cfg,
    )
    Q = train_q_agent(env, cfg, verbose=False)
    signals = policy_to_signals(feat, Q, cfg)

    assert len(signals) == len(feat)
    assert signals.isin([-1, 0, 1]).all()


def test_end_to_end_runs():
    """run_btc_qlearning_strategy should produce signals + Q-table without errors."""
    df = make_synthetic_btc(n=1000)
    cfg = BTCQLearningConfig(n_episodes=5)
    # Train on first 70%, test on last 30%
    train_end_str = str(df["datetime"].iloc[700])
    
    df_with_signals, Q = run_btc_qlearning_strategy(
        df, train_end_date=train_end_str, config=cfg, verbose=False
    )
    assert "signal" in df_with_signals.columns
    assert Q.shape == (state_size(cfg), N_ACTIONS)


def test_signals_feed_backtester():
    """End-to-end: strategy signals → backtester → valid result."""
    df = make_synthetic_btc(n=1000)
    cfg = BTCQLearningConfig(n_episodes=5)
    train_end_str = str(df["datetime"].iloc[700])
    
    df_with_signals, _ = run_btc_qlearning_strategy(
        df, train_end_date=train_end_str, config=cfg, verbose=False
    )
    result = run_backtest(df_with_signals, signal_col="signal",
                          initial_capital=1000.0)
    assert result is not None
    assert "sharpe" in result.metrics
    print(f"BTC synthetic test: {result.metrics['total_trades']} trades, "
          f"Sharpe={result.metrics['sharpe']:.3f}")


def test_log_utility_reward_works():
    """The Moody-Saffell log-utility reward should also train successfully."""
    df = make_synthetic_btc(n=500)
    cfg = BTCQLearningConfig(n_episodes=5, reward_type="log_utility")
    feat = compute_features(df, cfg).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"])
    env = TradingEnvironment(
        feat["close"].values,
        {
            "rsi_signal": feat["rsi_signal"].values,
            "ema_signal": feat["ema_signal"].values,
            "aroon_signal": feat["aroon_signal"].values,
            "pct_change": feat["pct_change"].values,
        },
        cfg,
    )
    Q = train_q_agent(env, cfg, verbose=False)
    # The Q-table should not be dominated by 'always long' if the reward works
    # We just check training completed
    assert Q.shape == (state_size(cfg), N_ACTIONS)


def test_original_vs_log_utility_produce_different_policies():
    """The two reward functions should produce different learned policies."""
    df = make_synthetic_btc(n=1500)
    cfg_orig = BTCQLearningConfig(n_episodes=20, reward_type="original")
    cfg_log = BTCQLearningConfig(n_episodes=20, reward_type="log_utility")

    feat = compute_features(df, cfg_orig).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"])
    feat_dict = {
        "rsi_signal": feat["rsi_signal"].values,
        "ema_signal": feat["ema_signal"].values,
        "aroon_signal": feat["aroon_signal"].values,
        "pct_change": feat["pct_change"].values,
    }
    env_orig = TradingEnvironment(feat["close"].values, feat_dict, cfg_orig)
    env_log = TradingEnvironment(feat["close"].values, feat_dict, cfg_log)

    Q_orig = train_q_agent(env_orig, cfg_orig)
    Q_log = train_q_agent(env_log, cfg_log)

    # Compare best actions in each state where both have visited
    visited_orig = (np.abs(Q_orig) > 0).any(axis=1)
    visited_log = (np.abs(Q_log) > 0).any(axis=1)
    both = visited_orig & visited_log
    if both.sum() < 10:
        # Not enough overlap to make a meaningful comparison
        return

    best_orig = Q_orig[both].argmax(axis=1)
    best_log = Q_log[both].argmax(axis=1)
    # They should differ in at least some states
    assert (best_orig != best_log).sum() > 0, \
        "Different reward functions produced identical greedy policies"


if __name__ == "__main__":
    import traceback
    tests = [
        test_state_size_correct,
        test_state_index_in_range,
        test_features_computed,
        test_environment_step_returns_correct_tuple,
        test_short_training_runs,
        test_policy_to_signals_produces_valid_signals,
        test_end_to_end_runs,
        test_signals_feed_backtester,
        test_log_utility_reward_works,
        test_original_vs_log_utility_produce_different_policies,
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
