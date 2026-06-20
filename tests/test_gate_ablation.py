"""
test_gate_ablation.py
=====================
Tests for the ETH gate ablation experiment.
"""
import sys, os
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')

import numpy as np
import pandas as pd

from eth_regime_confirmation import ETHStrategyConfig, run_eth_strategy
from gate_ablation import (
    run_gate_ablation, GateAblationResult, GateConfigResult,
)


# =====================================================================
# Synthetic data
# =====================================================================
def make_synthetic_btc_eth(n=2000, seed=42):
    """Synthetic correlated BTC/ETH OHLCV with three regimes."""
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


# =====================================================================
# Strategy-level tests: ablation flags do something
# =====================================================================
def test_ablation_flags_default_to_true():
    """Default config should have all three gates enabled."""
    cfg = ETHStrategyConfig()
    assert cfg.enable_hurst_gate is True
    assert cfg.enable_correlation_gate is True
    assert cfg.enable_atr_gate is True


def test_disabling_a_gate_produces_more_or_equal_trades():
    """Disabling a gate (relaxing constraints) should produce ≥ trades vs all-on."""
    df = make_synthetic_btc_eth(n=600)
    base = ETHStrategyConfig()
    base_signals = run_eth_strategy(df, config=base)
    n_base = int((base_signals["signal"] != base_signals["signal"].shift(1)).sum())

    from dataclasses import replace
    for flag in ("enable_hurst_gate", "enable_correlation_gate", "enable_atr_gate"):
        cfg = replace(base, **{flag: False})
        ablated_signals = run_eth_strategy(df, config=cfg)
        n_ablated = int((ablated_signals["signal"] != ablated_signals["signal"].shift(1)).sum())
        # Relaxing a constraint should never produce fewer signal transitions
        assert n_ablated >= n_base, (
            f"{flag}=False produced fewer transitions ({n_ablated}) than baseline "
            f"({n_base}). Disabling a gate should be a relaxation."
        )


def test_disabling_all_gates_produces_most_trades():
    """All gates off should be the most permissive of all configurations."""
    df = make_synthetic_btc_eth(n=600)
    base = ETHStrategyConfig()
    from dataclasses import replace
    n_signals = {}
    for flags in [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ]:
        cfg = replace(
            base,
            enable_hurst_gate=flags[0],
            enable_correlation_gate=flags[1],
            enable_atr_gate=flags[2],
        )
        signals = run_eth_strategy(df, config=cfg)
        n_signals[flags] = int((signals["signal"] != signals["signal"].shift(1)).sum())

    # All gates off must give the most (or tied-most) signal transitions
    most_permissive = n_signals[(False, False, False)]
    for flags, n in n_signals.items():
        assert most_permissive >= n, (
            f"all-gates-off has {most_permissive} transitions, "
            f"but {flags} has {n} (more). Logic error in gate disabling."
        )


# =====================================================================
# Ablation harness tests
# =====================================================================
def test_run_gate_ablation_returns_all_five_configs():
    """The harness should produce 5 configurations."""
    df = make_synthetic_btc_eth(n=600)
    result = run_gate_ablation(df, verbose=False)
    assert isinstance(result, GateAblationResult)
    assert set(result.configs.keys()) == {"all_on", "hurst_off", "corr_off", "atr_off", "all_off"}


def test_baseline_uses_all_gates():
    """The 'all_on' config is the baseline and has all gates active."""
    df = make_synthetic_btc_eth(n=600)
    result = run_gate_ablation(df, verbose=False)
    base = result.baseline()
    assert base.enable_hurst is True
    assert base.enable_corr is True
    assert base.enable_atr is True


def test_ablated_configs_have_correct_gate_state():
    """Each config should have the correct gate state set."""
    df = make_synthetic_btc_eth(n=600)
    result = run_gate_ablation(df, verbose=False)
    expected = {
        "all_on":    (True, True, True),
        "hurst_off": (False, True, True),
        "corr_off":  (True, False, True),
        "atr_off":   (True, True, False),
        "all_off":   (False, False, False),
    }
    for name, (h, c, a) in expected.items():
        cfg = result.configs[name]
        assert cfg.enable_hurst == h, f"{name}: hurst mismatch"
        assert cfg.enable_corr == c, f"{name}: corr mismatch"
        assert cfg.enable_atr == a, f"{name}: atr mismatch"


def test_attribution_has_three_gates():
    """Attribution should report all three gates."""
    df = make_synthetic_btc_eth(n=600)
    result = run_gate_ablation(df, verbose=False)
    attr = result.attribution()
    assert set(attr.keys()) == {"hurst", "correlation", "atr"}
    for gate_name, deltas in attr.items():
        for key in ("delta_sharpe", "delta_return", "delta_mdd",
                     "delta_trades", "delta_win_rate"):
            assert key in deltas


def test_report_renders():
    """report() returns a string with key sections."""
    df = make_synthetic_btc_eth(n=600)
    result = run_gate_ablation(df, verbose=False)
    rep = result.report()
    assert "GATE ABLATION" in rep
    assert "PER-GATE ATTRIBUTION" in rep
    assert "VERDICT" in rep


def test_all_off_has_most_trades():
    """The 'all_off' config should produce ≥ trades than any other."""
    df = make_synthetic_btc_eth(n=600)
    result = run_gate_ablation(df, verbose=False)
    trades_off = result.configs["all_off"].total_trades
    for name, cfg in result.configs.items():
        if name == "all_off":
            continue
        assert trades_off >= cfg.total_trades, (
            f"'all_off' has {trades_off} trades but '{name}' has "
            f"{cfg.total_trades}. Gate-disable logic is wrong."
        )


if __name__ == "__main__":
    import traceback
    tests = [
        test_ablation_flags_default_to_true,
        test_disabling_a_gate_produces_more_or_equal_trades,
        test_disabling_all_gates_produces_most_trades,
        test_run_gate_ablation_returns_all_five_configs,
        test_baseline_uses_all_gates,
        test_ablated_configs_have_correct_gate_state,
        test_attribution_has_three_gates,
        test_report_renders,
        test_all_off_has_most_trades,
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
