"""
test_backtester.py
==================
Tests for the in-house backtester. Run with: pytest test_backtester.py

These tests validate:
1. No-trade case (signal always 0) → capital unchanged
2. Single long trade in flat market → small fee loss
3. Single long trade in trending market → profit
4. Buy-and-hold equivalence (one trade from start to end)
5. Determinism (same input → same output)
6. No lookahead bias (signal at t executes at t+1 open)
"""
import sys
sys.path.insert(0, '/home/claude/quant_portfolio/src')

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, run_backtest, Trade, BacktestResult


def make_dummy_data(prices, signals=None, freq="1h"):
    """Build a small DataFrame for testing."""
    n = len(prices)
    times = pd.date_range("2023-01-01", periods=n, freq=freq)
    df = pd.DataFrame({
        "open": prices,
        "close": prices,
        "high": [p * 1.001 for p in prices],
        "low": [p * 0.999 for p in prices],
        "volume": [100.0] * n,
        "signal": signals if signals is not None else [0] * n,
    }, index=times)
    return df


# =====================================================================
# Tests
# =====================================================================

def test_no_trades_no_change():
    """If signal is always 0, equity should equal initial capital throughout."""
    df = make_dummy_data([100, 110, 120, 130], signals=[0, 0, 0, 0])
    result = run_backtest(df, initial_capital=1000.0)
    
    assert len(result.trades) == 0
    assert result.equity_curve.iloc[-1] == pytest.approx(1000.0)
    assert result.metrics["total_trades"] == 0


def test_single_long_flat_market():
    """A long in a flat market should lose roughly 2*fee_rate."""
    # Signal at t=0 -> open long at t=1's open. Signal back to 0 at t=2 -> close at t=3's open.
    df = make_dummy_data([100, 100, 100, 100, 100], signals=[1, 1, 0, 0, 0])
    result = run_backtest(df, initial_capital=1000.0, fee_rate=0.0015)
    
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == 1
    assert trade.entry_price == 100
    assert trade.exit_price == 100
    
    # Net loss should be ~2 * fee_rate * capital_deployed = 2 * 0.0015 * 1000 = $3
    # (approximately; the second fee is on a slightly smaller amount)
    assert -3.5 < trade.pnl_net < -2.5


def test_single_long_trending_market():
    """A long held during a 10% rise should net roughly 10% minus fees."""
    df = make_dummy_data([100, 100, 105, 108, 110, 110], signals=[1, 1, 1, 1, 0, 0])
    result = run_backtest(df, initial_capital=1000.0, fee_rate=0.0015)
    
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == 100
    assert trade.exit_price == 110
    # Net return should be ~10% minus ~0.3% in fees
    assert 0.095 < trade.return_pct < 0.10


def test_short_profitable_drop():
    """A short during a 10% drop should net positive."""
    df = make_dummy_data([100, 100, 95, 92, 90, 90], signals=[-1, -1, -1, -1, 0, 0])
    result = run_backtest(df, initial_capital=1000.0, fee_rate=0.0015,
                          short_capital_fraction=1.0)
    
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == -1
    assert trade.entry_price == 100
    assert trade.exit_price == 90
    # 10% gain on the short, minus ~0.3% fees
    assert 0.095 < trade.return_pct < 0.10


def test_determinism():
    """Same input → same output."""
    df = make_dummy_data(
        [100 + i * 0.5 for i in range(50)],
        signals=[1] * 20 + [0] * 10 + [-1] * 10 + [0] * 10
    )
    
    r1 = run_backtest(df, initial_capital=1000.0, seed=42)
    r2 = run_backtest(df, initial_capital=1000.0, seed=42)
    
    assert r1.equity_curve.iloc[-1] == r2.equity_curve.iloc[-1]
    assert len(r1.trades) == len(r2.trades)


def test_no_lookahead_at_execution():
    """A signal at time t should result in entry at the OPEN of t+1, not t's open."""
    # Construct: open prices differ from close prices, so we can detect which was used
    n = 5
    times = pd.date_range("2023-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open":  [100, 200, 300, 400, 500],  # bar 0=100, bar 1=200, ...
        "close": [150, 250, 350, 450, 550],
        "high":  [160, 260, 360, 460, 560],
        "low":   [90, 190, 290, 390, 490],
        "volume": [100.0] * n,
        # Signal +1 at bar 0 → execute at bar 1's OPEN = 200
        "signal": [1, 0, 0, 0, 0],
    }, index=times)
    
    result = run_backtest(df, initial_capital=1000.0, execution="next_open")
    
    assert len(result.trades) >= 1
    # Entry should be at price 200 (open of bar 1), NOT 100 (open of bar 0)
    assert result.trades[0].entry_price == 200, \
        f"Expected entry at 200, got {result.trades[0].entry_price}. Lookahead bias!"


def test_position_reversal():
    """A long-to-short transition should close the long and open a short."""
    df = make_dummy_data([100, 100, 100, 100, 100, 100],
                         signals=[1, 1, -1, -1, 0, 0])
    result = run_backtest(df, initial_capital=1000.0, fee_rate=0.0015,
                          short_capital_fraction=1.0)
    
    assert len(result.trades) == 2
    assert result.trades[0].side == 1  # first trade was long
    assert result.trades[1].side == -1  # second trade was short


def test_end_of_data_forces_close():
    """An open position at end of data should be closed."""
    df = make_dummy_data([100, 100, 110, 120], signals=[1, 1, 1, 1])
    result = run_backtest(df, initial_capital=1000.0)
    
    # Even though signal stays 1, the last bar forces close
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_data"


def test_buy_and_hold_equivalence():
    """A trade from start to end should approximately match buy-and-hold returns minus 2x fees."""
    n = 100
    prices = [100 * (1 + 0.001 * i) for i in range(n)]  # gradual rise
    signals = [1] * n
    df = make_dummy_data(prices, signals=signals)
    
    result = run_backtest(df, initial_capital=1000.0, fee_rate=0.0015)
    
    # Buy-and-hold from open of bar 1 (≈100.1) to close of bar 99 (≈109.9)
    # Approx 9.8% gross return
    final_equity = result.equity_curve.iloc[-1]
    gross_pct = (final_equity / 1000.0 - 1) * 100
    # Should be between 9% and 10% net of fees
    assert 9.0 < gross_pct < 10.0, f"Expected 9-10%, got {gross_pct:.2f}%"


def test_metrics_present():
    """All standard metrics should be in the result."""
    df = make_dummy_data([100 + i for i in range(50)], signals=[1] * 30 + [0] * 20)
    result = run_backtest(df, initial_capital=1000.0)
    
    required_keys = ["sharpe", "sortino", "max_drawdown", "calmar",
                     "total_trades", "win_rate", "total_fees"]
    for key in required_keys:
        assert key in result.metrics, f"Missing metric: {key}"


def test_summary_string():
    """Summary should produce a non-empty string."""
    df = make_dummy_data([100, 105, 110, 108, 112], signals=[1, 1, 0, -1, 0])
    result = run_backtest(df)
    summary = result.summary()
    assert isinstance(summary, str)
    assert "Backtest Summary" in summary


if __name__ == "__main__":
    # Allow running directly: python test_backtester.py
    import traceback
    tests = [
        test_no_trades_no_change,
        test_single_long_flat_market,
        test_single_long_trending_market,
        test_short_profitable_drop,
        test_determinism,
        test_no_lookahead_at_execution,
        test_position_reversal,
        test_end_of_data_forces_close,
        test_buy_and_hold_equivalence,
        test_metrics_present,
        test_summary_string,
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
