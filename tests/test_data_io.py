"""
test_data_io.py
===============
Tests for the data loading and parity-check binning utilities.
"""
import sys, os, tempfile, warnings
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/data_io')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')

import numpy as np
import pandas as pd

from loader import load_ohlcv_csv, load_btc_eth_pair, slice_by_date
from parity_check import (
    bin_trades_into_quarters, compare_quarters, load_original_summary,
    QuarterComparison,
)


def make_tmp_csv(rows: list[dict], extension: str = ".csv") -> str:
    """Write a small CSV to a temporary file and return its path."""
    df = pd.DataFrame(rows)
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=extension, delete=False)
    df.to_csv(tf.name, index=False)
    tf.close()
    return tf.name


# =====================================================================
# Loader tests
# =====================================================================
def test_loader_basic_iso_timestamps():
    """Load a CSV with ISO timestamps."""
    rows = [
        {"datetime": "2023-01-01 00:00:00", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 50},
        {"datetime": "2023-01-01 01:00:00", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 55},
        {"datetime": "2023-01-01 02:00:00", "open": 102, "high": 104, "low": 101, "close": 103, "volume": 60},
    ]
    p = make_tmp_csv(rows)
    df = load_ohlcv_csv(p, warn_on_gaps=False)
    assert len(df) == 3
    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == 103
    os.unlink(p)


def test_loader_unix_ms_timestamps():
    """Load a CSV with unix millisecond timestamps."""
    ms = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
    rows = [
        {"open_time": ms,          "open": 100, "high": 102, "low": 99, "close": 101},
        {"open_time": ms + 3600000, "open": 101, "high": 103, "low": 100, "close": 102},
    ]
    p = make_tmp_csv(rows)
    df = load_ohlcv_csv(p, warn_on_gaps=False)
    assert len(df) == 2
    # Should have parsed timestamps correctly
    assert df["datetime"].iloc[0] == pd.Timestamp("2023-01-01")
    os.unlink(p)


def test_loader_handles_dedup():
    """Duplicate timestamps should be dropped."""
    rows = [
        {"datetime": "2023-01-01 00:00:00", "open": 100, "high": 101, "low": 99, "close": 100},
        {"datetime": "2023-01-01 00:00:00", "open": 105, "high": 106, "low": 104, "close": 105},
        {"datetime": "2023-01-01 01:00:00", "open": 101, "high": 102, "low": 100, "close": 101},
    ]
    p = make_tmp_csv(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = load_ohlcv_csv(p, warn_on_gaps=False)
    assert len(df) == 2
    os.unlink(p)


def test_loader_missing_ohlc_raises():
    """Missing required columns should raise a ValueError."""
    rows = [{"datetime": "2023-01-01", "open": 100}]  # no high/low/close
    p = make_tmp_csv(rows)
    try:
        load_ohlcv_csv(p, warn_on_gaps=False)
        assert False, "Should have raised"
    except ValueError:
        pass
    os.unlink(p)


def test_loader_handles_uppercase_columns():
    """Case-insensitive column names."""
    rows = [
        {"DateTime": "2023-01-01 00:00:00", "Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 50},
    ]
    p = make_tmp_csv(rows)
    df = load_ohlcv_csv(p, warn_on_gaps=False)
    assert "open" in df.columns  # normalized to lowercase
    assert df["open"].iloc[0] == 100
    os.unlink(p)


def test_slice_by_date():
    """slice_by_date should filter by inclusive date range."""
    idx = pd.date_range("2023-01-01", periods=100, freq="1h")
    df = pd.DataFrame({"close": range(100)}, index=idx)
    df.index.name = "datetime"
    sliced = slice_by_date(df, start="2023-01-01 10:00:00", end="2023-01-01 20:00:00")
    assert len(sliced) == 11


def test_load_pair_aligns_on_inner_join():
    """When BTC has more bars than ETH, alignment should drop the extras."""
    btc_rows = [
        {"datetime": f"2023-01-01 0{i}:00:00", "open": 100+i, "high": 101+i, "low": 99+i, "close": 100+i}
        for i in range(5)
    ]
    eth_rows = btc_rows[:3]
    btc_p = make_tmp_csv(btc_rows)
    eth_p = make_tmp_csv(eth_rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = load_btc_eth_pair(btc_p, eth_p, align=True)
    assert len(merged) == 3  # inner join drops 2 extra BTC bars
    assert "btc_close" in merged.columns
    assert "eth_close" in merged.columns
    os.unlink(btc_p); os.unlink(eth_p)


# =====================================================================
# Quarterly binning tests
# =====================================================================
def test_bin_trades_into_quarters_basic():
    """A trade closed in Q1 2023 should appear in the Q1 2023 bin."""
    trades = pd.DataFrame([
        {"exit_time": pd.Timestamp("2023-02-15"), "pnl_net": 100, "side": 1},
        {"exit_time": pd.Timestamp("2023-05-15"), "pnl_net": -50, "side": -1},
        {"exit_time": pd.Timestamp("2023-05-20"), "pnl_net": 200, "side": 1},
    ])
    summary = bin_trades_into_quarters(trades, initial_capital=1000.0)
    assert len(summary) == 2
    q1 = summary[summary["quarter_label"] == "2023Q1"].iloc[0]
    q2 = summary[summary["quarter_label"] == "2023Q2"].iloc[0]
    assert q1["Total Trades"] == 1
    assert abs(q1["Profit (%)"] - 10.0) < 0.01  # $100 / $1000 = 10%
    assert q2["Total Trades"] == 2
    assert abs(q2["Profit (%)"] - 15.0) < 0.01  # (-50 + 200) / 1000 = 15%


def test_compare_quarters_matching():
    """When refactor matches original, all comparisons should be 'match'."""
    refactor_summary = pd.DataFrame([
        {"quarter_label": "2023Q1", "Profit (%)": 10.0, "Total Trades": 5, "Win Rate (%)": 60.0,
         "Long Trades": 3, "Short Trades": 2},
    ])
    original_summary = pd.DataFrame([
        {"quarter_label": "2023Q1", "Profit (%)": 12.0, "Total Trades": 5, "Win Rate (%)": 60.0,
         "From": pd.Timestamp("2023-01-01"), "To": pd.Timestamp("2023-03-31")},
    ])
    comps = compare_quarters(refactor_summary, original_summary)
    assert len(comps) == 1
    assert comps[0].profit_match  # within 15% tolerance
    assert comps[0].trades_match
    assert comps[0].win_rate_match


def test_compare_quarters_mismatch():
    """When refactor diverges, mismatch flags should be set."""
    refactor_summary = pd.DataFrame([
        {"quarter_label": "2023Q1", "Profit (%)": 50.0, "Total Trades": 20, "Win Rate (%)": 30.0,
         "Long Trades": 10, "Short Trades": 10},
    ])
    original_summary = pd.DataFrame([
        {"quarter_label": "2023Q1", "Profit (%)": 10.0, "Total Trades": 5, "Win Rate (%)": 60.0,
         "From": pd.Timestamp("2023-01-01"), "To": pd.Timestamp("2023-03-31")},
    ])
    comps = compare_quarters(refactor_summary, original_summary)
    c = comps[0]
    assert not c.profit_match    # 50% vs 10% = 40pp gap > 15pp
    assert not c.trades_match    # 20 vs 5 = +300%
    assert not c.win_rate_match  # 30 vs 60 = 30pp > 20pp


def test_load_original_summary():
    """Loader for the original Inter IIT summary CSV."""
    rows = [
        {"Index": 1, "From": "2020-01-01 00:00:00", "To": "2020-03-31 23:59:59",
         "Initial Balance": 1000.0, "Final Balance": 1765.29,
         "Profit (%)": 76.53, "Benchmark (%)": 2.94, "Benchmark Beaten?": "Yes",
         "Total Trades": 7, "Long Trades": 6, "Short Trades": 1, "Win Rate (%)": 57.14},
    ]
    p = make_tmp_csv(rows)
    df = load_original_summary(p)
    assert "quarter_label" in df.columns
    assert df["quarter_label"].iloc[0] == "2020Q1"
    os.unlink(p)


if __name__ == "__main__":
    import traceback
    tests = [
        test_loader_basic_iso_timestamps,
        test_loader_unix_ms_timestamps,
        test_loader_handles_dedup,
        test_loader_missing_ohlc_raises,
        test_loader_handles_uppercase_columns,
        test_slice_by_date,
        test_load_pair_aligns_on_inner_join,
        test_bin_trades_into_quarters_basic,
        test_compare_quarters_matching,
        test_compare_quarters_mismatch,
        test_load_original_summary,
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
