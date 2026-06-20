"""
test_cold_oos.py
================
Tests for the cold OOS evaluator. We can't hit Binance in CI, but we can
verify the runner functions work end-to-end when handed pre-loaded data.
"""
import sys, os, json, tempfile, shutil
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')
sys.path.insert(0, '/home/claude/quant_portfolio/src/validation')
sys.path.insert(0, '/home/claude/quant_portfolio/src/data_io')

from pathlib import Path
import numpy as np
import pandas as pd

from cold_oos import run_eth_on_cold, run_btc_on_cold, write_summary


def make_synthetic_btc_eth(n=1500, seed=42):
    """Synthetic correlated BTC/ETH for OOS testing."""
    np.random.seed(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    rets = np.zeros(n)
    a, b = n // 3, 2 * n // 3
    rets[:a] = np.random.randn(a) * 0.005 + 0.001
    rets[a:b] = np.random.randn(b - a) * 0.008 - 0.001
    rets[b:] = np.random.randn(n - b) * 0.003
    btc_close = 20000 * np.cumprod(1 + rets)
    eth_close = 1500 * np.cumprod(1 + 0.85 * rets + np.random.randn(n) * 0.003)
    def ohlcv(c):
        o = np.concatenate([[c[0]], c[:-1]])
        return o, np.maximum(c, o) * 1.001, np.minimum(c, o) * 0.999
    bo, bh, bl = ohlcv(btc_close)
    eo, eh, el = ohlcv(eth_close)
    df = pd.DataFrame({
        "datetime": times,
        "btc_open": bo, "btc_high": bh, "btc_low": bl, "btc_close": btc_close, "btc_volume": np.ones(n) * 100,
        "eth_open": eo, "eth_high": eh, "eth_low": el, "eth_close": eth_close, "eth_volume": np.ones(n) * 100,
    })
    df = df.set_index("datetime", drop=False)
    df.index.name = "datetime_idx"
    return df


def test_eth_cold_run_produces_files():
    """run_eth_on_cold should write trades CSV, equity CSV, metrics JSON, report text."""
    df = make_synthetic_btc_eth(n=800)
    out = Path(tempfile.mkdtemp())
    try:
        payload = run_eth_on_cold(df, out)
        assert (out / "eth_trades.csv").exists()
        assert (out / "eth_equity.csv").exists()
        assert (out / "eth_metrics.json").exists()
        assert (out / "eth_report.txt").exists()
        # JSON should parse
        with open(out / "eth_metrics.json") as f:
            data = json.load(f)
        assert data["strategy"] == "eth_regime_confirmation"
        assert "metrics" in data
    finally:
        shutil.rmtree(out)


def test_btc_cold_run_produces_files():
    """run_btc_on_cold should write all expected artifacts."""
    df = make_synthetic_btc_eth(n=1200)  # enough for 70/30 split + Q-training
    out = Path(tempfile.mkdtemp())
    try:
        payload = run_btc_on_cold(df, out)
        assert (out / "btc_trades.csv").exists()
        assert (out / "btc_equity.csv").exists()
        assert (out / "btc_metrics.json").exists()
        assert (out / "btc_report.txt").exists()
        with open(out / "btc_metrics.json") as f:
            data = json.load(f)
        assert data["strategy"] == "btc_qlearning"
        assert "caveat" in data
        assert data["n_bars_train"] + data["n_bars_test"] == data["n_bars_total"]
    finally:
        shutil.rmtree(out)


def test_summary_writer():
    """write_summary should produce a non-empty file."""
    out = Path(tempfile.mkdtemp())
    try:
        eth_payload = {
            "metrics": {"sharpe": 1.5, "total_trades": 20, "total_return": 0.10,
                        "max_drawdown": -0.15, "win_rate": 0.55},
            "bootstrap_cis": None,
        }
        btc_payload = {
            "metrics": {"sharpe": 0.8, "total_trades": 15, "total_return": 0.05,
                        "max_drawdown": -0.20, "win_rate": 0.50},
            "bootstrap_cis": None,
            "caveat": "test caveat",
        }
        write_summary(out, eth_payload, btc_payload, "2024-01-01", "2024-06-30")
        summary_path = out / "cold_oos_summary.txt"
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "COLD OUT-OF-SAMPLE EVALUATION" in content
        assert "ETH/USDT" in content
        assert "BTC/USDT" in content
    finally:
        shutil.rmtree(out)


def test_metrics_json_includes_config():
    """The output JSON should record the frozen strategy config (for audit)."""
    df = make_synthetic_btc_eth(n=800)
    out = Path(tempfile.mkdtemp())
    try:
        run_eth_on_cold(df, out)
        with open(out / "eth_metrics.json") as f:
            data = json.load(f)
        assert "config" in data
        # The frozen ETH config has known fields
        assert "hurst_threshold" in data["config"]
        assert "cusum_delta" in data["config"]
    finally:
        shutil.rmtree(out)


def test_btc_caveat_present_in_fallback_mode():
    """In fallback (no Q-table) mode, BTC payload must document the retraining caveat."""
    df = make_synthetic_btc_eth(n=1200)
    out = Path(tempfile.mkdtemp())
    try:
        # No q_table_path provided → fallback mode → caveat should be set
        payload = run_btc_on_cold(df, out, q_table_path=None)
        assert payload.get("mode") == "fallback_retrain"
        assert payload.get("caveat") is not None
        assert "retrained" in payload["caveat"].lower()
        with open(out / "btc_report.txt") as f:
            content = f.read()
        assert "CAVEAT" in content
    finally:
        shutil.rmtree(out)


def test_btc_true_cold_mode_when_q_table_provided():
    """When a pre-trained Q-table is supplied, BTC runs in true cold OOS mode."""
    import sys
    sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')
    from btc_qlearning import (
        BTCQLearningConfig, compute_features, TradingEnvironment,
        train_q_agent, save_q_table, state_size, N_ACTIONS,
    )
    # Train a small Q-table on a different synthetic dataset (simulating 2020-2023)
    cfg = BTCQLearningConfig(n_episodes=5)
    # Make a separate "training" dataset
    train_df = make_synthetic_btc_eth(n=800, seed=99)
    btc_only = train_df.rename(columns={
        "btc_open": "open", "btc_high": "high",
        "btc_low": "low", "btc_close": "close",
        "btc_volume": "volume",
    })[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    feat = compute_features(btc_only, cfg).dropna(
        subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"]
    ).reset_index(drop=True)
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

    # Save it
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        q_path = f.name
    save_q_table(Q, cfg, q_path, metadata={"trained_on": "synthetic_pre_oos"})

    # Now run cold OOS with this Q-table
    cold_df = make_synthetic_btc_eth(n=600, seed=42)
    out = Path(tempfile.mkdtemp())
    try:
        payload = run_btc_on_cold(cold_df, out, q_table_path=q_path)
        assert payload.get("mode") == "true_cold_oos"
        assert payload.get("caveat") is None
        assert payload.get("n_bars_train") == 0
        # Report should NOT include the retrain caveat
        with open(out / "btc_report.txt") as f:
            content = f.read()
        assert "true cold OOS evaluation" in content
    finally:
        shutil.rmtree(out)
        os.unlink(q_path)


if __name__ == "__main__":
    import traceback
    tests = [
        test_eth_cold_run_produces_files,
        test_summary_writer,
        test_metrics_json_includes_config,
        test_btc_cold_run_produces_files,
        test_btc_caveat_present_in_fallback_mode,
        test_btc_true_cold_mode_when_q_table_provided,
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
