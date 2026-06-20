"""
eth_regime_confirmation.py
===========================
ETH/USDT regime-confirmation strategy.

Generates +1/-1/0 signals using a hierarchical gate-then-signal architecture:
  Stage 1: Three pre-condition gates must hold (Hurst, correlation, ATR)
  Stage 2: CUSUM regime detector on BTC (Kalman-filtered reference)
  Stage 3: Signal confirmation (RSI + Bollinger + Supertrend)
  Stage 4: Risk management (trailing stop, ATR exit, time exit, cooldown)

This module produces a `signal` column. The actual P&L is computed by
the backtester (src/backtester.py), keeping concerns separated.

Refactored from archive/main_1_eth.py with the following improvements:
- Indicator computation delegated to indicators.py (pure functions, tested)
- No global state, no SDK dependency
- Position state tracked explicitly with named variables
- Each stage of the decision can be ablated independently

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import (
    rsi, atr, bollinger_bands, supertrend,
    rolling_hurst, rolling_dfa_hurst,
    kalman_filter_1d, cusum_regime, rolling_correlation,
)


# =====================================================================
# Configuration dataclass
# =====================================================================
@dataclass
class ETHStrategyConfig:
    """All hyperparameters for the ETH strategy in one place."""

    # Pre-condition gate thresholds
    hurst_threshold: float = 0.5
    hurst_window: int = 120  # 5 days @ 1h bars
    hurst_method: str = "rs"  # 'rs' (original) or 'dfa' (better small-sample)

    correlation_window: int = 7
    correlation_threshold: float = 0.6

    atr_window: int = 12
    atr_entry_threshold: float = 0.01  # 1% of open
    atr_exit_threshold: float = 0.025  # 2.5% of open

    # Ablation flags: set any of these False to disable that gate.
    # Default True = original strategy with all three gates active.
    # See validation/gate_ablation.py for the ablation experiment.
    enable_hurst_gate: bool = True
    enable_correlation_gate: bool = True
    enable_atr_gate: bool = True

    # CUSUM parameters
    kalman_obs_cov: float = 0.1
    kalman_trans_cov: float = 0.01
    cusum_sigma_window: int = 5
    cusum_sigma_method: str = "rolling"  # 'rolling' or 'ewma'
    cusum_ewma_halflife: float = 24.0
    cusum_delta: float = 0.8
    cusum_h_factor: float = 1.5

    # Signal indicators
    rsi_window: int = 14
    rsi_high: float = 70.0
    rsi_low: float = 30.0
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    supertrend_period: int = 12
    supertrend_multiplier: float = 2.5

    # Risk management
    trailing_stop_pct: float = 0.10
    max_holding_hours: int = 28 * 24
    cooldown_hours: int = 24


# =====================================================================
# Main strategy function
# =====================================================================
def compute_features(
    btc_eth: pd.DataFrame,
    config: Optional[ETHStrategyConfig] = None,
) -> pd.DataFrame:
    """
    Compute all features required by the strategy.

    Parameters
    ----------
    btc_eth : pd.DataFrame
        DataFrame with columns:
        - datetime (index or column)
        - btc_open, btc_high, btc_low, btc_close
        - eth_open, eth_high, eth_low, eth_close

    config : ETHStrategyConfig
        Strategy hyperparameters.

    Returns
    -------
    pd.DataFrame with all indicators appended.
    """
    if config is None:
        config = ETHStrategyConfig()

    df = btc_eth.copy()
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index)

    # RSI
    df["btc_rsi"] = rsi(df["btc_close"], window=config.rsi_window)
    df["eth_rsi"] = rsi(df["eth_close"], window=config.rsi_window)

    # ATR
    df["btc_atr"] = atr(df["btc_high"], df["btc_low"], df["btc_close"],
                         window=config.atr_window)
    df["eth_atr"] = atr(df["eth_high"], df["eth_low"], df["eth_close"],
                         window=config.atr_window)

    # Correlation between BTC and ETH
    df["btc_eth_corr"] = rolling_correlation(
        df["btc_close"], df["eth_close"],
        window=config.correlation_window,
    )

    # Hurst exponent on ETH
    hurst_fn = rolling_dfa_hurst if config.hurst_method == "dfa" else rolling_hurst
    df["eth_hurst"] = hurst_fn(df["eth_close"], window=config.hurst_window)

    # Bollinger Bands on BTC (used for signal confirmation)
    btc_mid, btc_up, btc_lo = bollinger_bands(
        df["btc_close"], window=config.bollinger_window, num_std=config.bollinger_std
    )
    df["btc_bollinger_middle"] = btc_mid
    df["btc_bollinger_upper"] = btc_up
    df["btc_bollinger_lower"] = btc_lo

    # Supertrend on ETH (direction signal for entry confirmation)
    _, eth_st_dir = supertrend(
        df["eth_high"], df["eth_low"], df["eth_close"],
        period=config.supertrend_period, multiplier=config.supertrend_multiplier
    )
    df["eth_supertrend_direction"] = eth_st_dir

    # Kalman-filtered BTC reference for CUSUM
    df["btc_kalman"] = kalman_filter_1d(
        df["btc_close"],
        observation_covariance=config.kalman_obs_cov,
        transition_covariance=config.kalman_trans_cov,
    )

    # CUSUM regime on BTC
    cusum_hi, cusum_lo, regime = cusum_regime(
        df["btc_close"],
        df["btc_kalman"],
        sigma_window=config.cusum_sigma_window,
        delta=config.cusum_delta,
        h_factor=config.cusum_h_factor,
        sigma_method=config.cusum_sigma_method,
        ewma_halflife=config.cusum_ewma_halflife,
    )
    df["btc_cusum_hi"] = cusum_hi
    df["btc_cusum_lo"] = cusum_lo
    df["btc_regime"] = regime

    return df


def generate_signals(
    features: pd.DataFrame,
    config: Optional[ETHStrategyConfig] = None,
) -> pd.DataFrame:
    """
    Generate +1/-1/0 signals from a feature-enriched DataFrame.

    The signal at bar t indicates the desired position at the open of bar t+1.
    Risk-management-driven exits override entry signals.

    Parameters
    ----------
    features : pd.DataFrame
        Output of compute_features().
    config : ETHStrategyConfig
        Strategy hyperparameters.

    Returns
    -------
    pd.DataFrame with the original columns plus:
        - signal: int in {-1, 0, +1}
        - signal_reason: str describing why the signal fired (for auditability)
    """
    if config is None:
        config = ETHStrategyConfig()

    df = features.copy()
    n = len(df)

    signals = np.zeros(n, dtype=int)
    reasons = np.empty(n, dtype=object)
    reasons[:] = ""

    # State variables
    position = 0  # -1, 0, +1
    entry_idx = -1
    highest_since_entry = -np.inf
    lowest_since_entry = np.inf
    last_stop_idx = -10**9

    # Extract arrays for speed
    eth_close = df["eth_close"].values
    eth_high = df["eth_high"].values
    eth_low = df["eth_low"].values
    btc_atr = df["btc_atr"].values
    btc_open = df["btc_open"].values
    btc_rsi = df["btc_rsi"].values
    btc_regime = df["btc_regime"].values
    btc_close = df["btc_close"].values
    btc_bb_mid = df["btc_bollinger_middle"].values
    btc_bb_lo = df["btc_bollinger_lower"].values
    eth_st_dir = df["eth_supertrend_direction"].values
    eth_hurst = df["eth_hurst"].values
    btc_eth_corr = df["btc_eth_corr"].values

    for t in range(n):
        # --------- Risk management exits (apply if in a position) ---------
        in_cooldown = (t - last_stop_idx) < config.cooldown_hours
        if in_cooldown:
            signals[t] = position
            reasons[t] = "cooldown"
            continue

        if position != 0:
            # Update extreme since entry
            highest_since_entry = max(highest_since_entry, eth_high[t])
            lowest_since_entry = min(lowest_since_entry, eth_low[t])

            # Trailing stop
            if position == 1:
                stop_price = highest_since_entry * (1 - config.trailing_stop_pct)
                if eth_close[t] <= stop_price:
                    signals[t] = 0
                    reasons[t] = "trailing_stop_long"
                    position = 0
                    last_stop_idx = t
                    highest_since_entry = -np.inf
                    lowest_since_entry = np.inf
                    continue
            else:  # short
                stop_price = lowest_since_entry * (1 + config.trailing_stop_pct)
                if eth_close[t] >= stop_price:
                    signals[t] = 0
                    reasons[t] = "trailing_stop_short"
                    position = 0
                    last_stop_idx = t
                    highest_since_entry = -np.inf
                    lowest_since_entry = np.inf
                    continue

            # Volatility exit
            if not np.isnan(btc_atr[t]) and btc_atr[t] > config.atr_exit_threshold * btc_open[t]:
                signals[t] = 0
                reasons[t] = "atr_exit"
                position = 0
                last_stop_idx = t
                highest_since_entry = -np.inf
                lowest_since_entry = np.inf
                continue

            # Time-based exit
            if (t - entry_idx) >= config.max_holding_hours:
                signals[t] = 0
                reasons[t] = "time_exit"
                position = 0
                last_stop_idx = t
                highest_since_entry = -np.inf
                lowest_since_entry = np.inf
                continue

        # --------- Pre-condition gates (must all hold for entry consideration) ---------
        # Each gate is short-circuited if disabled via the corresponding flag.
        # NaN check: skip it for disabled gates so warm-up bars aren't gated by data we don't use.
        if config.enable_atr_gate:
            atr_ok = (not np.isnan(btc_atr[t])) and (btc_atr[t] < config.atr_entry_threshold * btc_open[t])
        else:
            atr_ok = True

        if config.enable_correlation_gate:
            corr_ok = (not np.isnan(btc_eth_corr[t])) and (btc_eth_corr[t] > config.correlation_threshold)
        else:
            corr_ok = True

        if config.enable_hurst_gate:
            hurst_ok = (not np.isnan(eth_hurst[t])) and (eth_hurst[t] > config.hurst_threshold)
        else:
            hurst_ok = True

        gates_open = atr_ok and corr_ok and hurst_ok

        if not gates_open:
            signals[t] = position
            continue

        # --------- Entry / exit signals ---------
        if position == 0:
            # LONG entry
            if (
                btc_rsi[t] > config.rsi_high
                and btc_regime[t] == "bullish"
                and btc_close[t] > btc_bb_mid[t]
                and eth_st_dir[t] == 1
            ):
                signals[t] = 1
                reasons[t] = "long_entry"
                position = 1
                entry_idx = t
                highest_since_entry = eth_high[t]
                lowest_since_entry = eth_low[t]

            # SHORT entry
            elif (
                btc_rsi[t] < config.rsi_low
                and btc_regime[t] == "bearish"
                and btc_close[t] < btc_bb_lo[t]
                and eth_st_dir[t] == -1
            ):
                signals[t] = -1
                reasons[t] = "short_entry"
                position = -1
                entry_idx = t
                highest_since_entry = eth_high[t]
                lowest_since_entry = eth_low[t]

        elif position == 1:
            # LONG exit on bearish signal
            if (
                btc_rsi[t] < config.rsi_low
                and btc_regime[t] == "bearish"
                and btc_close[t] < btc_bb_lo[t]
                and eth_st_dir[t] == -1
            ):
                signals[t] = 0
                reasons[t] = "long_exit_signal"
                position = 0
                highest_since_entry = -np.inf
                lowest_since_entry = np.inf
            else:
                signals[t] = 1  # hold long

        elif position == -1:
            # SHORT exit on bullish signal
            if (
                btc_rsi[t] > config.rsi_high
                and btc_regime[t] == "bullish"
                and btc_close[t] > btc_bb_mid[t]
                and eth_st_dir[t] == 1
            ):
                signals[t] = 0
                reasons[t] = "short_exit_signal"
                position = 0
                highest_since_entry = -np.inf
                lowest_since_entry = np.inf
            else:
                signals[t] = -1  # hold short

    df["signal"] = signals
    df["signal_reason"] = reasons
    return df


def run_eth_strategy(
    btc_eth: pd.DataFrame,
    config: Optional[ETHStrategyConfig] = None,
) -> pd.DataFrame:
    """
    Convenience function: compute features and generate signals in one call.

    Parameters
    ----------
    btc_eth : pd.DataFrame
        Required columns: btc_open, btc_high, btc_low, btc_close,
                          eth_open, eth_high, eth_low, eth_close
    config : ETHStrategyConfig

    Returns
    -------
    pd.DataFrame with signal column appended. Pass directly to backtester:
    
    >>> from backtester import run_backtest
    >>> signals = run_eth_strategy(btc_eth_df)
    >>> # Rename ETH columns for backtester convention:
    >>> signals = signals.rename(columns={
    ...     "eth_open": "open", "eth_high": "high",
    ...     "eth_low": "low", "eth_close": "close"
    ... })
    >>> result = run_backtest(signals)
    """
    features = compute_features(btc_eth, config=config)
    return generate_signals(features, config=config)
