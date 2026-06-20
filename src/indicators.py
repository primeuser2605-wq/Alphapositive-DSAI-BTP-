"""
indicators.py
=============
All technical indicators used by the strategies, in one auditable place.

Every function is:
- Pure (no global state)
- Causal (uses only data <= time t)
- Documented with the standard formula
- Tested in tests/test_indicators.py
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# =====================================================================
# RSI
# =====================================================================
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index. Classical Wilder-style with simple averages.
    
    RSI = 100 - 100 / (1 + RS),  RS = avg_gain / avg_loss

    Values:
      RSI > 70: overbought (in mean-reversion context); strong upward momentum (in trend context)
      RSI < 30: oversold (in mean-reversion context); strong downward momentum (in trend context)

    Edge cases:
      avg_loss = 0, avg_gain > 0  → RSI = 100 (no losses, only gains)
      avg_loss = 0, avg_gain = 0  → RSI = 50 (no movement, neutral)
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta).clip(lower=0).rolling(window=window).mean()
    
    # Handle edge cases explicitly
    rsi_vals = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # Where loss is zero but gain is positive, RSI = 100
    rsi_vals = rsi_vals.where(~((loss == 0) & (gain > 0)), 100.0)
    # Where both are zero, RSI = 50 (neutral)
    rsi_vals = rsi_vals.where(~((loss == 0) & (gain == 0)), 50.0)
    return rsi_vals


# =====================================================================
# ATR
# =====================================================================
def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range. Captures volatility including overnight gaps.
    
    True Range_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    ATR = rolling mean of TR over `window` bars.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


# =====================================================================
# EMA
# =====================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. Standard pandas implementation."""
    return series.ewm(span=span, adjust=False).mean()


# =====================================================================
# Bollinger Bands
# =====================================================================
def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0
                    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: SMA ± num_std * rolling stdev.
    Returns (middle, upper, lower).
    """
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


# =====================================================================
# Aroon
# =====================================================================
def aroon(high: pd.Series, low: pd.Series, window: int = 14
          ) -> Tuple[pd.Series, pd.Series]:
    """
    Aroon Up/Down. Measures bars since last high/low in a rolling window.
    
    Aroon_Up   = 100 * (window - bars_since_window_high) / window
    Aroon_Down = 100 * (window - bars_since_window_low) / window
    """
    rolling_window = window + 1
    aroon_up = 100 * (window - high.rolling(rolling_window)
                       .apply(lambda x: window - x.argmax(), raw=True)) / window
    aroon_down = 100 * (window - low.rolling(rolling_window)
                         .apply(lambda x: window - x.argmin(), raw=True)) / window
    return aroon_up, aroon_down


# =====================================================================
# Supertrend
# =====================================================================
def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 12, multiplier: float = 2.5
               ) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.
    
    Returns (supertrend_value, direction) where direction is +1 or -1.
    Direction flips only when a clear reversal is detected.
    """
    atr_vals = atr(high, low, close, window=period)
    hl_mid = (high + low) / 2
    upper_band = hl_mid + multiplier * atr_vals
    lower_band = hl_mid - multiplier * atr_vals

    n = len(close)
    st = np.zeros(n)
    direction = np.zeros(n, dtype=int)

    # Initialize
    if n > 0:
        st[0] = upper_band.iloc[0] if not pd.isna(upper_band.iloc[0]) else close.iloc[0]
        direction[0] = 1

    for i in range(1, n):
        if close.iloc[i] > st[i - 1]:
            trend = 1
        elif close.iloc[i] < st[i - 1]:
            trend = -1
        else:
            trend = direction[i - 1]

        if trend == 1:
            if lower_band.iloc[i] < st[i - 1]:
                st[i] = st[i - 1]
            else:
                st[i] = lower_band.iloc[i]
        else:  # trend == -1
            if upper_band.iloc[i] > st[i - 1]:
                st[i] = st[i - 1]
            else:
                st[i] = upper_band.iloc[i]
        direction[i] = trend

    return (
        pd.Series(st, index=close.index, name="supertrend"),
        pd.Series(direction, index=close.index, name="supertrend_direction"),
    )


# =====================================================================
# Hurst exponent (R/S analysis)
# =====================================================================
def _hurst_rs(ts: np.ndarray) -> float:
    """
    Hurst exponent via R/S analysis on a single window of prices.
    
    H ≈ 0.5: random walk
    H > 0.5: persistent / trending
    H < 0.5: mean-reverting / anti-persistent
    
    NOTE: For windows < ~500, this estimator is biased and noisy.
    The `hurst.compute_Hc` package gives slightly different results due to
    different lag selection. This implementation matches Mandelbrot's
    classical R/S definition.
    """
    n = len(ts)
    if n < 20:
        return np.nan

    # Use only sufficient lag scales
    lags = [2, 4, 8, 16, 32, 64]
    lags = [lag for lag in lags if lag < n // 2]
    if len(lags) < 3:
        return np.nan

    rs_values = []
    for lag in lags:
        # Partition the series into blocks of size `lag`
        n_blocks = n // lag
        rs_for_lag = []
        for b in range(n_blocks):
            block = ts[b * lag : (b + 1) * lag]
            if len(block) < lag:
                continue
            mean = np.mean(block)
            dev = block - mean
            cum_dev = np.cumsum(dev)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(block)
            if S > 0:
                rs_for_lag.append(R / S)
        if rs_for_lag:
            rs_values.append((lag, np.mean(rs_for_lag)))

    if len(rs_values) < 3:
        return np.nan

    lags_arr = np.array([x[0] for x in rs_values])
    rs_arr = np.array([x[1] for x in rs_values])
    # Hurst = slope of log(R/S) vs log(lag)
    log_lags = np.log(lags_arr)
    log_rs = np.log(rs_arr)
    H, _ = np.polyfit(log_lags, log_rs, 1)
    return H


def rolling_hurst(prices: pd.Series, window: int = 120) -> pd.Series:
    """
    Rolling Hurst exponent computed via R/S analysis.
    
    Default window=120 matches the original strategy.
    CAVEAT: 120 is below the n>1000 typically recommended for R/S stability.
    The DFA estimator in dfa_hurst() has better small-sample properties.
    """
    return prices.rolling(window=window).apply(_hurst_rs, raw=True)


# =====================================================================
# Detrended Fluctuation Analysis (DFA) - better small-sample Hurst
# =====================================================================
def _dfa_hurst(ts: np.ndarray, min_scale: int = 4, max_scale: int = None) -> float:
    """
    Hurst exponent via Detrended Fluctuation Analysis.
    
    More stable than R/S for windows under ~500. Recommended replacement.
    """
    n = len(ts)
    if n < 20:
        return np.nan

    if max_scale is None:
        max_scale = n // 4

    # Step 1: integrate (cumulative sum of mean-centered series)
    y = np.cumsum(ts - np.mean(ts))

    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_scale), num=10, dtype=int
    ))
    scales = scales[scales >= 4]
    if len(scales) < 3:
        return np.nan

    fluctuations = []
    for s in scales:
        n_segments = n // s
        if n_segments < 1:
            continue
        rms_values = []
        for i in range(n_segments):
            segment = y[i * s : (i + 1) * s]
            x = np.arange(len(segment))
            # Linear detrend
            coef = np.polyfit(x, segment, 1)
            trend = np.polyval(coef, x)
            rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_values:
            fluctuations.append((s, np.mean(rms_values)))

    if len(fluctuations) < 3:
        return np.nan

    log_s = np.log(np.array([f[0] for f in fluctuations]))
    log_f = np.log(np.array([f[1] for f in fluctuations]))
    H, _ = np.polyfit(log_s, log_f, 1)
    return H


def rolling_dfa_hurst(prices: pd.Series, window: int = 120) -> pd.Series:
    """Rolling Hurst exponent via DFA. Better small-sample properties than R/S."""
    return prices.rolling(window=window).apply(_dfa_hurst, raw=True)


# =====================================================================
# Kalman filter (1D, one-sided / causal)
# =====================================================================
def kalman_filter_1d(
    prices: pd.Series,
    observation_covariance: float = 0.1,
    transition_covariance: float = 0.01,
) -> pd.Series:
    """
    One-sided Kalman filter for 1D price series.
    
    Causal: state at time t uses only observations up to t. No lookahead.
    
    Parameters
    ----------
    prices : pd.Series
        Price observations.
    observation_covariance : float
        R, noise variance in the observation. Higher R → smoother estimate.
    transition_covariance : float
        Q, process noise variance. Higher Q → more responsive estimate.

    Returns
    -------
    pd.Series of filtered estimates, indexed like prices.
    """
    n = len(prices)
    if n == 0:
        return prices.copy()

    x = np.zeros(n)
    P = np.zeros(n)
    R = observation_covariance
    Q = transition_covariance

    # Initialize
    x[0] = prices.iloc[0]
    P[0] = 1.0

    for t in range(1, n):
        # Predict
        x_pred = x[t - 1]
        P_pred = P[t - 1] + Q

        # Update
        K = P_pred / (P_pred + R)
        x[t] = x_pred + K * (prices.iloc[t] - x_pred)
        P[t] = (1 - K) * P_pred

    return pd.Series(x, index=prices.index, name="kalman_filtered")


# =====================================================================
# CUSUM regime detection
# =====================================================================
def cusum_regime(
    prices: pd.Series,
    reference: pd.Series,
    sigma_window: int = 5,
    delta: float = 0.8,
    h_factor: float = 1.5,
    sigma_method: str = "rolling",  # "rolling" or "ewma"
    ewma_halflife: float = 24,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    CUSUM-based regime detection with dynamic volatility-scaled thresholds.
    
    S_hi[t] = max(0, S_hi[t-1] + (price - ref - k))
    S_lo[t] = max(0, S_lo[t-1] + (ref - price - k))
    where k = delta * sigma, threshold h = h_factor * sigma.
    
    Bullish regime when S_hi > h; bearish when S_lo > h.
    
    Parameters
    ----------
    prices : pd.Series
        Observed prices.
    reference : pd.Series
        Reference value (e.g., Kalman-filtered prices).
    sigma_window : int
        Window for rolling stdev (used if sigma_method='rolling').
    delta : float
        Slack scaling factor. k_t = delta * sigma_t.
    h_factor : float
        Threshold scaling factor. h_t = h_factor * sigma_t.
    sigma_method : str
        'rolling' (original) or 'ewma' (more stable).
    ewma_halflife : float
        Halflife for EWMA volatility (if sigma_method='ewma').
    
    Returns
    -------
    (S_hi, S_lo, regime)
        regime is 'bullish', 'bearish', or 'neutral' per bar.
    """
    n = len(prices)
    price_arr = prices.values
    ref_arr = reference.values

    # Compute volatility series
    if sigma_method == "rolling":
        sigma = prices.rolling(window=sigma_window).std()
    elif sigma_method == "ewma":
        sigma = prices.ewm(halflife=ewma_halflife, adjust=False).std()
    else:
        raise ValueError(f"Unknown sigma_method: {sigma_method}")

    sigma_arr = sigma.values
    k_arr = delta * sigma_arr
    h_arr = h_factor * sigma_arr

    S_hi = np.zeros(n)
    S_lo = np.zeros(n)
    regime = np.empty(n, dtype=object)
    regime[:] = "neutral"

    for t in range(1, n):
        if np.isnan(k_arr[t]) or np.isnan(ref_arr[t]):
            continue
        S_hi[t] = max(0.0, S_hi[t - 1] + (price_arr[t] - ref_arr[t] - k_arr[t]))
        S_lo[t] = max(0.0, S_lo[t - 1] + (ref_arr[t] - price_arr[t] - k_arr[t]))
        if S_hi[t] > h_arr[t]:
            regime[t] = "bullish"
        elif S_lo[t] > h_arr[t]:
            regime[t] = "bearish"

    return (
        pd.Series(S_hi, index=prices.index, name="cusum_hi"),
        pd.Series(S_lo, index=prices.index, name="cusum_lo"),
        pd.Series(regime, index=prices.index, name="regime"),
    )


# =====================================================================
# Rolling Pearson correlation
# =====================================================================
def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 7) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return a.rolling(window=window).corr(b)
