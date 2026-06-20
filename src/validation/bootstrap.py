"""
bootstrap.py
============
Politis–Romano stationary bootstrap for honest confidence intervals on
backtest performance metrics.

Why this matters
----------------
A backtest produces a SINGLE realization of trade returns. The reported
Sharpe is a point estimate; its uncertainty is invisible until you compute
the sampling distribution. Naive IID bootstrap destroys the autocorrelation
in trade returns (winning/losing streaks). Politis–Romano fixes this by
resampling blocks of random geometric length.

Reference: Politis & Romano (1994), "The Stationary Bootstrap",
Journal of the American Statistical Association, 89(428):1303-1313.

Usage
-----
>>> from bootstrap import bootstrap_metric_ci, bootstrap_all_metrics
>>> result = run_backtest(df)
>>> trade_returns = result.trades_df()["return_pct"].values
>>> ci = bootstrap_metric_ci(trade_returns, metric="sharpe", n_resamples=10000)
>>> print(f"Sharpe 95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd


# =====================================================================
# Result containers
# =====================================================================
@dataclass
class BootstrapCI:
    """Confidence interval and point estimate from a bootstrap."""
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    mean_block_length: float
    distribution: np.ndarray  # full bootstrap distribution for plotting

    def __repr__(self) -> str:
        return (f"BootstrapCI(point={self.point_estimate:.4f}, "
                f"{int(self.confidence_level*100)}% CI=[{self.lower:.4f}, {self.upper:.4f}], "
                f"n={self.n_resamples})")


# =====================================================================
# The stationary bootstrap core
# =====================================================================
def stationary_bootstrap_indices(
    n: int,
    mean_block_length: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a single stationary-bootstrap resample of indices.

    Block lengths are drawn from a Geometric distribution with mean
    `mean_block_length`. The starting index of each block is drawn
    uniformly from [0, n). This preserves stationarity (resample
    distribution invariant under index shifts) while preserving
    autocorrelation up to roughly the mean block length.

    Parameters
    ----------
    n : int
        Length of the desired resample (and of the original series).
    mean_block_length : float
        Expected block length. Typically chosen to match the autocorrelation
        horizon of the data. Equivalent to 1/p in geometric(p) notation.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray of int indices, length n.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block_length  # geometric prob parameter

    indices = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        block_len = rng.geometric(p)
        for j in range(block_len):
            if i >= n:
                break
            indices[i] = (start + j) % n  # circular wrap to maintain stationarity
            i += 1
    return indices


# =====================================================================
# Metric functions (applied to a series of trade-level returns)
# =====================================================================
def _sharpe_ratio(rets: np.ndarray, annualization: float) -> float:
    """Sharpe ratio. For per-trade returns, annualization factor depends on
    trade frequency; for hourly bar returns, it's sqrt(8760)."""
    if len(rets) < 2 or np.std(rets) == 0:
        return 0.0
    return np.sqrt(annualization) * np.mean(rets) / np.std(rets)


def _sortino_ratio(rets: np.ndarray, annualization: float) -> float:
    """Sortino: like Sharpe but using downside deviation."""
    if len(rets) < 2:
        return 0.0
    downside = rets[rets < 0]
    if len(downside) < 2 or np.std(downside) == 0:
        return 0.0
    return np.sqrt(annualization) * np.mean(rets) / np.std(downside)


def _total_return(rets: np.ndarray) -> float:
    """Total compounded return from a sequence of per-trade returns."""
    return float(np.prod(1.0 + rets) - 1.0)


def _max_drawdown(rets: np.ndarray) -> float:
    """Max drawdown computed from a per-trade return sequence (compounded equity curve)."""
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def _win_rate(rets: np.ndarray) -> float:
    """Fraction of trades with positive return."""
    if len(rets) == 0:
        return 0.0
    return float((rets > 0).mean())


METRIC_FUNCTIONS = {
    "sharpe": _sharpe_ratio,
    "sortino": _sortino_ratio,
    "total_return": _total_return,
    "max_drawdown": _max_drawdown,
    "win_rate": _win_rate,
}


# =====================================================================
# Main entry points
# =====================================================================
def bootstrap_metric_ci(
    trade_returns: np.ndarray | pd.Series,
    metric: str = "sharpe",
    n_resamples: int = 10_000,
    mean_block_length: float = 10.0,
    confidence_level: float = 0.95,
    annualization: float = 1.0,
    seed: int | None = 42,
) -> BootstrapCI:
    """
    Compute a stationary-bootstrap confidence interval for a metric.

    Parameters
    ----------
    trade_returns : array-like
        Per-trade returns (e.g., from result.trades_df()["return_pct"]).
        These are treated as a 1D autocorrelated series.
    metric : str
        One of: 'sharpe', 'sortino', 'total_return', 'max_drawdown', 'win_rate'.
    n_resamples : int
        Number of bootstrap resamples. 10,000 is standard for stable CIs.
    mean_block_length : float
        Expected block length in the stationary bootstrap. For hourly trade
        returns with autocorrelation horizon of ~10 trades, 10 is reasonable.
        Should be calibrated to autocorrelation horizon of your data.
    confidence_level : float
        Two-sided confidence level. 0.95 produces a 95% CI.
    annualization : float
        Factor for Sharpe/Sortino annualization. For per-trade returns,
        use approximately (trades_per_year). For hourly bar returns,
        use 8760 = 24*365.
    seed : int | None
        For reproducibility.

    Returns
    -------
    BootstrapCI with point_estimate, lower, upper, distribution.
    """
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(METRIC_FUNCTIONS)}")

    rets = np.asarray(trade_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)
    if n < 2:
        raise ValueError(f"Need at least 2 trade returns, got {n}")

    metric_fn = METRIC_FUNCTIONS[metric]

    # Point estimate on the original data
    if metric in ("sharpe", "sortino"):
        point = metric_fn(rets, annualization)
    else:
        point = metric_fn(rets)

    # Bootstrap distribution
    rng_master = np.random.default_rng(seed)
    distribution = np.empty(n_resamples)
    for i in range(n_resamples):
        sub_seed = int(rng_master.integers(0, 2**31 - 1))
        idx = stationary_bootstrap_indices(n, mean_block_length, seed=sub_seed)
        resample = rets[idx]
        if metric in ("sharpe", "sortino"):
            distribution[i] = metric_fn(resample, annualization)
        else:
            distribution[i] = metric_fn(resample)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(distribution, 100 * alpha / 2))
    upper = float(np.percentile(distribution, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        mean_block_length=mean_block_length,
        distribution=distribution,
    )


def bootstrap_all_metrics(
    trade_returns: np.ndarray | pd.Series,
    n_resamples: int = 10_000,
    mean_block_length: float = 10.0,
    confidence_level: float = 0.95,
    annualization: float = 1.0,
    seed: int | None = 42,
) -> dict[str, BootstrapCI]:
    """
    Convenience: bootstrap CIs for all standard metrics.

    Returns dict mapping metric name → BootstrapCI.
    """
    return {
        name: bootstrap_metric_ci(
            trade_returns, metric=name,
            n_resamples=n_resamples,
            mean_block_length=mean_block_length,
            confidence_level=confidence_level,
            annualization=annualization,
            seed=seed,
        )
        for name in METRIC_FUNCTIONS
    }


def summarize_bootstrap(cis: dict[str, BootstrapCI]) -> str:
    """Pretty-print a dict of bootstrap CIs as a table."""
    lines = [
        f"{'Metric':<16} {'Point':>10}  {'95% CI':>30}",
        "-" * 60,
    ]
    for name, ci in cis.items():
        ci_str = f"[{ci.lower:.4f}, {ci.upper:.4f}]"
        lines.append(f"{name:<16} {ci.point_estimate:>10.4f}  {ci_str:>30}")
    lines.append(f"\n  (n_resamples={list(cis.values())[0].n_resamples}, "
                 f"block_length={list(cis.values())[0].mean_block_length})")
    return "\n".join(lines)
