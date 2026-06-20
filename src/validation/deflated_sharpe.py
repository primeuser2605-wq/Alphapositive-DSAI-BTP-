"""
deflated_sharpe.py
==================
Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

What is the Deflated Sharpe Ratio?
----------------------------------
When you tune a strategy by trying many parameter combinations, the best
in-sample Sharpe is a biased estimator of the true Sharpe: by selection
alone, even random strategies can show high Sharpe.

The DSR is a hypothesis test:
  H_0: true Sharpe = 0 (no edge)
  H_1: true Sharpe > 0 (real edge)

It corrects the observed Sharpe for:
- Number of trials N (selection bias from picking the best of N)
- Sample length T (small samples have high Sharpe variance)
- Non-normality of returns (skewness and kurtosis)

Formula
-------
Expected maximum Sharpe under H_0 across N trials:
    SR_0 = sqrt(V) * [(1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/N·e⁻¹)]

where V = variance of Sharpes across the N trials, γ ≈ 0.5772 (Euler-Mascheroni),
Z⁻¹ is the standard normal quantile.

Deflated Sharpe Ratio:
    DSR = Z( (SR_obs - SR_0) · sqrt(T-1) / sqrt(1 - γ₃·SR_obs + (γ₄-1)/4·SR_obs²) )

where γ₃ = sample skewness, γ₄ = sample kurtosis (NOT excess kurtosis;
Pearson's standard kurtosis), Z is the standard normal CDF.

DSR is a probability in [0, 1]:
- DSR > 0.95: strong evidence true Sharpe > 0 after deflation
- DSR ≈ 0.50: no evidence; observed Sharpe is consistent with H_0
- DSR < 0.10: observed Sharpe is below what selection bias alone would produce

Reference
---------
Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting, and Non-Normality." Journal of Portfolio Management
40(5), 94-107, 2014.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


EULER_MASCHERONI = 0.5772156649015329


# =====================================================================
# Result container
# =====================================================================
@dataclass
class DSRResult:
    """Deflated Sharpe Ratio computation result."""
    observed_sharpe: float
    expected_max_sharpe: float  # SR_0
    deflated_sharpe: float       # the probability
    n_observations: int
    n_trials: int
    sharpe_variance: float
    skewness: float
    kurtosis: float

    def __str__(self) -> str:
        return (
            f"DSR result:\n"
            f"  Observed Sharpe (in-sample):   {self.observed_sharpe:+.4f}\n"
            f"  Expected max SR under H_0:     {self.expected_max_sharpe:+.4f}\n"
            f"  Deflated Sharpe (prob):        {self.deflated_sharpe:.4f}\n"
            f"  Sample size T:                 {self.n_observations}\n"
            f"  Number of trials N:            {self.n_trials}\n"
            f"  Sharpe variance across trials: {self.sharpe_variance:.6f}\n"
            f"  Skewness of returns:           {self.skewness:+.4f}\n"
            f"  Kurtosis of returns:           {self.kurtosis:.4f}"
        )

    def verdict(self, threshold: float = 0.95) -> str:
        """Human-readable interpretation."""
        if self.deflated_sharpe > threshold:
            return (f"STRONG EVIDENCE of positive edge (DSR={self.deflated_sharpe:.3f} > "
                    f"{threshold:.2f}). After correcting for {self.n_trials} trials and "
                    f"sample-size effects, observed Sharpe is unlikely under H_0.")
        elif self.deflated_sharpe > 0.5:
            return (f"WEAK EVIDENCE of edge (DSR={self.deflated_sharpe:.3f}). Observed "
                    f"Sharpe is above the selection-bias-corrected null, but not "
                    f"conclusively so.")
        else:
            return (f"NO EVIDENCE of edge after deflation (DSR={self.deflated_sharpe:.3f}). "
                    f"Observed Sharpe is consistent with what {self.n_trials} random "
                    f"trials would produce by chance.")


# =====================================================================
# Helpers
# =====================================================================
def _expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """
    Expected maximum Sharpe ratio under the null hypothesis of zero true edge,
    given N trials with sample variance V of Sharpes.

    From Bailey & López de Prado (2014) eq. 5:
        E[max_n SR_n] ≈ sqrt(V) · [(1-γ) Z⁻¹(1-1/N) + γ Z⁻¹(1-1/(N·e))]
    """
    if n_trials < 2:
        return 0.0
    if sharpe_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    # Z⁻¹(1 - 1/N): inverse normal CDF at 1 - 1/N
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return np.sqrt(sharpe_variance) * ((1 - g) * q1 + g * q2)


def _sample_moments(returns: np.ndarray) -> tuple[float, float]:
    """Return (skewness, kurtosis). Kurtosis is Pearson's (= excess + 3)."""
    if len(returns) < 4:
        return 0.0, 3.0
    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))  # Pearson, not excess
    return skew, kurt


# =====================================================================
# Main function
# =====================================================================
def deflated_sharpe(
    returns: np.ndarray,
    trial_sharpes: Sequence[float],
    observed_sharpe: float | None = None,
    annualization: float = 1.0,
) -> DSRResult:
    """
    Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    returns : np.ndarray
        Per-bar (or per-trade) returns of the BEST strategy. Used for T,
        skewness, kurtosis. NOT used for Sharpe variance.
    trial_sharpes : Sequence[float]
        Sharpe ratios of ALL configurations tried, including the winner.
        Used to estimate Sharpe variance across trials (the selection-bias
        correction). Length is N (number of trials).
    observed_sharpe : float, optional
        Observed Sharpe ratio of the best strategy. If None, computed from
        `returns` as mean(r)/std(r) × sqrt(annualization).
    annualization : float
        Multiplier to scale Sharpe to annual basis. Use 8760 for hourly
        crypto (24*365). Use 252 for daily equities. Default 1.0 (no scaling).

    Returns
    -------
    DSRResult

    Notes
    -----
    The same annualization is applied to `observed_sharpe` if computed and
    to nothing else. If you pass `observed_sharpe` directly, ensure it's
    already on the same scale as the trial_sharpes.
    """
    returns = np.asarray(returns).flatten()
    returns = returns[np.isfinite(returns)]
    T = len(returns)
    if T < 4:
        raise ValueError(f"Need at least 4 returns for skewness/kurtosis, got {T}")

    trial_sharpes = np.asarray(trial_sharpes).flatten()
    trial_sharpes = trial_sharpes[np.isfinite(trial_sharpes)]
    N = len(trial_sharpes)
    if N < 2:
        raise ValueError(f"Need at least 2 trial Sharpes for variance, got {N}")

    # Compute observed Sharpe if not provided
    if observed_sharpe is None:
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r <= 0:
            observed_sharpe = 0.0
        else:
            observed_sharpe = (mean_r / std_r) * np.sqrt(annualization)

    # Variance of trial Sharpes
    V = float(np.var(trial_sharpes, ddof=1))

    # Expected max Sharpe under H_0
    SR0 = _expected_max_sharpe(N, V)

    # Sample moments
    skew, kurt = _sample_moments(returns)

    # Deflated Sharpe (probability)
    # numerator: (SR_obs - SR_0) * sqrt(T-1)
    # denominator: sqrt(1 - skew*SR_obs + (kurt-1)/4 * SR_obs^2)
    SR = observed_sharpe
    denom_sq = 1.0 - skew * SR + (kurt - 1.0) / 4.0 * SR * SR
    if denom_sq <= 0:
        # Degenerate non-normality; clamp to a tiny positive
        denom_sq = 1e-9
    z_stat = (SR - SR0) * np.sqrt(max(T - 1, 1)) / np.sqrt(denom_sq)
    dsr = float(stats.norm.cdf(z_stat))

    return DSRResult(
        observed_sharpe=float(observed_sharpe),
        expected_max_sharpe=float(SR0),
        deflated_sharpe=dsr,
        n_observations=T,
        n_trials=N,
        sharpe_variance=V,
        skewness=skew,
        kurtosis=kurt,
    )


# =====================================================================
# Convenience: DSR directly from a WalkForwardResult
# =====================================================================
def deflated_sharpe_from_walkforward(
    wf_result,
    annualization: float = 8760.0,
) -> DSRResult:
    """
    Convenience: compute DSR from a WalkForwardResult.
    
    The fold Sharpes are used as the 'trials'. The pooled per-bar returns
    are used for T, skewness, kurtosis. The mean of fold Sharpes is used
    as the observed Sharpe.

    Parameters
    ----------
    wf_result : WalkForwardResult
        Output of walkforward.run_walkforward.
    annualization : float
        Default 8760 (hourly crypto: 24 * 365).

    Returns
    -------
    DSRResult
    """
    fold_sharpes = [f.sharpe for f in wf_result.folds]
    if len(fold_sharpes) < 2:
        raise ValueError("Need at least 2 folds for DSR")
    pooled_returns = wf_result.pooled_returns
    mean_sharpe = float(np.mean(fold_sharpes))
    # The fold Sharpes are already annualized by the backtester; pass through.
    return deflated_sharpe(
        returns=pooled_returns,
        trial_sharpes=fold_sharpes,
        observed_sharpe=mean_sharpe,
        annualization=1.0,  # already annualized in the input
    )
