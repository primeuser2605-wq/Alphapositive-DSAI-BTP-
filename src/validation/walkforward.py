"""
walkforward.py
==============
Time-series walk-forward validation for trading strategies.

Two modes
---------
- 'rolling':   fixed train size, slide forward. Better for non-stationary markets.
- 'expanding': train grows over time. Tighter estimates if regime is stable.

Purging and embargo
-------------------
- Purge: drop the last `purge_bars` of each train window. These bars overlap
  (via rolling indicators) with the start of the test window, creating
  information leakage if not removed.
- Embargo: leave `embargo_bars` of gap between train and test. Protects
  against label leakage from positions that were open during the boundary
  in serial-correlated returns. Standard practice from López de Prado.

Output
------
WalkForwardResult containing:
- folds: per-fold metrics
- stitched_equity: equity curve concatenated across all test slices,
  with each fold restarting at $initial_capital (so a "fold-level" curve)
- pooled_returns: per-bar log returns across all test slices, for
  bootstrap or DSR consumption

Caveats
-------
- ETH strategy: no training step → walk-forward measures stability across
  time periods. RL strategies: full retrain per fold, which is expensive
  but the correct semantic.
- The strategy_fn must be deterministic given inputs (or seeded externally).
  Otherwise walk-forward variance conflates strategy noise with regime shifts.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from backtester import run_backtest, BacktestResult


# =====================================================================
# Containers
# =====================================================================
@dataclass
class WalkForwardFold:
    """One train-test fold of the walk-forward experiment."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_bars: int
    n_test_bars: int
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    final_equity: float

    def to_dict(self) -> dict:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train_bars": self.n_train_bars,
            "n_test_bars": self.n_test_bars,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "final_equity": self.final_equity,
        }


@dataclass
class WalkForwardResult:
    """Full walk-forward experiment result."""
    folds: List[WalkForwardFold]
    pooled_returns: np.ndarray  # log returns concatenated across test slices
    config_summary: dict

    def folds_df(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        return pd.DataFrame([f.to_dict() for f in self.folds])

    def summary_stats(self) -> dict:
        """Aggregate Sharpe/return stats across folds."""
        if not self.folds:
            return {}
        sharpes = np.array([f.sharpe for f in self.folds])
        returns = np.array([f.total_return for f in self.folds])
        mdds = np.array([f.max_drawdown for f in self.folds])
        return {
            "n_folds": len(self.folds),
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
            "min_sharpe": float(np.min(sharpes)),
            "max_sharpe": float(np.max(sharpes)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "mean_mdd": float(np.mean(mdds)),
            "worst_mdd": float(np.min(mdds)),  # most negative
            "positive_fold_rate": float(np.mean(sharpes > 0)),
        }

    def report(self) -> str:
        """Pretty-print summary."""
        s = self.summary_stats()
        if not s:
            return "(no folds)"
        cfg = self.config_summary
        lines = [
            "=" * 78,
            "WALK-FORWARD VALIDATION RESULT",
            "=" * 78,
            f"Mode: {cfg.get('mode', '?')} | Train bars: {cfg.get('train_bars', '?')} | "
            f"Test bars: {cfg.get('test_bars', '?')} | Folds: {s['n_folds']}",
            f"Purge: {cfg.get('purge_bars', 0)} | Embargo: {cfg.get('embargo_bars', 0)}",
            "",
            f"Sharpe across folds:   mean={s['mean_sharpe']:+.3f}  "
            f"std={s['std_sharpe']:.3f}  "
            f"range=[{s['min_sharpe']:+.3f}, {s['max_sharpe']:+.3f}]",
            f"Return across folds:   mean={s['mean_return']*100:+.2f}%  "
            f"std={s['std_return']*100:.2f}%",
            f"Worst drawdown across folds: {s['worst_mdd']*100:.2f}%",
            f"Folds with positive Sharpe: {s['positive_fold_rate']*100:.1f}%",
            "",
            "Per-fold details:",
            f"{'Fold':>4} {'TestStart':>12} {'TestEnd':>12}  "
            f"{'Sharpe':>8} {'Return':>8} {'MaxDD':>7} {'Trades':>6} {'WinR':>6}",
            "-" * 78,
        ]
        for f in self.folds:
            lines.append(
                f"{f.fold_index:>4} "
                f"{f.test_start.strftime('%Y-%m-%d'):>12} "
                f"{f.test_end.strftime('%Y-%m-%d'):>12}  "
                f"{f.sharpe:>+8.3f} {f.total_return*100:>+7.2f}% "
                f"{f.max_drawdown*100:>+6.2f}% "
                f"{f.total_trades:>6d} {f.win_rate*100:>5.1f}%"
            )
        lines.append("=" * 78)
        return "\n".join(lines)


# =====================================================================
# Fold generation
# =====================================================================
def generate_folds(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: Optional[int] = None,
    mode: str = "rolling",
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> List[tuple]:
    """
    Generate (train_idx, test_idx) tuples for walk-forward.

    Parameters
    ----------
    df : DataFrame
        Data sorted by datetime.
    train_bars : int
        Size of training window (in bars).
    test_bars : int
        Size of test window (in bars).
    step_bars : int, optional
        How far to slide the window forward each fold. Default = test_bars
        (non-overlapping test slices, which is the standard choice).
    mode : str
        'rolling' (fixed train size) or 'expanding' (train grows).
    purge_bars : int
        Number of bars to drop from end of train (to prevent indicator
        warm-up overlap with test).
    embargo_bars : int
        Number of bars to skip between train end and test start (to prevent
        position-overlap leakage).

    Returns
    -------
    list of (train_start_idx, train_end_idx, test_start_idx, test_end_idx)
    """
    if step_bars is None:
        step_bars = test_bars
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"mode must be 'rolling' or 'expanding', got {mode!r}")

    n = len(df)
    folds = []
    test_start = train_bars + embargo_bars
    train_start_init = 0
    fold_idx = 0

    while test_start + test_bars <= n:
        # Train window
        if mode == "rolling":
            tr_start = test_start - train_bars - embargo_bars
        else:  # expanding
            tr_start = train_start_init
        tr_end = test_start - embargo_bars  # exclusive
        # Apply purge
        tr_end_purged = tr_end - purge_bars
        if tr_end_purged <= tr_start:
            # Not enough train data; skip this fold
            test_start += step_bars
            continue

        te_start = test_start
        te_end = test_start + test_bars  # exclusive

        folds.append((tr_start, tr_end_purged, te_start, te_end))
        fold_idx += 1
        test_start += step_bars

    return folds


# =====================================================================
# Walk-forward runner
# =====================================================================
def run_walkforward(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    train_bars: int,
    test_bars: int,
    step_bars: Optional[int] = None,
    mode: str = "rolling",
    purge_bars: int = 0,
    embargo_bars: int = 0,
    initial_capital: float = 1000.0,
    fee_rate: float = 0.0015,
    price_col: str = "close",
    open_col: str = "open",
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Run a walk-forward experiment.

    Parameters
    ----------
    df : DataFrame
        Full dataset sorted by datetime. Must have a 'datetime' column
        and the open_col/price_col columns.
    strategy_fn : Callable[(train_df, test_df) -> test_df_with_signal]
        Strategy adapter. Given a train slice and a test slice, returns
        the test slice with an added 'signal' column (+1/-1/0). The
        train slice is for any training step (e.g., RL). For
        non-training strategies, the function can ignore train_df.
    train_bars, test_bars : int
        Window sizes.
    step_bars : int, optional
        Slide distance per fold. Default = test_bars (non-overlapping).
    mode : str
        'rolling' or 'expanding'.
    purge_bars, embargo_bars : int
        See generate_folds.
    initial_capital : float
        Capital at start of each fold (each fold starts fresh).
    fee_rate : float
        Per-side transaction cost.
    verbose : bool
        Print progress.

    Returns
    -------
    WalkForwardResult
    """
    fold_specs = generate_folds(
        df, train_bars=train_bars, test_bars=test_bars,
        step_bars=step_bars, mode=mode,
        purge_bars=purge_bars, embargo_bars=embargo_bars,
    )

    if verbose:
        print(f"[walkforward] Generated {len(fold_specs)} folds "
              f"(mode={mode}, train={train_bars}, test={test_bars})")

    folds: List[WalkForwardFold] = []
    pooled_log_returns: List[np.ndarray] = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(fold_specs):
        train_df = df.iloc[tr_s:tr_e].reset_index(drop=True)
        test_df = df.iloc[te_s:te_e].reset_index(drop=True)
        if verbose:
            print(f"[walkforward] Fold {i + 1}/{len(fold_specs)}: "
                  f"train [{tr_s}:{tr_e}] ({len(train_df)} bars) -> "
                  f"test [{te_s}:{te_e}] ({len(test_df)} bars)")

        # Run strategy
        test_with_signals = strategy_fn(train_df, test_df)

        # Backtest
        bt_result = run_backtest(
            test_with_signals,
            signal_col="signal",
            price_col=price_col,
            open_col=open_col,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
        )

        # Extract metrics
        m = bt_result.metrics
        # Compute per-bar log returns from the equity curve (for DSR pooling)
        eq = bt_result.equity_curve
        log_rets = np.log(eq / eq.shift(1)).dropna().values
        pooled_log_returns.append(log_rets)

        # Determine test start/end timestamps
        if "datetime" in test_df.columns:
            test_start_ts = pd.Timestamp(test_df["datetime"].iloc[0])
            test_end_ts = pd.Timestamp(test_df["datetime"].iloc[-1])
        else:
            test_start_ts = pd.Timestamp(test_df.index[0])
            test_end_ts = pd.Timestamp(test_df.index[-1])
        if "datetime" in train_df.columns:
            train_start_ts = pd.Timestamp(train_df["datetime"].iloc[0])
            train_end_ts = pd.Timestamp(train_df["datetime"].iloc[-1])
        else:
            train_start_ts = pd.Timestamp(train_df.index[0])
            train_end_ts = pd.Timestamp(train_df.index[-1])

        folds.append(WalkForwardFold(
            fold_index=i,
            train_start=train_start_ts,
            train_end=train_end_ts,
            test_start=test_start_ts,
            test_end=test_end_ts,
            n_train_bars=len(train_df),
            n_test_bars=len(test_df),
            sharpe=m.get("sharpe", 0.0),
            sortino=m.get("sortino", 0.0),
            total_return=m.get("total_return", 0.0),
            max_drawdown=m.get("max_drawdown", 0.0),
            total_trades=m.get("total_trades", 0),
            win_rate=m.get("win_rate", 0.0),
            final_equity=float(eq.iloc[-1]) if len(eq) else initial_capital,
        ))

    pooled = np.concatenate(pooled_log_returns) if pooled_log_returns else np.array([])

    return WalkForwardResult(
        folds=folds,
        pooled_returns=pooled,
        config_summary={
            "mode": mode,
            "train_bars": train_bars,
            "test_bars": test_bars,
            "step_bars": step_bars or test_bars,
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "initial_capital": initial_capital,
            "fee_rate": fee_rate,
        },
    )
