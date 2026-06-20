"""
backtester.py
=============
An in-house, auditable, deterministic backtester for hourly OHLCV crypto data.

Replaces the external Untrade SDK dependency. Designed for transparency:
every assumption is explicit in code and tested.

Execution model
---------------
- Signals are computed on bar t using only data <= t (no lookahead).
- Trades execute at the OPEN of bar t+1 (next-bar-open execution).
- Each transaction incurs a configurable proportional fee (default 0.15% per side).
- Initial capital and leverage are configurable.

Outputs
-------
BacktestResult dataclass containing:
- trades: list of Trade objects (entry/exit time, price, side, pnl, return_pct)
- equity_curve: pd.Series indexed by datetime
- metrics: dict of standard performance metrics
- per_bar: pd.DataFrame with position, equity, drawdown per bar

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd


# =====================================================================
# Trade record
# =====================================================================
@dataclass
class Trade:
    """A single completed round-trip trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    side: int  # +1 = long, -1 = short
    size: float  # position size in base currency (e.g., ETH)
    pnl_gross: float  # before fees
    pnl_net: float  # after fees
    return_pct: float  # net return on capital deployed
    fees_paid: float
    holding_hours: float
    exit_reason: str  # 'signal', 'stop_loss', 'time_exit', 'end_of_data'

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "side": self.side,
            "size": self.size,
            "pnl_gross": self.pnl_gross,
            "pnl_net": self.pnl_net,
            "return_pct": self.return_pct,
            "fees_paid": self.fees_paid,
            "holding_hours": self.holding_hours,
            "exit_reason": self.exit_reason,
        }


# =====================================================================
# Backtest result container
# =====================================================================
@dataclass
class BacktestResult:
    """Container holding all backtest outputs."""
    trades: list[Trade]
    equity_curve: pd.Series
    per_bar: pd.DataFrame
    config: dict
    metrics: dict = field(default_factory=dict)

    def trades_df(self) -> pd.DataFrame:
        """Return trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def summary(self) -> str:
        """Return a human-readable summary string."""
        m = self.metrics
        lines = [
            f"=== Backtest Summary ===",
            f"Period: {self.per_bar.index[0]} to {self.per_bar.index[-1]}",
            f"Initial capital: ${self.config.get('initial_capital', 0):,.2f}",
            f"Final equity: ${self.equity_curve.iloc[-1]:,.2f}",
            f"Total return: {(self.equity_curve.iloc[-1] / self.config.get('initial_capital', 1) - 1) * 100:.2f}%",
            f"",
            f"Trades: {m.get('total_trades', 0)}",
            f"  Long: {m.get('long_trades', 0)}, Short: {m.get('short_trades', 0)}",
            f"  Win rate: {m.get('win_rate', 0) * 100:.2f}%",
            f"  Avg holding: {m.get('avg_holding_hours', 0):.1f} hours",
            f"",
            f"Risk-adjusted:",
            f"  Sharpe (annualized): {m.get('sharpe', 0):.3f}",
            f"  Sortino (annualized): {m.get('sortino', 0):.3f}",
            f"  Max drawdown: {m.get('max_drawdown', 0) * 100:.2f}%",
            f"  Calmar: {m.get('calmar', 0):.3f}",
            f"",
            f"Costs:",
            f"  Total fees paid: ${m.get('total_fees', 0):,.2f}",
            f"  Gross profit: ${m.get('gross_profit', 0):,.2f}",
            f"  Net profit: ${m.get('net_profit', 0):,.2f}",
        ]
        return "\n".join(lines)


# =====================================================================
# Main backtester
# =====================================================================
class Backtester:
    """
    Deterministic event-loop backtester.

    Parameters
    ----------
    initial_capital : float
        Starting balance in USDT. Default 1000.
    fee_rate : float
        Proportional fee per side. Default 0.0015 (0.15%).
    leverage : float
        Position leverage. Default 1.0.
    short_capital_fraction : float
        Fraction of capital used for short positions (vs full for longs).
        Default 0.75 to match the original strategy's risk asymmetry.
        Set to 1.0 for symmetric sizing.
    bars_per_year : int
        For annualization. 24 * 365 = 8760 for hourly crypto.
    seed : int
        For any stochastic components. Default 42.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        fee_rate: float = 0.0015,
        leverage: float = 1.0,
        short_capital_fraction: float = 0.75,
        bars_per_year: int = 24 * 365,
        seed: int = 42,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.short_capital_fraction = short_capital_fraction
        self.bars_per_year = bars_per_year
        self.seed = seed

    def run(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close",
        open_col: str = "open",
        execution: str = "next_open",
    ) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ['datetime' or index of datetime, open_col,
            price_col, signal_col]. signal_col uses convention:
              +1 = enter or hold long
              -1 = enter or hold short
               0 = exit any position / stay flat
            Transitions (e.g., +1 -> -1) are treated as: exit, then open new.
        signal_col : str
            Column name for position signals.
        price_col : str
            Column name for closing price.
        open_col : str
            Column name for opening price (used for execution).
        execution : str
            'next_open' (default): trades execute at next bar's open.
            'close': trades execute at current bar's close (lookahead risk;
            only use for sanity checks).

        Returns
        -------
        BacktestResult
        """
        np.random.seed(self.seed)

        # Validate
        if execution not in ("next_open", "close"):
            raise ValueError(f"Unknown execution mode: {execution}")
        for col in (signal_col, price_col, open_col):
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

        # Ensure DataFrame is properly indexed by datetime
        df = df.copy()
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        n = len(df)
        signals = df[signal_col].fillna(0).astype(int).values
        prices_close = df[price_col].values
        prices_open = df[open_col].values
        timestamps = df.index

        # State
        cash = self.initial_capital
        position_side = 0  # +1 long, -1 short, 0 flat
        position_size = 0.0  # units of base asset
        entry_price = 0.0
        entry_time_idx = -1
        entry_cash = 0.0  # capital deployed at entry, for return calc

        trades: list[Trade] = []
        equity_arr = np.zeros(n)
        position_arr = np.zeros(n, dtype=int)
        drawdown_arr = np.zeros(n)
        peak_equity = self.initial_capital

        for t in range(n):
            # Compute mark-to-market equity at current bar's close
            if position_side == 0:
                equity = cash
            elif position_side == 1:
                equity = cash + position_size * prices_close[t]
            else:  # short
                equity = cash + position_size * (entry_price - prices_close[t]) + position_size * entry_price
                # Equivalent: for short, we got cash at entry; loss/profit = entry-current

            equity_arr[t] = equity
            position_arr[t] = position_side
            peak_equity = max(peak_equity, equity)
            drawdown_arr[t] = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

            # Signal-driven action: act on the signal observed at t,
            # executing at the next bar's open (no lookahead).
            if execution == "next_open" and t == n - 1:
                # Last bar: force close any open position at current close
                if position_side != 0:
                    self._close_position(
                        trades, position_side, position_size, entry_price,
                        prices_close[t], entry_time_idx, t, timestamps,
                        entry_cash, exit_reason="end_of_data"
                    )
                    cash = equity_arr[t]
                    position_side, position_size, entry_price, entry_time_idx = 0, 0.0, 0.0, -1
                continue

            sig = signals[t]
            # Determine execution price for any trade triggered at bar t
            if execution == "next_open":
                if t + 1 >= n:
                    continue
                exec_price = prices_open[t + 1]
                exec_t = t + 1
            else:
                exec_price = prices_close[t]
                exec_t = t

            # Case 1: signal wants flat
            if sig == 0:
                if position_side != 0:
                    cash_after = self._close_position(
                        trades, position_side, position_size, entry_price,
                        exec_price, entry_time_idx, exec_t, timestamps,
                        entry_cash, exit_reason="signal"
                    )
                    cash = cash_after
                    position_side, position_size, entry_price, entry_time_idx = 0, 0.0, 0.0, -1

            # Case 2: signal wants long
            elif sig == 1:
                if position_side == 1:
                    pass  # already long, hold
                else:
                    if position_side == -1:
                        cash = self._close_position(
                            trades, position_side, position_size, entry_price,
                            exec_price, entry_time_idx, exec_t, timestamps,
                            entry_cash, exit_reason="signal"
                        )
                        position_side, position_size, entry_price, entry_time_idx = 0, 0.0, 0.0, -1
                    # Open long
                    capital_to_deploy = cash * self.leverage
                    fee = capital_to_deploy * self.fee_rate
                    position_size = (capital_to_deploy - fee) / exec_price
                    entry_price = exec_price
                    entry_time_idx = exec_t
                    entry_cash = cash
                    cash = cash - capital_to_deploy + (capital_to_deploy - fee)  # cash decreases by fee
                    cash = cash - (capital_to_deploy - fee)  # all in
                    position_side = 1

            # Case 3: signal wants short
            elif sig == -1:
                if position_side == -1:
                    pass  # already short
                else:
                    if position_side == 1:
                        cash = self._close_position(
                            trades, position_side, position_size, entry_price,
                            exec_price, entry_time_idx, exec_t, timestamps,
                            entry_cash, exit_reason="signal"
                        )
                        position_side, position_size, entry_price, entry_time_idx = 0, 0.0, 0.0, -1
                    # Open short
                    capital_to_deploy = cash * self.short_capital_fraction * self.leverage
                    fee = capital_to_deploy * self.fee_rate
                    position_size = (capital_to_deploy - fee) / exec_price
                    entry_price = exec_price
                    entry_time_idx = exec_t
                    entry_cash = cash
                    # For short: cash reserved as margin, fee paid
                    cash = cash - fee
                    position_side = -1

        # Build per-bar DataFrame
        per_bar = pd.DataFrame({
            "equity": equity_arr,
            "position": position_arr,
            "drawdown": drawdown_arr,
        }, index=timestamps)

        equity_curve = per_bar["equity"]

        config = {
            "initial_capital": self.initial_capital,
            "fee_rate": self.fee_rate,
            "leverage": self.leverage,
            "short_capital_fraction": self.short_capital_fraction,
            "bars_per_year": self.bars_per_year,
            "execution": execution,
            "seed": self.seed,
        }

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            per_bar=per_bar,
            config=config,
        )
        result.metrics = self._compute_metrics(result)
        return result

    def _close_position(
        self, trades, side, size, entry_price, exit_price,
        entry_idx, exit_idx, timestamps, entry_cash, exit_reason
    ) -> float:
        """Close a position and append to trades. Returns the new cash balance."""
        # Gross P&L on the position
        if side == 1:  # long
            pnl_gross = size * (exit_price - entry_price)
            capital_deployed = size * entry_price
        else:  # short
            pnl_gross = size * (entry_price - exit_price)
            capital_deployed = size * entry_price

        # Fees: entry fee was paid at entry (already debited from cash), exit fee paid here
        notional_exit = size * exit_price
        exit_fee = notional_exit * self.fee_rate
        entry_fee = capital_deployed * self.fee_rate  # approximate; was paid at entry
        total_fees = entry_fee + exit_fee

        # Net P&L for the round trip
        pnl_net = pnl_gross - total_fees

        return_pct = pnl_net / entry_cash if entry_cash > 0 else 0.0
        holding_hours = (timestamps[exit_idx] - timestamps[entry_idx]).total_seconds() / 3600.0

        trades.append(Trade(
            entry_time=timestamps[entry_idx],
            exit_time=timestamps[exit_idx],
            entry_price=entry_price,
            exit_price=exit_price,
            side=side,
            size=size,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            return_pct=return_pct,
            fees_paid=total_fees,
            holding_hours=holding_hours,
            exit_reason=exit_reason,
        ))

        # Return new cash: for longs we get back size*exit_price - exit_fee
        # for shorts we get back entry_cash + pnl_gross - exit_fee
        if side == 1:
            return notional_exit - exit_fee
        else:
            return entry_cash + pnl_gross - exit_fee

    def _compute_metrics(self, result: BacktestResult) -> dict:
        """Compute standard performance metrics."""
        eq = result.equity_curve
        n = len(eq)

        if n < 2:
            return {}

        # Per-bar log returns (well-defined for equity curve)
        log_rets = np.log(eq / eq.shift(1)).dropna()

        # Sharpe (annualized). Risk-free rate assumed 0.
        if log_rets.std() > 0:
            sharpe = np.sqrt(self.bars_per_year) * log_rets.mean() / log_rets.std()
        else:
            sharpe = 0.0

        # Sortino: only downside std
        downside = log_rets[log_rets < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = np.sqrt(self.bars_per_year) * log_rets.mean() / downside.std()
        else:
            sortino = 0.0

        # Max drawdown
        max_dd = result.per_bar["drawdown"].min()

        # Calmar = annualized return / max_drawdown_magnitude
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
        if years > 1e-6:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    ann_return = float((1 + total_return) ** (1 / years) - 1)
                if not np.isfinite(ann_return):
                    ann_return = float("nan")
            except (OverflowError, ValueError):
                ann_return = float("nan")
        else:
            ann_return = 0.0
        calmar = ann_return / abs(max_dd) if max_dd < 0 and np.isfinite(ann_return) else 0.0

        # Trade-level stats
        trades = result.trades
        if trades:
            tdf = pd.DataFrame([t.to_dict() for t in trades])
            total_trades = len(tdf)
            long_trades = int((tdf["side"] == 1).sum())
            short_trades = int((tdf["side"] == -1).sum())
            wins = (tdf["pnl_net"] > 0).sum()
            win_rate = wins / total_trades
            avg_holding = tdf["holding_hours"].mean()
            total_fees = tdf["fees_paid"].sum()
            gross_profit = tdf["pnl_gross"].sum()
            net_profit = tdf["pnl_net"].sum()
        else:
            total_trades = long_trades = short_trades = 0
            win_rate = avg_holding = total_fees = gross_profit = net_profit = 0.0

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "total_return": total_return,
            "annualized_return": ann_return,
            "total_trades": total_trades,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "win_rate": win_rate,
            "avg_holding_hours": avg_holding,
            "total_fees": total_fees,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
        }


# =====================================================================
# Convenience function
# =====================================================================
def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    open_col: str = "open",
    initial_capital: float = 1000.0,
    fee_rate: float = 0.0015,
    leverage: float = 1.0,
    short_capital_fraction: float = 0.75,
    execution: str = "next_open",
    seed: int = 42,
) -> BacktestResult:
    """Convenience wrapper that builds a Backtester and runs it."""
    bt = Backtester(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        leverage=leverage,
        short_capital_fraction=short_capital_fraction,
        seed=seed,
    )
    return bt.run(df, signal_col=signal_col, price_col=price_col,
                   open_col=open_col, execution=execution)
