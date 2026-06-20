"""
btc_qlearning.py
================
BTC/USDT tabular Q-learning strategy, refactored for clarity and rigor.

Differences from the original (archive/btc_strategy_original.py):
- TradingEnvironment is its own class (not nested), enabling testing/ablation
- State discretization is a pure function
- Reward function is pluggable: 'original' (P&L mix) or 'log_utility' (Moody-Saffell)
- The training loop is a separate function that returns the learned Q-table
- The strategy outputs a `signal` column (+1/-1/0) consumed by the backtester
  instead of executing trades inside the strategy itself
- All parameters are surfaced in a config dataclass (no hidden constants)

The interpretability-first design (tabular, inspectable policy) is preserved.

NOTE ON STATE SIZE
------------------
The original report claims '540 discrete states'. The actual state space size
is 1620 = 20 (pct bins) × 3 (position) × 3 (RSI signal) × 3 (EMA signal) × 3
(Aroon signal). The original report's '540' figure is an arithmetic error
(it counted only 4 of the 5 dimensions). The implementation here matches the
original code's actual behavior.

Author: Ankit Sinha (Roll 240102117)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # for relative imports inside src/

from indicators import rsi as compute_rsi, ema as compute_ema, aroon as compute_aroon


# =====================================================================
# Config
# =====================================================================
@dataclass
class BTCQLearningConfig:
    """All hyperparameters for the BTC Q-learning strategy."""

    # State space
    n_pct_bins: int = 20            # bins for percentage price change
    pct_clip: float = 5.0           # clip pct change to ±this
    n_signal_bins: int = 3          # ternary signals (-1, 0, +1)
    n_position_states: int = 3      # short, flat, long

    # Q-learning
    learning_rate: float = 0.05     # alpha
    discount: float = 0.95          # gamma
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay: float = 0.995    # epsilon *= decay each episode
    n_episodes: int = 1400

    # Indicators
    rsi_window: int = 14
    rsi_high: float = 75.0
    rsi_low: float = 35.0
    ema_short: int = 7
    ema_med: int = 14
    ema_long: int = 28
    aroon_window: int = 14

    # Execution / risk
    initial_balance: float = 10000.0
    min_trade_amount: float = 5000.0
    commission_rate: float = 0.0015
    max_short_position: float = 0.75
    stop_loss_pct: float = 0.05

    # Reward function
    reward_type: str = "original"   # 'original' or 'log_utility'
    drawdown_lambda: float = 1.0    # for log_utility: weight on drawdown penalty
    bankruptcy_reward: float = -1e8 # only for 'original' reward

    # Reproducibility
    seed: int = 42


# =====================================================================
# State features
# =====================================================================
def compute_features(df: pd.DataFrame, config: BTCQLearningConfig) -> pd.DataFrame:
    """
    Compute discrete state features for the BTC Q-learning agent.

    Adds columns: rsi_signal, ema_signal, aroon_signal, pct_change.
    """
    out = df.copy()

    # RSI signal: +1 if RSI > high, -1 if RSI < low, 0 otherwise
    rsi_vals = compute_rsi(out["close"], window=config.rsi_window)
    out["rsi_signal"] = 0
    out.loc[rsi_vals > config.rsi_high, "rsi_signal"] = 1
    out.loc[rsi_vals < config.rsi_low, "rsi_signal"] = -1

    # EMA ordering signal: +1 if EMA7 > EMA14 > EMA28, -1 if reverse, 0 otherwise
    ema_s = compute_ema(out["close"], span=config.ema_short)
    ema_m = compute_ema(out["close"], span=config.ema_med)
    ema_l = compute_ema(out["close"], span=config.ema_long)
    out["ema_signal"] = 0
    out.loc[(ema_s > ema_m) & (ema_m > ema_l), "ema_signal"] = 1
    out.loc[(ema_s < ema_m) & (ema_m < ema_l), "ema_signal"] = -1

    # Aroon signal: +1 if Aroon_Up > Aroon_Down, -1 if down > up
    aroon_up, aroon_down = compute_aroon(out["high"], out["low"], window=config.aroon_window)
    out["aroon_signal"] = 0
    out.loc[aroon_up > aroon_down, "aroon_signal"] = 1
    out.loc[aroon_up < aroon_down, "aroon_signal"] = -1

    # Percentage change
    out["pct_change"] = out["close"].pct_change() * 100

    return out


def get_state_index(
    rsi_signal: int, ema_signal: int, aroon_signal: int,
    position: int, pct_change: float,
    config: BTCQLearningConfig,
) -> int:
    """
    Convert (5 features) → single state index in [0, state_size).

    State size = n_pct_bins * 3 * 3 * 3 * 3 = 540 for defaults.
    """
    sig_map = {-1: 0, 0: 1, 1: 2}
    pos_map = {-1: 0, 0: 1, 1: 2}

    aroon_b = sig_map.get(int(aroon_signal), 1)
    rsi_b = sig_map.get(int(rsi_signal), 1)
    ema_b = sig_map.get(int(ema_signal), 1)
    pos_b = pos_map.get(int(position), 1)

    n_sig = config.n_signal_bins
    n_pct = config.n_pct_bins
    n_pos = config.n_position_states

    bin_edges = np.linspace(-config.pct_clip, config.pct_clip, n_pct + 1)
    pct_clipped = np.clip(pct_change if not np.isnan(pct_change) else 0.0,
                          -config.pct_clip, config.pct_clip)
    pct_b = int(np.clip(np.digitize(pct_clipped, bin_edges, right=False) - 1, 0, n_pct - 1))

    state_size = n_pct * n_pos * n_sig * n_sig * n_sig
    state_index = (
        aroon_b * (n_sig ** 2 * n_pct * n_pos) +
        rsi_b * (n_sig * n_pct * n_pos) +
        ema_b * (n_pct * n_pos) +
        pos_b * n_pct +
        pct_b
    )
    return int(np.clip(state_index, 0, state_size - 1))


def state_size(config: BTCQLearningConfig) -> int:
    """Total number of discrete states."""
    return (config.n_pct_bins * config.n_position_states *
            config.n_signal_bins ** 3)


# =====================================================================
# TradingEnvironment
# =====================================================================
ACTION_HOLD = 0
ACTION_LONG = 1   # enter long (or reverse from short)
ACTION_EXIT_LONG = 2
ACTION_SHORT = 3  # enter short (or reverse from long)
ACTION_EXIT_SHORT = 4
N_ACTIONS = 5


class TradingEnvironment:
    """
    Trading environment for BTC Q-learning.

    Stateful but resettable. The reward function is pluggable.

    Reward conventions:
    - 'original': matches the archive code — mixes commissions, realized P&L,
      unrealized P&L, large bankruptcy penalty, and a flat-position penalty.
    - 'log_utility': Moody-Saffell style — r_t = log(W_t / W_{t-1}) with
      a drawdown penalty. Scale-invariant, no flat-position incentive.
    """

    def __init__(self, prices: np.ndarray, features: dict, config: BTCQLearningConfig):
        """
        Parameters
        ----------
        prices : np.ndarray of close prices
        features : dict with keys 'rsi_signal', 'ema_signal', 'aroon_signal',
                                  'pct_change' — each an np.ndarray of same length as prices.
        config : BTCQLearningConfig
        """
        self.prices = np.asarray(prices, dtype=float)
        self.rsi = np.asarray(features["rsi_signal"], dtype=int)
        self.ema = np.asarray(features["ema_signal"], dtype=int)
        self.aroon = np.asarray(features["aroon_signal"], dtype=int)
        self.pct = np.asarray(features["pct_change"], dtype=float)
        self.n_steps = len(self.prices)
        self.config = config
        self.reset()

    def reset(self):
        """Reset to initial state."""
        cfg = self.config
        self.current_step = 0
        self.balance = cfg.initial_balance
        self.holdings = 0.0
        self.position = 0  # -1 short, 0 flat, +1 long
        self.entry_price = 0.0
        self.net_worth = cfg.initial_balance
        self.last_worth = cfg.initial_balance
        self.peak_worth = cfg.initial_balance
        self.trades_log: list = []  # list of dicts with step, signal, price
        return self._obs()

    def _obs(self):
        return (
            self.aroon[self.current_step],
            self.rsi[self.current_step],
            self.ema[self.current_step],
            self.position,
            self.pct[self.current_step],
        )

    def _net_worth_at(self, price: float) -> float:
        """Compute net worth at a given price."""
        if self.position == 1:
            return self.balance + self.holdings * price
        elif self.position == -1:
            # holdings is negative for shorts; cash includes short proceeds
            return self.balance + self.holdings * price
        else:
            return self.balance

    def _compute_reward_original(self, prev_worth: float, price: float, commission_paid: float) -> float:
        """Original reward function (matches archive code)."""
        cfg = self.config
        nw = self._net_worth_at(price)

        if self.position != 0:
            # While holding, reward is the change in net worth
            return nw - prev_worth
        else:
            # While flat, an inactivity penalty proportional to abs price move
            if self.current_step > 0 and self.prices[self.current_step - 1] > 0:
                inactivity = abs(price - self.prices[self.current_step - 1]) \
                             * nw / self.prices[self.current_step - 1]
                return -inactivity
            return 0.0

    def _compute_reward_log_utility(self, prev_worth: float, price: float, commission_paid: float) -> float:
        """
        Moody-Saffell (2001) log-utility differential reward.

        r_t = log(W_t / W_{t-1}) - lambda * max(0, drawdown_increase)

        Properties:
        - Scale-invariant (per-step log-return)
        - Naturally penalizes large losses more than gains (concavity of log)
        - Drawdown penalty integrates smoothly, no cliff at bankruptcy
        - Being flat → W_t = W_{t-1} → r_t = 0 (correct baseline, no inactivity bias)
        """
        cfg = self.config
        nw = self._net_worth_at(price)
        # Guard against W=0 (bankruptcy)
        if nw <= 0 or prev_worth <= 0:
            return -10.0  # large but bounded penalty

        log_ret = np.log(nw / prev_worth)

        # Drawdown component
        self.peak_worth = max(self.peak_worth, nw)
        prev_dd = max(0.0, (self.peak_worth - prev_worth) / self.peak_worth) if self.peak_worth > 0 else 0.0
        curr_dd = max(0.0, (self.peak_worth - nw) / self.peak_worth) if self.peak_worth > 0 else 0.0
        dd_increase = max(0.0, curr_dd - prev_dd)

        return log_ret - cfg.drawdown_lambda * dd_increase

    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        Take one step in the environment.

        Returns (next_obs, reward, done).
        """
        cfg = self.config
        price = self.prices[self.current_step]
        prev_worth = self._net_worth_at(price)
        commission_paid = 0.0
        done = False

        # --- Action handling ---
        if action == ACTION_LONG:  # enter long (or reverse from short)
            if self.position == -1:
                # Close short
                cost = -self.holdings * price
                commission = cost * cfg.commission_rate
                self.balance -= (cost + commission)
                commission_paid += commission
                self.holdings = 0.0
                self.position = 0
                if self.balance < cfg.min_trade_amount:
                    # Bankrupt during short close
                    if cfg.reward_type == "original":
                        return self._obs(), cfg.bankruptcy_reward, True
                    return self._obs(), -10.0, True
                self.trades_log.append({
                    "step": self.current_step, "price": price,
                    "trade_type": "exit_short", "commission": commission,
                })
            if self.position == 0 and self.balance >= cfg.min_trade_amount:
                # Open long
                investment = self.balance * (1 - cfg.commission_rate)
                commission = self.balance * cfg.commission_rate
                commission_paid += commission
                self.holdings = investment / price
                self.balance = 0.0
                self.position = 1
                self.entry_price = price
                self.last_worth = prev_worth
                self.trades_log.append({
                    "step": self.current_step, "price": price,
                    "trade_type": "enter_long", "commission": commission,
                })

        elif action == ACTION_EXIT_LONG:
            if self.position == 1:
                gross = self.holdings * price
                commission = gross * cfg.commission_rate
                commission_paid += commission
                self.balance += gross - commission
                self.holdings = 0.0
                self.position = 0
                self.trades_log.append({
                    "step": self.current_step, "price": price,
                    "trade_type": "exit_long", "commission": commission,
                })

        elif action == ACTION_SHORT:  # enter short (or reverse from long)
            if self.position == 1:
                gross = self.holdings * price
                commission = gross * cfg.commission_rate
                self.balance += gross - commission
                commission_paid += commission
                self.holdings = 0.0
                self.position = 0
                self.trades_log.append({
                    "step": self.current_step, "price": price,
                    "trade_type": "exit_long", "commission": commission,
                })
            if self.position == 0 and self.balance >= cfg.min_trade_amount:
                # Open short
                short_value = self.balance * cfg.max_short_position
                commission = short_value * cfg.commission_rate
                commission_paid += commission
                self.holdings = -short_value / price
                self.balance += short_value - commission
                self.position = -1
                self.entry_price = price
                self.last_worth = prev_worth
                self.trades_log.append({
                    "step": self.current_step, "price": price,
                    "trade_type": "enter_short", "commission": commission,
                })

        elif action == ACTION_EXIT_SHORT:
            if self.position == -1:
                cost = -self.holdings * price
                commission = cost * cfg.commission_rate
                self.balance -= (cost + commission)
                commission_paid += commission
                if self.balance < cfg.min_trade_amount:
                    if cfg.reward_type == "original":
                        return self._obs(), cfg.bankruptcy_reward, True
                    return self._obs(), -10.0, True
                self.holdings = 0.0
                self.position = 0
                self.trades_log.append({
                    "step": self.current_step, "price": price,
                    "trade_type": "exit_short", "commission": commission,
                })

        # --- Stop-loss check ---
        if self.position == 1 and price <= self.entry_price * (1 - cfg.stop_loss_pct):
            gross = self.holdings * price
            commission = gross * cfg.commission_rate
            self.balance += gross - commission
            commission_paid += commission
            self.holdings = 0.0
            self.position = 0
            self.trades_log.append({
                "step": self.current_step, "price": price,
                "trade_type": "stop_loss_long", "commission": commission,
            })
        elif self.position == -1 and price >= self.entry_price * (1 + cfg.stop_loss_pct):
            cost = -self.holdings * price
            commission = cost * cfg.commission_rate
            self.balance -= (cost + commission)
            commission_paid += commission
            if self.balance < cfg.min_trade_amount:
                if cfg.reward_type == "original":
                    return self._obs(), cfg.bankruptcy_reward, True
                return self._obs(), -10.0, True
            self.holdings = 0.0
            self.position = 0
            self.trades_log.append({
                "step": self.current_step, "price": price,
                "trade_type": "stop_loss_short", "commission": commission,
            })

        # --- Compute reward ---
        if cfg.reward_type == "original":
            reward = self._compute_reward_original(prev_worth, price, commission_paid) - commission_paid
        elif cfg.reward_type == "log_utility":
            reward = self._compute_reward_log_utility(prev_worth, price, commission_paid)
        else:
            raise ValueError(f"Unknown reward_type: {cfg.reward_type}")

        # Update net worth and step
        self.net_worth = self._net_worth_at(price)
        self.current_step += 1

        # End of data?
        if self.current_step >= self.n_steps - 1:
            done = True
            # Force close
            if self.position == 1:
                price_final = self.prices[self.current_step]
                gross = self.holdings * price_final
                commission = gross * cfg.commission_rate
                self.balance += gross - commission
                self.trades_log.append({
                    "step": self.current_step, "price": price_final,
                    "trade_type": "exit_long_eod", "commission": commission,
                })
                self.holdings, self.position = 0.0, 0
            elif self.position == -1:
                price_final = self.prices[self.current_step]
                cost = -self.holdings * price_final
                commission = cost * cfg.commission_rate
                self.balance -= (cost + commission)
                self.trades_log.append({
                    "step": self.current_step, "price": price_final,
                    "trade_type": "exit_short_eod", "commission": commission,
                })
                self.holdings, self.position = 0.0, 0

        return self._obs(), reward, done


# =====================================================================
# Q-Learning training
# =====================================================================
def train_q_agent(
    env: TradingEnvironment,
    config: BTCQLearningConfig,
    verbose: bool = False,
) -> np.ndarray:
    """
    Train a tabular Q-agent via epsilon-greedy Q-learning.

    Returns the learned Q-table (state_size × N_ACTIONS).
    """
    np.random.seed(config.seed)
    ss = state_size(config)
    Q = np.zeros((ss, N_ACTIONS))
    epsilon = config.epsilon_start

    for ep in range(config.n_episodes):
        obs = env.reset()
        done = False
        while not done:
            aroon, rsi_sig, ema_sig, pos, pct = obs
            s = get_state_index(rsi_sig, ema_sig, aroon, pos, pct, config)
            if np.random.rand() < epsilon:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[s]))

            next_obs, r, done = env.step(a)
            aroon2, rsi_sig2, ema_sig2, pos2, pct2 = next_obs
            s2 = get_state_index(rsi_sig2, ema_sig2, aroon2, pos2, pct2, config)

            old = Q[s, a]
            target = r + config.discount * np.max(Q[s2])
            new = (1 - config.learning_rate) * old + config.learning_rate * target
            if np.isfinite(new):
                Q[s, a] = new

            obs = next_obs

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        if verbose and (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{config.n_episodes}, eps={epsilon:.3f}")

    return Q


# =====================================================================
# Evaluation - convert Q-policy to signal series
# =====================================================================
def policy_to_signals(
    df_features: pd.DataFrame,
    Q: np.ndarray,
    config: BTCQLearningConfig,
) -> pd.Series:
    """
    Run the learned greedy policy on test data, emitting a +1/-1/0 signal per bar.

    This output feeds into the in-house backtester.
    """
    prices = df_features["close"].values
    rsi = df_features["rsi_signal"].values
    ema = df_features["ema_signal"].values
    aroon = df_features["aroon_signal"].values
    pct = df_features["pct_change"].values

    n = len(prices)
    signals = np.zeros(n, dtype=int)
    position = 0  # track virtual position to compute state correctly

    for t in range(n):
        s = get_state_index(rsi[t], ema[t], aroon[t], position, pct[t], config)
        a = int(np.argmax(Q[s]))

        # Map action → new position
        if a == ACTION_LONG:
            position = 1
        elif a == ACTION_EXIT_LONG:
            if position == 1:
                position = 0
        elif a == ACTION_SHORT:
            position = -1
        elif a == ACTION_EXIT_SHORT:
            if position == -1:
                position = 0
        # ACTION_HOLD: position unchanged

        signals[t] = position

    return pd.Series(signals, index=df_features.index, name="signal")


# =====================================================================
# High-level entry point
# =====================================================================
def run_btc_qlearning_strategy(
    df: pd.DataFrame,
    train_end_date: str = "2022-12-31",
    config: Optional[BTCQLearningConfig] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    End-to-end: compute features → train Q-agent on train portion → emit signals on full data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: datetime, open, high, low, close, volume
    train_end_date : str
        Date splitting train from test.
    config : BTCQLearningConfig
        Hyperparameters.
    verbose : bool
        Print training progress.

    Returns
    -------
    (df_with_signals, Q_table)
        df_with_signals has all features + a 'signal' column for the backtester.
        Q_table is the learned Q-table (inspectable).
    """
    if config is None:
        config = BTCQLearningConfig()

    df_feat = compute_features(df, config)
    df_feat = df_feat.dropna(subset=["pct_change", "rsi_signal", "ema_signal", "aroon_signal"])

    # Train/test split
    if "datetime" in df_feat.columns:
        df_feat_indexed = df_feat.set_index("datetime")
    else:
        df_feat_indexed = df_feat
    df_feat_indexed.index = pd.to_datetime(df_feat_indexed.index)

    train = df_feat_indexed.loc[:train_end_date]
    if len(train) < 100:
        raise ValueError(f"Training set too small: {len(train)} bars")

    train_features = {
        "rsi_signal": train["rsi_signal"].values,
        "ema_signal": train["ema_signal"].values,
        "aroon_signal": train["aroon_signal"].values,
        "pct_change": train["pct_change"].values,
    }
    env = TradingEnvironment(train["close"].values, train_features, config)

    if verbose:
        print(f"Training Q-agent on {len(train)} bars, {config.n_episodes} episodes...")
    Q = train_q_agent(env, config, verbose=verbose)
    if verbose:
        print("Training complete.")

    # Emit signals on full data
    signals = policy_to_signals(df_feat_indexed, Q, config)
    df_out = df_feat_indexed.copy()
    df_out["signal"] = signals

    return df_out, Q


# =====================================================================
# Q-table serialization
# =====================================================================
def save_q_table(
    Q: np.ndarray,
    config: BTCQLearningConfig,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Persist a trained Q-table along with the config that produced it.

    Saving the config alongside the Q-table is critical because the policy's
    behavior depends on the state discretization (`get_state_index`), which
    depends on n_pct_bins, n_signal_bins, n_holdings_states. Loading a Q-table
    with a mismatched config produces silently-wrong predictions.

    Parameters
    ----------
    Q : np.ndarray
        Q-table of shape (state_size, N_ACTIONS).
    config : BTCQLearningConfig
        The config the agent was trained with.
    path : str
        Output file path (.pkl recommended).
    metadata : dict, optional
        Free-form metadata: training date, data range, total episodes, etc.
        Recorded in the artifact for audit purposes.
    """
    import pickle
    from datetime import datetime
    payload = {
        "format_version": 1,
        "q_table": Q,
        "config": config,  # dataclass; pickles cleanly
        "saved_at": datetime.now().isoformat(),
        "shape": list(Q.shape),
        "metadata": metadata or {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_q_table(
    path: str,
    expected_config: Optional[BTCQLearningConfig] = None,
    strict: bool = True,
) -> Tuple[np.ndarray, BTCQLearningConfig, dict]:
    """
    Load a Q-table and its config, optionally verifying compatibility.

    Parameters
    ----------
    path : str
        Path to a Q-table artifact saved by `save_q_table`.
    expected_config : BTCQLearningConfig, optional
        If provided, verify the saved config's state-affecting fields match.
        Other fields (learning rate, episodes) can differ — they only matter
        during training, not inference.
    strict : bool
        If True and `expected_config` is given, raise on any mismatch in
        state-affecting fields. If False, only warn.

    Returns
    -------
    (Q, config, metadata) tuple.
        Q : np.ndarray
        config : BTCQLearningConfig (the one the Q-table was trained with)
        metadata : dict (saved_at, shape, free-form metadata)

    Raises
    ------
    FileNotFoundError, ValueError, or RuntimeError on incompatible state
    space if strict=True.
    """
    import pickle
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "q_table" not in payload:
        raise ValueError(f"File {path} does not look like a Q-table artifact")

    Q = payload["q_table"]
    config = payload["config"]
    metadata = {
        "saved_at": payload.get("saved_at"),
        "shape": payload.get("shape"),
        **payload.get("metadata", {}),
    }

    # State-affecting fields — these change the discretization and thus the policy
    STATE_FIELDS = ("n_pct_bins", "pct_clip", "n_signal_bins", "n_position_states",
                    "rsi_high", "rsi_low")

    if expected_config is not None:
        mismatches = []
        for f in STATE_FIELDS:
            if not hasattr(config, f) or not hasattr(expected_config, f):
                continue
            if getattr(config, f) != getattr(expected_config, f):
                mismatches.append(
                    f"  {f}: saved={getattr(config, f)} vs expected={getattr(expected_config, f)}"
                )
        if mismatches:
            msg = ("Loaded Q-table config differs from expected on "
                    f"state-affecting fields:\n" + "\n".join(mismatches))
            if strict:
                raise RuntimeError(msg)
            else:
                import warnings
                warnings.warn(msg)

    return Q, config, metadata
