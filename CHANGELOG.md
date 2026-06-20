# Changelog

This project was built in seven milestones, each producing tested, working code.
The changelog below describes what each milestone delivered and why it mattered.

## v0.1.0 — Initial release

### Milestone 1: Infrastructure
**Why:** The original Inter IIT code depended on a closed-source SDK (Untrade). Nothing downstream is possible without replacing it.

- `src/backtester.py` (~500 lines): deterministic event-loop backtester. No-lookahead execution (signal at t executes at t+1's open). Configurable fees, leverage, position sizing. Returns a `BacktestResult` dataclass with trades, equity curve, per-bar log, and metrics.
- `src/indicators.py` (~420 lines): all technical indicators as pure causal functions. RSI, ATR, EMA, Bollinger, Supertrend, Aroon, Kalman filter (1D, one-sided), R/S Hurst, DFA Hurst, CUSUM regime detection with rolling/EWMA sigma options.
- 11 backtester tests + 14 indicator tests.

### Milestone 2: ETH strategy refactor
**Why:** Without separating indicators from strategy logic, validation experiments are impossible.

- `src/strategies/eth_regime_confirmation.py` (~400 lines): refactored ETH strategy using the new modules. Pluggable Hurst method (`'rs'` vs `'dfa'`) and CUSUM sigma method (`'rolling'` vs `'ewma'`).
- 5 smoke tests.

### Milestone 3: BTC strategy refactor
**Why:** The original BTC Q-learning code had a reward function bug (flat-position penalty) causing 30:1 long/short asymmetry. The refactor surfaces this and provides a fix.

- `src/strategies/btc_qlearning.py` (~610 lines): `TradingEnvironment` class, pluggable reward (`'original'` or `'log_utility'` Moody-Saffell), inspectable Q-table.
- Found and documented the "540 states" report error (actual is 1620).
- 10 smoke tests.

### Milestone 4: Bootstrap confidence intervals
**Why:** Point estimates without uncertainty are misleading. The Politis-Romano stationary bootstrap preserves trade-return autocorrelation.

- `src/validation/bootstrap.py` (~280 lines): CIs for Sharpe, Sortino, total return, max drawdown, win rate.
- 12 tests including a test that verifies block-bootstrap preserves AR(1) autocorrelation.

### Milestone 5: Data infrastructure
**Why:** Validation experiments are only meaningful on data the strategies haven't been tuned to.

- `src/data_io/loader.py` (~200 lines): robust CSV loader handling ISO/Unix-ms timestamps, deduplication, gap detection, BTC/ETH alignment.
- `src/data_io/binance_puller.py` (~180 lines): public-API data puller, paginating the 1000-bar limit.
- `src/validation/parity_check.py` (~270 lines): quarterly comparison harness against original Inter IIT summaries.
- 11 tests.

### Milestone 6: RL vs rule-based experiment
**Why:** The N4 novelty claim ("interpretability-first RL adds value over rules") was untested. This milestone runs the experiment.

- `src/strategies/btc_rule_based.py` (~190 lines): three baselines (momentum confluence, majority vote, mean reversion).
- `src/validation/rl_vs_rule_experiment.py` (~260 lines): comparison harness with seed variance.
- Caught a seed-propagation bug (all seeds converging to identical Q-tables); regression test added.
- 13 tests.

### Milestone 7: Walk-forward + Deflated Sharpe Ratio
**Why:** The most important question — does the strategy have edge after multiple-testing correction? — requires both walk-forward (to generate trials) and DSR (to correct for selection bias).

- `src/validation/walkforward.py` (~290 lines): rolling and expanding window CV with purging and embargo.
- `src/validation/deflated_sharpe.py` (~210 lines): Bailey & López de Prado (2014) DSR with skewness/kurtosis correction.
- On synthetic data: a strategy with 93.8% positive-Sharpe folds and Sharpe 3.71 produced DSR=0.000. The machinery correctly identified that the apparent edge was selection bias.
- 18 tests covering fold generation, purge/embargo, and DSR limit cases.

### Milestone 8: ETH gate ablation
**Why:** The report's N3 novelty claim ("hierarchical gate-then-signal architecture") was untested. This milestone runs the ablation that confirms or refutes which gates actually contribute.

- `src/validation/gate_ablation.py` (~250 lines): runs the strategy 5 times (all gates on, each gate individually off, all off) and reports per-gate attribution.
- Added 3 boolean flags to `ETHStrategyConfig` (`enable_hurst_gate`, `enable_correlation_gate`, `enable_atr_gate`) that default to True.
- **On synthetic data: only the ATR gate matters** (removing it drops Sharpe by 0.25). The Hurst and correlation gates are non-binding (removing them changes nothing). N3 is partially refuted — the architecture's value is concentrated in one gate, not distributed across three.
- 9 tests including invariants ("disabling a gate cannot reduce trade count").

### Milestone 9: Hurst window × method ablation
**Why:** The report's L4 limitation proposes replacing R/S Hurst with DFA; this milestone tests whether that fix actually changes outcomes.

- `src/validation/hurst_ablation.py` (~315 lines): 8-config grid (4 windows × 2 methods) with both Hurst-series diagnostics and downstream strategy metrics.
- Three findings on synthetic data:
  - R/S → DFA at the same window produces *identical* strategy outcomes (the L4 fix is a no-op here)
  - DFA values cluster around 1.5 (it returns α=H+1 for non-stationary series); the report's "H > 0.5" gate would silently disable when using DFA literally
  - Window size matters more than method choice (Sharpe drops 1.34 → 0.66 from w=120 to w=500)
- 11 tests.

### Milestone 10: Cold OOS evaluator
**Why:** The report's L3 — "the full 2020-2023 window was visible during development; a genuinely cold test requires data not available at design time." This milestone produces the runner script.

- `src/validation/cold_oos.py` (~340 lines): CLI tool that pulls 2024+ Binance data via `binance_puller`, runs both strategies with **frozen** parameters (no hyperparameters exposed in the CLI by design), backtests, computes bootstrap CIs, writes structured outputs.
- The frozen-parameter guarantee: changing strategy config requires editing source code, leaving an audit trail.
- Two modes: **true cold OOS** (preferred) when `--btc-q-table` is provided, and **fallback retrain** with a documented caveat when no pre-trained Q-table is supplied.
- 6 tests covering artifact generation, JSON schema, both BTC modes (fallback + true cold), and summary writing.

### Milestone 11: Q-table serialization
**Why:** Without serialization, the BTC agent must be retrained on cold data, which isn't a true OOS test. This milestone fixes that.

- Added `save_q_table` and `load_q_table` to `src/strategies/btc_qlearning.py`.
- Saved artifacts include the Q-table, the config that produced it, a timestamp, and free-form metadata.
- Load-time compatibility check: state-affecting fields (n_pct_bins, pct_clip, n_signal_bins, n_position_states, rsi_high, rsi_low) must match between saved and expected config. Strict mode raises on mismatch; non-strict mode warns.
- Integrated into `cold_oos.py` via the `--btc-q-table` flag. With a pre-trained Q-table, the BTC strategy runs in true-cold mode (no retraining).
- 8 tests covering round-trip, metadata preservation, strict/non-strict compatibility checks.

### Milestone 12: Reward function comparison (tests L5 fix)
**Why:** The original BTC strategy reports 30:1 long/short split, suggesting reward-induced bias. The Moody-Saffell log-utility reward was implemented as a fix (Milestone 3) but never actually compared.

- `src/validation/reward_comparison.py` (~270 lines): for each reward type × N seeds, train and evaluate the agent, recording long/short balance, Sharpe, and other metrics.
- **Finding on synthetic data**: log-utility reward produces ZERO trades across all seeds; original reward produces a few. Verdict logic articulates the real meaning: log-utility removes the activity-bias from the original reward, so when no edge exists, the agent correctly stays flat — but synthetic data has no edge, so log-utility's "right answer" is null performance. **The L5 fix changes what the agent optimizes, and the consequences are data-dependent.** Real-data verdict requires real data.
- 7 tests including a regression test for seed propagation through `reward_type` config changes.

---

## Statistics

- **6 runnable demos**, each producing actual experimental output
- **135 tests** across 13 test files
- **~3,500 lines** of production code
- **No external SDK dependencies** — runs anywhere with NumPy/Pandas/SciPy
- **Original Inter IIT code preserved** in `archive/`
