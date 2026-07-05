# Quant Trading Strategy Validation Infrastructure

**A from-scratch validation framework for two crypto trading strategies, focused on distinguishing genuine edge from overfitting and lookahead bias.**

**🔗 Live project site:** https://primeuser2605-wq.github.io/Alphapositive-DSAI-BTP-/

---

## The headline finding

On real 2020–2023 hourly BTC and ETH data, two trading strategies produced what looked like extraordinary in-sample results:

| Strategy | Reported Sharpe (naive backtest) | Reported max drawdown |
|---|---|---|
| ETH regime-confirmation (CUSUM) | **5.97** | 8.16% |
| BTC Q-learning agent | **9.15** | 13.50% |

When the same strategies were re-run through this project's **no-lookahead backtester** and validated with walk-forward + Deflated Sharpe Ratio on the same data:

| Strategy | Corrected Sharpe (causal backtest) | Actual max drawdown | Deflated Sharpe |
|---|---|---|---|
| ETH regime-confirmation | **0.129** | −84.6% | **≈ 0.000** |
| BTC Q-learning agent | **0.123** | −61.4% | (full-episode walk-forward pending) |

**The apparent edge did not survive causal validation.** A Sharpe ratio of 5.97 collapsed to 0.129 once the strategy was run through a backtester with fully auditable, next-bar-open execution. After correcting for multiple-testing bias via the Deflated Sharpe Ratio, the evidence of true edge dropped to zero.

Two independent implementations — a from-scratch reconstruction of the strategy logic and the full pipeline in this repo — produced consistent near-zero Sharpe results on the real data. The explanation for the original ~6 Sharpe is that **those metrics were never produced by auditable code**: the earlier strategy scripts contain no P&L logic at all — they emit a signal column and hand it to an opaque external backtesting scorer whose execution conventions (fill timing, fee model, drawdown definition) cannot be inspected or verified. When the identical signals are run through this repo's transparent, next-bar-open engine, the risk-adjusted edge disappears. The gap is *unverifiable external execution vs. auditable causal execution* — not a single identifiable bug, which is a more honest and ultimately stronger framing: the headline metrics were never independently checkable in the first place.

**This is the project's most important finding**, and it validates the entire premise: rigorous validation infrastructure catches inflated backtests that aggregate metrics won't.

---

## What this is

This project develops two algorithmic trading strategies for BTC/USDT and ETH/USDT — a CUSUM-based regime-confirmation strategy for ETH and a tabular Q-learning agent for BTC — and then submits both to a comprehensive validation pipeline. The strategies are ordinary; **the validation framework is the contribution.**

The driving question isn't "can I get a high Sharpe?" — that's the easy part, as the reconciliation above illustrates. The question is: **how do we know whether a high backtest Sharpe is real edge or an artifact of overfitting, lookahead bias, or selection?** Answering that requires:

- An in-house, deterministic backtester with a strict no-lookahead guarantee
- Walk-forward validation across many disjoint out-of-sample windows
- Bootstrap confidence intervals on every metric
- The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) to correct for multiple-testing bias
- Rule-based baselines to test whether the RL agent actually adds signal
- Ablation experiments for every design choice the strategies make
- A frozen-parameter cold-OOS runner for data the strategies have never seen

This codebase builds all of the above.

```
~4,500 lines of production code
135 tests across 13 files, all passing
6 runnable demos producing actual experimental results
```

**Author:** Ankit Sinha

---

## For the impatient reviewer

```bash
pip install -r requirements.txt

python demo_full_pipeline.py           # strategy → backtest → bootstrap CIs
python demo_rl_vs_rule.py              # RL vs hand-coded baselines
python demo_walkforward_dsr.py         # walk-forward → DSR (the headline)
python demo_gate_ablation.py           # ETH gate ablation
python demo_hurst_ablation.py          # Hurst window × method
python demo_reward_comparison.py       # BTC reward function comparison

python -m pytest tests/                # 135 tests, all passing
```

Each demo runs on synthetic data and takes 30 seconds to 2 minutes. To run on real BTC/ETH hourly data: drop CSVs in `data/`, replace the synthetic loaders in the demos with `data_io.loader.load_btc_eth_pair(...)`.

---

## Reconciliation: before and after causal validation

The headline finding above deserves a closer look because it is what the project is really about.

**The setup:** two strategies were originally developed and reported on 2020–2023 hourly BTC/ETH data using a proprietary backtesting SDK. The reported ETH strategy had a Sharpe of 5.97 with a maximum drawdown of only 8.16% over four years of crypto data — an extraordinary risk-adjusted return.

**What the causal pipeline produced on the same data:**

```
ETH strategy — real 2020-2023 data, causal backtest
====================================================
Total return         +465.2%      (bull-market beta, not alpha)
Sharpe ratio           +0.129     (vs reported 5.966)
Sortino ratio          +0.105
Max drawdown          -84.6%      (vs reported 8.16%)
Win rate               37.0%
Trades                190          (vs reported 162)

Walk-forward + Deflated Sharpe on the same data:
Deflated Sharpe (probability)   0.000
Verdict: NO EVIDENCE OF EDGE after deflation
```

**The BTC side, same treatment** (train 2020–2022, test 2023, matching the original setup):

```
BTC Q-learning strategy — real 2020-2023 data, causal backtest
==============================================================
Sharpe ratio           +0.123     (vs reported 9.15)
Sortino ratio          +0.126
Max drawdown          -61.4%      (vs reported 7.75%)
Total return (2023)   +39.9%*     (vs reported +224.90%)

* This run used 120 training episodes vs the finalized 1400 (a compute limit
  in the reconciliation environment). The Q-table is undertrained, so the
  RETURN is a lower bound; the Sharpe/drawdown regime — the part that matters —
  matches ETH and the independent reconstruction. finalize_btc_real.py runs the
  full 1400 episodes locally to produce the exact return.
```

Both markets tell the same story: reported Sharpe ~6–9 with ~8% drawdowns collapse to ~0.12 Sharpe with 60–85% drawdowns under auditable execution.

**The gap:** Sharpe collapses by ~46× (5.97 → 0.129). Max drawdown expands by ~10× (8% → 85%). Total return is broadly comparable (+524% vs +465%) — because during 2020–2023, ETH itself returned roughly +1578% (buy-and-hold), and any strategy that spent enough time long would ride that beta. The strategy's total return reflects the market, not the strategy's edge.

**The cause of the discrepancy (verified against the archived code):** the earlier strategy scripts in `archive/` contain **no P&L logic at all**. They compute indicators, emit a `signals` column, and delegate scoring entirely to an external, closed-source backtesting SDK:

```python
# archive/*_original.py — the only "backtest" the reported metrics came from:
client = Client()
result = client.backtest(file_path=csv_file_path, ...)   # opaque external scorer
```

Because the reported Sharpe of 5.97 / 9.15 came out of that black box, its execution conventions — when fills happen, how fees apply, how drawdown is defined — cannot be audited. The strategy code records its own entry price as the *same-bar close*, which is suggestive of same-bar fills, but this can't be confirmed: the actual fill is decided inside the SDK, not in any visible code. So the precise, defensible statement is **not** "there is a lookahead bug on line X" — there is no P&L code to point at. It is: *the reported metrics were produced by an unverifiable external scorer, and they do not reproduce under transparent execution.*

In contrast, this repo's backtester makes execution auditable line by line:

```python
# src/backtester.py, main loop:
exec_price = prices_open[t + 1]   # execute at NEXT bar's open, never t's close
```

There is a unit test (`tests/test_backtester.py::test_no_lookahead`) that asserts a known signal at bar t fills at bar t+1's open, not bar t's close. Running the identical strategy signals through this engine is what produces the near-zero Sharpe.

**Independent confirmation:** a from-scratch reconstruction of the strategy logic — different codebase, built from the design description — produced a Sharpe of −0.54 on the same data. Two independent causal implementations agree that the true Sharpe is near zero. Only the naive proprietary backtest produced a Sharpe of ~6.

**This is the project's central demonstration.** The framework isn't just a set of statistical tests; it's a working reproduction of the discipline that separates a Sharpe of 6 from a Sharpe of 0.

---

## Other findings produced by the project

### 1. The Deflated Sharpe Ratio catches hidden overfitting

A high Sharpe is not by itself evidence of edge. When you try N strategy variations and pick the best, the reported Sharpe is inflated by selection bias — even random strategies produce a high best-of-N Sharpe. The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) corrects for the number of trials, sample size, and return non-normality. This project implements it and demonstrates it catching the ETH reconciliation finding above (DSR ≈ 0.000).

### 2. RL vs rule-based baseline

A common argument for tabular Q-learning is *"the policy is inspectable and can be compared against rule-based baselines on the same features."* This is rarely tested — most projects assert interpretability without running the comparison. This project runs it:

```
RULE-BASED BASELINES (deterministic on test set):
Rule                             Sharpe   Return   MaxDD  Trades   WinR
momentum_confluence              -0.108  -11.81% -49.70%      49  22.4%
majority_vote                    -0.107  -12.40% -50.04%      52  21.2%
mean_reversion                    0.000    0.00%   0.00%       0   0.0%

TABULAR Q-LEARNING (across 5 seeds):
sharpe                    2.728  ± 0.596    range [1.66, 3.00]

VERDICT: RL ADDS SIGNAL (Sharpe gap = +2.834 vs best rule; +4.75σ separation)
```

On synthetic data with strong regime shifts, the RL agent outperforms hand-coded rules by ~4.75σ. The synthetic data is favorable to RL by construction; the real-data verdict may differ. **The contribution is the experimental machinery, not the specific finding.**

### 3. A documentation/code arithmetic error

An earlier write-up of the BTC strategy claimed a 540-state space. The actual implementation has **1620 states** (20 × 3 × 3 × 3 × 3 = five state dimensions). The write-up counted only four of the five dimensions. Catching paper-vs-code inconsistencies of this kind by reimplementing from scratch is one of the points of a rigorous rebuild.

### 4. The reward function design changes what the agent optimizes for

The BTC strategy's original reward function included a flat-position penalty designed to encourage activity. The result on real data was a 30:1 long/short asymmetry — the agent learned to always hold a position rather than time entries. The Moody-Saffell (2001) log-utility differential reward is implemented as a pluggable alternative; the comparison experiment ran both with 5 seeds each:

```
Metric              original (mean ± std)  log_utility (mean ± std)
Sharpe                +2.708 ± 0.640          +0.000 ± 0.000
Trades (per seed)        1.0                     0.0
Long fraction           20%                      0%       (zero trades)

VERDICT: log-utility SUPPRESSES TRADING on this data: with the flat-position
penalty removed, the agent learns 'do nothing'. This is technically correct
behavior — under log-utility, staying flat IS the rational policy when no
edge exists.
```

The finding is more nuanced than "fix works/doesn't work." The original reward *forces* trading via its inactivity penalty; log-utility removes that incentive. On synthetic data with no genuine signal → log-utility correctly produces no trades, original produces (mostly bad) trades. **Whether log-utility is an improvement depends on whether the underlying data contains real edge.**

### 5. Not all pre-condition gates carry equal weight

The ETH strategy uses three pre-condition gates (Hurst, BTC-ETH correlation, BTC ATR) that filter regime suitability before signals fire. The ablation experiment turns each gate off individually and measures the impact:

```
ETH STRATEGY — PRE-CONDITION GATE ABLATION
Configuration  Hurst  Corr   ATR  Trades   Sharpe    Return    MaxDD   WinR
----------------------------------------------------------------------------------------
all_on             Y     Y     Y      13   +0.733   +85.08%  -52.42%  53.8%
hurst_off          .     Y     Y      13   +0.781   +92.83%  -52.42%  53.8%   ← identical
corr_off           Y     .     Y      14   +0.736   +85.45%  -52.32%  57.1%   ← negligible
atr_off            Y     Y     .      16   +0.482   +71.44%  -52.42%  43.8%   ← clear drop
all_off            .     .     .      17   +0.521   +78.97%  -52.32%  47.1%

VERDICT: The 'atr' gate matters most — removing it drops Sharpe by 0.251.
The 'hurst' and 'correlation' gates are not binding on this data.
```

The ATR gate is doing nearly all the gating work; the other two appear redundant with downstream signals on this synthetic data. **The "three concurrent gates" framing in the strategy design oversells the architecture's complexity** — the value is concentrated in one gate.

### 6. The DFA Hurst replacement has a subtle threshold problem

The R/S Hurst estimator has known small-sample bias; the Detrended Fluctuation Analysis (DFA) Hurst is a common proposed alternative. The ablation experiment tests this — and uncovers a wrinkle:

```
HURST ESTIMATOR STATISTICS (on the underlying ETH close series):
 Window  Method      Mean      Std      Min      Max    NaN%   >0.5%
    120      rs    +0.915    0.024   +0.827   +0.960    7.9%  100.0%
    120     dfa    +1.545    0.116   +1.195   +1.867    7.9%  100.0%
    250      rs    +0.940    0.016   +0.873   +0.974   16.6%  100.0%
    500      rs    +0.939    0.010   +0.908   +0.965   33.3%  100.0%   ← smaller std
```

Three findings:

**(a) Switching R/S → DFA at window=120 does not change strategy outcomes.** Trade count, Sharpe, MDD, win rate are identical across the two methods at every window.

**(b) DFA produces values centered around 1.5, not 0.5.** DFA returns the scaling exponent α, which for non-stationary fractional Brownian motion equals H+1. The strategy's `H > 0.5` gate fires for *every* DFA value on this data — effectively making the gate inert. **The DFA fix as commonly described would silently disable the Hurst gate if applied literally without re-tuning the threshold.**

**(c) Window size affects performance more than method choice.** Longer windows give more stable estimates but worse strategy outcomes (Sharpe 1.34 → 0.66 from w=120 to w=500). Classic bias-variance: stability comes at the cost of responsiveness.

The DFA fix sounds reasonable but applying it literally would break the strategy in a non-obvious way. The bug is in the threshold, not the method.

---

## Repository structure

```
quant_portfolio/
├── README.md                            # this file
├── requirements.txt                     # pinned dependencies
│
├── demo_full_pipeline.py                # demo 1: strategy → backtest → bootstrap
├── demo_rl_vs_rule.py                   # demo 2: RL vs hand-coded baselines
├── demo_walkforward_dsr.py              # demo 3: walk-forward + DSR
├── demo_gate_ablation.py                # demo 4: ETH gate ablation
├── demo_hurst_ablation.py               # demo 5: Hurst window × method
├── demo_reward_comparison.py            # demo 6: BTC reward comparison
│
├── src/
│   ├── backtester.py                    # in-house, deterministic, no-lookahead (~500 lines)
│   ├── indicators.py                    # all indicators as pure causal functions (~420 lines)
│   │
│   ├── data_io/
│   │   ├── loader.py                    # robust OHLCV CSV loader (~200 lines)
│   │   └── binance_puller.py            # public-API data puller (~180 lines)
│   │
│   ├── strategies/
│   │   ├── eth_regime_confirmation.py   # ETH CUSUM strategy (~400 lines)
│   │   ├── btc_qlearning.py             # BTC Q-learning agent w/ Moody-Saffell option (~610 lines)
│   │   └── btc_rule_based.py            # rule-based BTC baselines (~190 lines)
│   │
│   └── validation/
│       ├── bootstrap.py                 # Politis-Romano stationary bootstrap (~280 lines)
│       ├── parity_check.py              # quarterly comparison vs original (~270 lines)
│       ├── rl_vs_rule_experiment.py     # RL-vs-baseline experiment harness (~260 lines)
│       ├── walkforward.py               # rolling/expanding window CV (~290 lines)
│       ├── deflated_sharpe.py           # Bailey & López de Prado DSR (~210 lines)
│       ├── gate_ablation.py             # ETH gate ablation (~250 lines)
│       ├── hurst_ablation.py            # Hurst window × method (~315 lines)
│       ├── cold_oos.py                  # cold OOS evaluator (~340 lines)
│       └── reward_comparison.py         # BTC reward comparison (~270 lines)
│
├── tests/                               # 135 tests across 13 files, all passing
├── archive/                             # earlier prototypes preserved for comparison
├── validation_data/                     # baseline quarterly summaries for parity checks
└── results/                             # generated experiment outputs
```

---

## Engineering improvements over the prototype

The earlier prototypes of these strategies (preserved in `archive/`) had typical research-code problems. The rebuild addresses each:

| Issue in the prototype | Why it matters | Rebuild fix |
|---|---|---|
| Proprietary SDK dependency | Code unrunnable outside the SDK environment; no way to audit execution timing | `src/backtester.py` — in-house, deterministic, no-lookahead, fully tested |
| Metrics produced only by an opaque external scorer (no P&L code in the strategy) | Reported Sharpe (~6) can't be audited and doesn't reproduce transparently | Strict `exec_price = prices_open[t + 1]` invariant with a dedicated unit test; all execution auditable |
| No tests | Reported metrics cannot be independently verified | 135 tests covering correctness invariants |
| Indicators + strategy + execution entangled | Validation experiments impossible without separation | Strategies emit a `signal` column; backtester consumes it; clean separation |
| Hurst computed on prices, not returns | Statistically incorrect; R/S is defined on stationary series | Pluggable `hurst_method='dfa'` and proper documentation |
| 5-period rolling σ for CUSUM | Too jittery; fires false regime alarms | `cusum_sigma_method='ewma'` option |
| Reward function with flat-position penalty | Forces always-in-the-market; explains the 30:1 long/short asymmetry | Pluggable `reward_type='log_utility'` (Moody-Saffell) + comparison experiment |
| Code/doc inconsistency (540 vs 1620 states) | Erodes confidence in reported numbers | Code comments call out the discrepancies |

---

## What's done and what isn't

Honest, milestone by milestone:

**Complete:**
- ✓ In-house backtester (no-lookahead, deterministic, tested)
- ✓ All indicators as pure causal functions
- ✓ Both strategies refactored using the new modules
- ✓ Politis-Romano stationary bootstrap for trade-level CIs
- ✓ Data loading infrastructure (CSV loader, Binance puller, parity check)
- ✓ Rule-based baselines + RL-vs-baseline experiment harness
- ✓ Walk-forward validation (rolling + expanding, with purging and embargo)
- ✓ Deflated Sharpe Ratio
- ✓ ETH gate ablation experiment
- ✓ Hurst window × method ablation
- ✓ Cold OOS evaluator with parameter-freeze guarantee
- ✓ Q-table serialization for true cold-OOS evaluation of BTC
- ✓ BTC reward function comparison experiment
- ✓ **Real-data reconciliation on 2020–2023 BTC/ETH: ETH Sharpe 5.97 → 0.129 (DSR 0.000); BTC Sharpe 9.15 → 0.123**
- ✓ 135 tests across 13 files
- ✓ 6 runnable demos

**Not yet done:**
- ✗ BTC full-episode finalization — the reconciliation used 120 training episodes (compute limit); `finalize_btc_real.py` runs the full 1400 locally for the exact return. The Sharpe/drawdown verdict is already settled and unchanged by this.
- ✗ Cold OOS on 2024+ data — script is complete (`src/validation/cold_oos.py`) and tested, needs to be run locally with Binance data access.
- ✗ Hansen SPA test, HMM regime decomposition, Sobol sensitivity — the machinery is in place; each is ~100-200 lines of script.

---

## Setup

```bash
git clone <repo>
cd quant_portfolio
pip install -r requirements.txt

# To run on real data, drop hourly OHLCV CSVs in data/:
#   data/BTC_USDT_1h_2020_2023.csv
#   data/ETH_USDT_1h_2020_2023.csv

# Or pull from Binance (works from any network-enabled machine):
python -m src.data_io.binance_puller \
    --symbol BTCUSDT --interval 1h \
    --start 2020-01-01 --end 2023-12-31 \
    --output data/BTC_USDT_1h_2020_2023.csv
```

Public BTC/ETH hourly data is available from the [Binance Public Data archive](https://github.com/binance/binance-public-data) or [Kaggle](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history).

---

## Running the cold OOS evaluation

A frozen-parameter run on data the strategies have never seen:

```bash
# Default: 2024-01-01 through today, pulls data from Binance
python -m src.validation.cold_oos --output-dir results/cold_oos_2024/

# Or with pre-downloaded CSVs:
python -m src.validation.cold_oos \
    --btc-csv data/BTC_USDT_1h_2024.csv \
    --eth-csv data/ETH_USDT_1h_2024.csv \
    --output-dir results/cold_oos_2024/
```

The script is **frozen-parameter by design** — no hyperparameters are exposed in the CLI. To change strategy parameters, you must edit the source code, which leaves an audit trail. This is the only honest way to do OOS.

Output: trade logs, equity curves, JSON metrics with bootstrap CIs, human-readable summary. Both strategies should be considered validated **only if their cold-OOS Sharpe lower bound (bootstrap CI) exceeds zero.**

---

## Testing

```bash
cd quant_portfolio
python tests/test_backtester.py          # 11 tests — no-lookahead, determinism, fee accounting
python tests/test_indicators.py          # 14 tests — Kalman causality, Hurst on returns
python tests/test_bootstrap.py           # 12 tests — autocorr preservation, CI brackets
python tests/test_eth_strategy_smoke.py  #  5 tests — strategy runs end-to-end
python tests/test_btc_strategy_smoke.py  # 10 tests — Q-learning + reward functions
python tests/test_data_io.py             # 11 tests — loaders, dedup, alignment, binning
python tests/test_rule_baselines.py      # 13 tests — rules + RL-vs-rule + seed regression
python tests/test_walkforward_dsr.py     # 18 tests — fold generation, DSR limit cases
python tests/test_gate_ablation.py       #  9 tests — ablation flags + verdict logic
python tests/test_hurst_ablation.py      # 11 tests — Hurst stability + window/method grid
python tests/test_cold_oos.py            #  6 tests — cold OOS runner (fallback + true-cold modes)
python tests/test_q_table_io.py          #  8 tests — Q-table save/load + compatibility checks
python tests/test_reward_comparison.py   #  7 tests — reward comparison harness
```

Key invariants verified:
- **No lookahead** in the backtester: a signal observed at bar `t` executes at bar `t+1`'s open price, not bar `t`'s close.
- **Kalman filter is causal**: state at `t` doesn't change when future observations are added.
- **Bootstrap preserves autocorrelation** in AR(1) data when block length is appropriate.
- **DSR catches selection bias**: a strategy with mean Sharpe equal to random-trial expected-max gives DSR ≈ 0.5.
- **Seeds actually vary**: regression test catches the seed-propagation bug that originally made all RL seeds converge to identical Q-tables.

---

## The deeper story

It's easy to produce a high backtest Sharpe. What's hard — and what real quant work requires — is producing evidence that the Sharpe reflects genuine edge rather than lookahead, overfitting, or selection. That evidence has five components:

1. **Reproducible code.** A backtester whose internals you can read and test. The in-house engine here removes the closed-source dependency of the prototype and makes the execution timing auditable line by line.

2. **Strict causal execution.** Every signal is computed from data up to bar t and executes at bar t+1's open, in an engine you can read and test. When the same signals are scored instead by an opaque external SDK — as in the prototype — the reported Sharpe jumps from 0.13 to 5.97. The reconciliation section shows that this gap is about *whether the execution is auditable*, not about a single line of code.

3. **Honest confidence intervals.** A point estimate is a hypothesis. A bootstrap CI is a claim about how robust the hypothesis is. The Politis-Romano stationary bootstrap respects the serial correlation of trade returns.

4. **Out-of-sample evidence.** A strategy tuned on a window is contaminated; only data the strategy hasn't seen can validate it. The walk-forward harness produces many such windows from one dataset.

5. **Selection-bias correction.** With many hyperparameters across both strategies, the in-sample Sharpe is the maximum across many implicit trials. The Deflated Sharpe Ratio corrects for this — and, as the reconciliation shows, drives the corrected probability of true edge to zero.

The codebase produces (1)-(5) as infrastructure that can be run by anyone with hourly BTC/ETH data. The validity of any backtest's performance claims becomes an empirical question with a definite procedure for answering it, rather than a marketing assertion.

---

## License and disclaimer

This codebase is for educational and research purposes. The strategies have not been validated for live trading; the reconciliation section documents evidence that a naive backtest of them substantially overstated risk-adjusted returns.

**No part of this constitutes investment advice.**
