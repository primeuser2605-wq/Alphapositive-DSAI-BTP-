# Archive

These files are the original Inter IIT Tech Meet 13.0 submissions, preserved for reproducibility and historical reference. They are **not used by the main codebase** — see `../src/` for the refactored, tested versions.

## Files

### Code (Python scripts)
- **`eth_strategy_original.py`** (was `main_1_eth.py`) — The canonical ETH/USDT regime-confirmation strategy using CUSUM, Kalman filter, Hurst exponent, Bollinger Bands, Supertrend, and RSI. This is the version whose results (Sharpe 5.97, 162 trades, 17.1% MDD) are reported in the project documentation.

- **`eth_macd_variant_archived.py`** (was `main_eth.py`) — An earlier ETH variant using MACD-based signals instead of CUSUM. Not the production version; archived for historical reference.

- **`btc_strategy_original.py`** (was `main_btc.py`) — The BTC/USDT tabular Q-learning agent whose 2023 results (Sharpe 9.15, 31 trades, 13.5% MDD) are reported in the project documentation.

- **`btc_experimental_archived.py`** (was `main_btc__1_.py`) — A BTC variant with experimental features (consolidation signals, fractional Brownian motion imports). Not the production version; archived.

### Notebooks
- **`eth_strategy.ipynb`** — The original Jupyter notebook used for ETH strategy development. Useful for seeing the exploratory analysis that led to the final design.
- **`exec_eth_notebook.ipynb`** (was `exec_1_eth.ipynb`) — Execution notebook showing the original strategy run with Untrade SDK calls.

### Documentation
- **`Zelta_Final_Report.pdf`** — The complete competition report submitted to Zelta Automations for Inter IIT Tech Meet 13.0. Contains the full methodology, results, and analysis as originally presented (the Gold-winning submission).
- **`original_README.md`** (was `README__3_.md`) — The README that accompanied the original competition submission.

## Dependencies

These scripts require:

- `untrade.client` (Zelta-internal SDK) — **not publicly available**, which is the primary reason for the rebuild
- `hurst`, `pykalman`, `pandas_ta`, `fbm` — various PyPI packages

The new code in `../src/` reimplements the indicator logic from scratch as pure Python (numpy + pandas), eliminating these dependencies and making the work reviewable by anyone.

## Why kept

These are kept to:

1. Allow comparison: anyone with Untrade access can rerun the originals and verify the refactor reproduces results within tolerance.
2. Preserve provenance: the reported Sharpe numbers came from these files, not the refactor.
3. Documentation: the gap between this code and `../src/` illustrates the engineering work involved in productionizing research code.
