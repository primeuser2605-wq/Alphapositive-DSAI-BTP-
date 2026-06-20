> 🏆 **Gold Medal Winner!**
>
> This project was awarded the **Gold Medal** in the Quant Track of the **Inter IIT Tech Meet 13.0** held at **IIT Bombay**.

# Curating Alphas for the ETH/USDT and the BTC/USDT Markets

This repository contains the implementation of two distinct algorithmic trading strategies developed for the BTC/USDT and ETH/USDT cryptocurrency pairs. This project was created by **IIT Guwahati** for the **Inter IIT Tech Meet 13.0**, with the problem statement provided by **Zelta Automations and Untrade**.

The strategies are backtested on historical hourly data from **January 1, 2020, to December 31, 2023**.

## Table of Contents
- [Project Overview](#project-overview)
- [Strategy 1: Lead-Lag Statistical Strategy for ETH/USDT](#strategy-1-lead-lag-statistical-strategy-for-ethusdt)
  - [Hypothesis](#hypothesis)
  - [Methodology & Key Indicators](#methodology--key-indicators)
  - [Trade Logic Flow](#trade-logic-flow)
  - [Risk Management](#risk-management-1)
- [Strategy 2: Reinforcement Learning Strategy for BTC/USDT](#strategy-2-reinforcement-learning-strategy-for-btcusdt)
  - [Approach](#approach)
  - [State Space](#state-space)
  - [Action Space](#action-space)
  - [Reward Structure](#reward-structure)
  - [Risk Management](#risk-management-2)
- [Performance Summary](#performance-summary)
  - [ETH/USDT Strategy Performance (2020-2023)](#ethusdt-strategy-performance-2020-2023)
  - [BTC/USDT Strategy Performance (Annualized 2023)](#btcusdt-strategy-performance-annualized-2023)
- [Technology Stack](#technology-stack)
- [Installation & Usage](#installation--usage)
- [Project Context](#project-context)


## Project Overview

The goal of this project is to develop and backtest robust, automated trading strategies that can effectively manage risk and generate consistent returns in the highly volatile cryptocurrency markets. Two separate strategies were designed:

1.  A **statistical lead-lag strategy** for the `ETH/USDT` market, which leverages the strong correlation and predictive influence of Bitcoin's price movements on Ethereum.
2.  A **Reinforcement Learning (Q-learning) strategy** for the `BTC/USDT` market, where an agent learns optimal trading policies through interaction with the market environment.

## Strategy 1: Lead-Lag Statistical Strategy for ETH/USDT

This strategy operates on the principle that Bitcoin acts as a leading indicator for the broader crypto market, including Ethereum. By analyzing BTC's trend and momentum, we can make more informed trading decisions for ETH.

### Hypothesis

Bitcoin's hourly price movements have a significant positive correlation and a lead-lag relationship with Ethereum's price movements. Statistical analysis confirms this with a **Pearson correlation coefficient of 0.7851** (p-value < 0.05), rejecting the null hypothesis of a weak correlation.

### Methodology & Key Indicators

The strategy combines several indicators on both BTC (for trend confirmation) and ETH (for entry/exit timing).

**Key Components:**
*   **CUSUM (Cumulative Sum) Filter**: Applied to BTC price data to detect statistically significant shifts in market regimes (bullish/bearish). A **Kalman Filter** is used to generate a smooth reference price, making the CUSUM detection more robust.
*   **Hurst Exponent**: Used as a market filter to ensure trades are only taken in **trending markets (H > 0.5)**, avoiding choppy, mean-reverting periods.
*   **Rolling Correlation**: A 7-hour rolling correlation between BTC and ETH ensures trades are only executed when the assets are moving in tandem (correlation > 0.6).
*   **ATR (Average True Range)**: A dynamic volatility filter on BTC to ensure trading only occurs in low-volatility conditions (ATR < 1% of opening price), improving trend reliability.
*   **Confirmation Indicators**:
    *   **On BTC**: RSI and Bollinger Bands are used to confirm the strength of the bullish or bearish regime identified by CUSUM.
    *   **On ETH**: The **Supertrend** indicator is used to confirm the local trend direction for precise entry and exit timing.

### Trade Logic Flow

1.  **Pre-Trade Checks (Filters)**:
    *   Is the Hurst Exponent > 0.5?
    *   Is the BTC-ETH 7-hour correlation > 0.6?
    *   Is BTC's ATR < 1% of its open price?
    *   If all are true, proceed. Otherwise, do not trade.

2.  **Long Entry (Buy ETH)**:
    *   BTC regime is identified as **bullish** by CUSUM.
    *   BTC RSI > 70.
    *   BTC price is above the middle Bollinger Band.
    *   ETH Supertrend indicates a bullish phase.

3.  **Short Entry (Sell ETH)**:
    *   BTC regime is identified as **bearish** by CUSUM.
    *   BTC RSI < 30.
    *   BTC price is below the lower Bollinger Band.
    *   ETH Supertrend indicates a bearish phase.

4.  **Exit Conditions**: Positions are closed if market conditions reverse, signaled by a combination of BTC regime change, RSI crossovers, or a flip in the ETH Supertrend indicator.

### Risk Management

*   **Trailing Stop-Loss**: A dynamic stop-loss trails the peak price (for longs) or trough price (for shorts) to lock in profits and limit losses.
*   **Volatility-Based Exit**: All positions are immediately closed if BTC's ATR exceeds 2.5% of its opening price, indicating extreme volatility.
*   **Time-Based Exit**: A maximum holding period of **28 days** per trade prevents indefinite exposure.
*   **Cooldown Period**: A **1-day cooldown** is enforced after a stop-loss is triggered to avoid re-entering a turbulent market immediately.

## Strategy 2: Reinforcement Learning Strategy for BTC/USDT

This strategy employs a **Q-learning** agent to navigate the BTC/USDT market. The agent learns an optimal policy by maximizing a cumulative reward signal, balancing exploration of new actions with exploitation of known, profitable strategies.

### Approach
The model is trained on 3 years of data (2020-2022) and then tested and retrained quarterly throughout 2023. This continuous learning approach allows the agent to adapt to evolving market dynamics.

### State Space
The agent observes the market through a carefully crafted state space, which includes:
*   **Price Change**: Binned into 20 discrete categories from -5% to +5%.
*   **Current Position**: (1: Long, -1: Short, 0: No Position).
*   **RSI Signal**: (1: >75, -1: <35, 0: Otherwise).
*   **EMA Crossover**: (1: Bullish EMA(7)>EMA(14)>EMA(28), -1: Bearish trend, 0: Otherwise).
*   **Aroon Indicator Signal**: (1: Bullish, -1: Bearish).

### Action Space
The agent can choose from four distinct actions at each step:
1.  **Enter Long**: Open a new long position or reverse a short position.
2.  **Exit Long**: Close an existing long position.
3.  **Enter Short**: Open a new short position or reverse a long position.
4.  **Exit Short**: Close an existing short position.

### Reward Structure
The reward function is designed to guide the agent towards profitability while accounting for real-world costs:
*   **Profits/Losses**: The primary reward is the realized or unrealized profit/loss.
*   **Penalties**:
    *   A commission fee is subtracted for every trade execution.
    *   A small penalty for inactivity (holding no position) encourages market participation.
    *   Large negative rewards are given for actions leading toward bankruptcy.

### Risk Management
*   **Stop-Loss**: A hard stop-loss is coded into the environment, automatically closing any position that incurs a **5% loss**.
*   **Position Sizing**: To manage leverage and risk, short positions are limited to **75% of the available balance**.

## Performance Summary

### ETH/USDT Strategy Performance (2020-2023)
| Metric | Value |
| :--- | :--- |
| **Sharpe Ratio** | 5.966 |
| **Sortino Ratio** | 19.67 |
| **Final Balance (from $1,000)** | $77,845.15 |
| **Profit Percentage** | 7684.52% |
| **Win Rate** | 47.53% |
| **Maximum Drawdown** | 17.14% |
| **Total Trades** | 162 |

### BTC/USDT Strategy Performance (Annualized 2023)
| Metric | Value |
| :--- | :--- |
| **Sharpe Ratio** | 9.15 |
| **Sortino Ratio** | 26.95 |
| **Final Balance (from $1,000)** | $3,249.01 |
| **Profit Percentage** | 224.90% |
| **Win Rate** | 64.52% |
| **Maximum Drawdown** | 13.50% |
| **Total Trades** | 31 |

## Technology Stack
*   **Language**: Python 3.x
*   **Libraries**:
    *   `pandas` for data manipulation
    *   `numpy` for numerical operations
    *   `matplotlib` / `seaborn` for plotting
    *   `statsmodels` for statistical modeling
    *   A Reinforcement Learning library (e.g., `stable-baselines3`, `gym`) or a custom Q-learning implementation.

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/algorithmic-trading-btc-eth.git
    cd algorithmic-trading-btc-eth
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Place your data:**
    Ensure your historical BTC/USDT and ETH/USDT hourly data files (in `.csv` format) are placed in a `data/` directory.

4.  **Run the backtests:**
    ```bash
    # To run the backtest for the ETH/USDT strategy
    python backtest_eth_strategy.py

    # To run the backtest for the BTC/USDT RL strategy
    python backtest_btc_strategy.py
    ```
    *(Note: Script names are illustrative. Please refer to the source code for actual filenames.)*

## Project Context
This work was submitted for the **Inter IIT Tech Meet 13.0**.
-   **Problem Setters**: Zelta Automations & untrade
-   **Authors**: Team 67
-   **Achievement**: 🏆 **Gold Medal**
