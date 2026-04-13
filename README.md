# Alphapositive-DSAI-BTP-
# Cryptocurrency Trading Strategies Using Reinforcement Learning and Correlation Analysis

This project explores and implements algorithmic trading strategies for **BTC/USDT** and **ETH/USDT** by leveraging **Reinforcement Learning (Q-Learning)** and **correlation-based statistical techniques**, with a strong focus on **market dynamics**, **volatility clustering**, and **risk management**.

## 🚀 Overview

- **BTC Strategy**: Built using Reinforcement Learning with Q-Learning for sequential decision-making in highly volatile markets.
- **ETH Strategy**: Designed around statistical correlation with BTC using signal processing techniques like **CUSUM**, **Hurst Exponent**, and **ATR-based volatility mapping**.
- **Backtested** on historical data and benchmarked against market returns to measure real-world performance.

---
## 📊 Key Results

| Pair      | Net Return | Benchmark | Outperformance | Sharpe Ratio | Win Rate |
|-----------|------------|-----------|----------------|--------------|----------|
| BTC/USDT  | 224.9%     | 157.08%   | +42.68%        | 9.15         | 68%      |
| ETH/USDT  | 7684.52%   | 1687.59%  | +355.48%       | 5.97         | 48%      |

---

## 🧠 Methodology

### BTC – Reinforcement Learning Approach
- **Q-Learning Algorithm** with epsilon-greedy exploration
- **State space**: Price % change, RSI, Aroon, EMAs, and position
- **Actions**: Enter Long/Short, Exit Long/Short, Hold
- **Reward Function**: Profit/loss adjusted for commission and risk

### ETH – Correlation-Based Statistical Approach
- **Correlation Analysis** between BTC and ETH log returns
- **Signal Generation** using:
  - **CUSUM** for regime shift detection
  - **ATR-based volatility thresholds**
  - **Hurst Exponent** for trend filtering
  - **Kalman & Gaussian Filters** for smoothing

---

## 📈 Features
- Volatility clustering via PCA, GARCH, and Markov Switching
- Entry/exit signal generation from combined indicators
- Custom risk management: trailing stop-loss, cooldowns, position sizing
- Extensive backtesting framework with quarterly metrics tracking

---

## 🛠️ Tech Stack
- **Python**, **NumPy**, **Pandas**, **Matplotlib**, **Scikit-learn**
- **Statsmodels**, **TA-Lib**, **Gym/Custom RL Environment**
- Jupyter notebooks for experimentation and visualization

---

## 📂 Repository Structure

```bash
├── data/                   # Historical BTC & ETH price data             
├── notebooks/              # EDA, strategy development, testing
├── src/                    # Core implementation (RL agent, signals, filters)
├── results/                # Backtesting outputs and visualizations
├── requirements.txt
└── README.md
```

---

## 📌 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crypto-rl-correlation-strategy.git
   cd crypto-rl-correlation-strategy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run simulations:
   - Use Jupyter notebooks in the `notebooks/` folder for ETH correlation and BTC RL experiments.

---

## 📃 License

MIT License. See `LICENSE` for details.

---
