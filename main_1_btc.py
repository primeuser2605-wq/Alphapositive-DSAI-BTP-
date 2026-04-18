# Import libraries
import os
import uuid
from untrade.client import Client

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore")

# Set a seed value for reproducibility of results
np.random.seed(1)

def process_data(data_incoming):
    """
    Processes market data by filtering, handling missing values, and generating technical indicators/signals.

    This function uses the `MarketDataProcessor` class to:
    - Filter data within a specified date range.
    - Compute technical indicators: RSI, EMA (7, 14, 28), Aroon (Up, Down), and signals based on these indicators.
    - Clean data to ensure consistency for analysis.

    Parameters:
        data_incoming (pd.DataFrame): Input market data with required columns: 
                                      'datetime', 'close', 'high', 'low', 'volume'.

    Returns:
        pd.DataFrame: Processed data containing essential columns, indicators, and trading signals.
    """
    def load_data(df, start_datetime, end_datetime, drop_columns=None):
        data = df.copy()

        if drop_columns:
            existing_cols = [col for col in drop_columns if col in data.columns]
            data.drop(existing_cols, axis=1, inplace=True)
        
        data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S') 

        data.set_index('datetime', inplace=True)
        
        data = data.loc[start_datetime:end_datetime].copy()
        
        essential_columns = ['close', 'high', 'low', 'volume']
        for col in essential_columns:
            if col not in data.columns:
                raise ValueError(f"Missing essential column: '{col}' in the data.")
        
        data[essential_columns] = data[essential_columns].fillna(method='ffill')
        
        data['MA7'] = data['close'].rolling(window=7).mean()
        data['MA14'] = data['close'].rolling(window=14).mean()
        data['MA_Signal'] = 0
        data.loc[data['MA7'] > data['MA14'], 'MA_Signal'] = 1
        data.loc[data['MA7'] < data['MA14'], 'MA_Signal'] = -1

        data['Aroon_Up'] = 100 * (14 - data['high'].rolling(window=15).apply(lambda x: x.argmax())) / 14
        data['Aroon_Down'] = 100 * (14 - data['low'].rolling(window=15).apply(lambda x: x.argmin())) / 14
        data['Aroon_Signal'] = 0
        data.loc[data['Aroon_Up'] > data['Aroon_Down'], 'Aroon_Signal'] = 1
        data.loc[data['Aroon_Up'] < data['Aroon_Down'], 'Aroon_Signal'] = -1

        def rsi(data, window):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        data['RSI_14'] = rsi(data['close'], 14)
        data['RSI_Signal'] = 0
        data.loc[data['RSI_14'] > 75, 'RSI_Signal'] = 1
        data.loc[data['RSI_14'] < 35, 'RSI_Signal'] = -1

        data['returns'] = np.log(data.close / data.close.shift(1))

        data['EMA7'] = data['close'].ewm(span=7, adjust=False).mean()
        data['EMA14'] = data['close'].ewm(span=14, adjust=False).mean()
        data['EMA28'] = data['close'].ewm(span=28, adjust=False).mean()
        data['EMA_Signal'] = 0
        data.loc[(data['EMA7'] > data['EMA14']) & (data['EMA14'] > data['EMA28']), 'EMA_Signal'] = 1
        data.loc[(data['EMA7'] < data['EMA14']) & (data['EMA14'] < data['EMA28']), 'EMA_Signal'] = -1
        
        data['EMA_Signal'] = data['EMA_Signal'].astype(int)
        
        data['pct_change'] = data['close'].pct_change(periods=1).fillna(0) * 100
        
        data.dropna(subset=['EMA7', 'EMA14', 'EMA28'], inplace=True)
        
        data.dropna(inplace=True)
        
        signal_columns = ['Aroon_Signal', 'RSI_Signal', 'EMA_Signal']
        data[signal_columns] = data[signal_columns].astype(int)
        
        return data


    train_start_datetime = '2020-01-01'
    train_end_datetime = '2022-12-31'
    test_start_datetime = '2023-01-01'
    test_end_datetime = '2024-01-01'

    train_btc_data = pd.read_csv("BTC_2019_2023_1h.csv")

    train_data = load_data(train_btc_data,train_start_datetime, train_end_datetime, drop_columns=['Unnamed: 0'])
    test_data = load_data(data_incoming,test_start_datetime, test_end_datetime, drop_columns=['Unnamed: 0'])

    final_data = pd.concat([train_data, test_data])
    return final_data

def strat(data_incoming):
    """
    Implements a trading strategy using a reinforcement learning-based Q-learning algorithm.
    
    Parameters:
        data_incoming (DataFrame): Input data containing columns `close`, `Aroon_Signal`, `RSI_Signal`, 
                                   `EMA_Signal`, and `pct_change` indexed by time.
    
    The function:
    1. Splits the input data into training and testing periods using the nested `callit` function.
    2. Defines a `TradingEnvironment` class to simulate trading actions based on price signals.
    3. Configures a Q-learning algorithm to optimize trading decisions:
       - Defines discrete state-action spaces (e.g., price bins, signal bins, holdings states).
       - Updates Q-values for each episode and decays exploration over time.
    
    Returns:
        None: Runs the strategy and logs trading behavior and performance for analysis.
    """
    train_start_datetime = '2020-01-01'
    train_end_datetime = '2022-12-31'
    test_start_datetime = '2023-01-01'
    test_end_datetime = '2024-01-01'
    train_data = data_incoming.loc[train_start_datetime:train_end_datetime].copy()
    test_data = data_incoming.loc[test_start_datetime:test_end_datetime].copy()
    
    train_prices = train_data['close'].values 
    train_aroon_signal = train_data['Aroon_Signal'].values
    train_rsi_signal = train_data['RSI_Signal'].values
    train_ema_signal = train_data['EMA_Signal'].values
    train_pct_change = train_data['pct_change'].values

    test_prices = test_data['close'].values 
    test_aroon_signal = test_data['Aroon_Signal'].values
    test_rsi_signal = test_data['RSI_Signal'].values
    test_ema_signal = test_data['EMA_Signal'].values
    test_pct_change = test_data['pct_change'].values

    normalized_train_aroon_signal = train_aroon_signal  
    normalized_train_rsi_signal = train_rsi_signal     
    normalized_train_ema = train_ema_signal
    normalized_train_pct = train_pct_change

    normalized_test_aroon_signal = test_aroon_signal
    normalized_test_rsi_signal = test_rsi_signal
    normalized_test_ema = test_ema_signal
    normalized_test_pct = test_pct_change

 
    MIN_TRADE_AMOUNT = 5000     
    COMMISSION_RATE = 0.0015   
    MAX_SHORT_POSITION = 0.75  
    STOP_LOSS_PERCENT = 0.05 

   
    class TradingEnvironment:
        def __init__(self, actual_prices, aroon_signal, rsi_signal, ema_signal, pct_change):
            self.actual_prices = actual_prices
            self.aroon_signal = aroon_signal
            self.rsi_signal = rsi_signal
            self.ema_signal = ema_signal
            self.pct_change = pct_change
            self.n_steps = len(actual_prices)
            self.current_step = 0
            self.position = 0         
            self.balance = 10000.0    
            self.net_worth = self.balance
            self.initial_balance = self.balance
            self.trades = []
            self.entry_price = 0.0
            self.holdings = 0.0        
            self.history = [self.net_worth]
            self.cooldown = 0        
            self.last_worth = self.balance

        def _get_observation(self):
            return np.array([
                self.aroon_signal[self.current_step],
                self.rsi_signal[self.current_step],
                self.ema_signal[self.current_step],
                self.position,
                self.pct_change[self.current_step]
            ])
        
        def reset(self):
            self.current_step = 0
            self.position = 0
            self.balance = self.initial_balance
            self.net_worth = self.balance
            self.trades = []
            self.entry_price = 0.0
            self.holdings = 0.0
            self.history = [self.net_worth]
            self.cooldown = 0
            self.last_worth = self.balance
            return self._get_observation()

        def step(self, action):
            actual_price = self.actual_prices[self.current_step]
            done = False
            reward = 0.0  
            traded = 0
            if action == 1 and (self.position==-1 or self.position == 0 ):

                if self.position ==-1:
                    traded = 1
                    gross_purchase = -self.holdings * actual_price
                    commission_exit = gross_purchase * COMMISSION_RATE
                    total_cost = gross_purchase + commission_exit
                    self.balance -= total_cost  
                    profit = (self.balance + self.holdings * actual_price) - self.last_worth
                    if self.balance < MIN_TRADE_AMOUNT:
                        self.balance = 0.0
                        self.holdings = 0.0
                        self.position = 0
                        reward += -100000000
                        done = True
                        return self._get_observation(), reward, done
                    else:
                        self.holdings = 0.0
                        self.position = 0 
                        reward += profit
                    self.net_worth = self.balance + self.holdings * actual_price
                    
                    
                if self.balance < MIN_TRADE_AMOUNT:
                    reward -= 100000000
                    done = True
                    return self._get_observation(), reward, done
                
                if(self.position == 0 ) :
                    self.last_worth = self.net_worth
                    total_cost = self.balance
                    commission = total_cost * COMMISSION_RATE 
                    investment_amount = total_cost - commission
                    self.holdings = investment_amount / actual_price
                    self.position = 1 
                    self.entry_price = actual_price
                    self.balance = 0.0  
                    if traded:
                        self.trades.append({
                                'step': self.current_step,
                                'trade_type': 'short_reversal',
                                'price': actual_price,
                                'commission': commission,
                                'signals' : 2
                            })
                    else :
                        self.trades.append({
                                'step': self.current_step,
                                'trade_type': 'long',
                                'price': actual_price,
                                'commission': commission,
                                'signals' : 1
                            })
                        
                    traded  = 2
                    reward -= commission 
                    pass
                

            elif action == 2 and self.position == 1:
                gross_sale = self.holdings * actual_price
                commission = gross_sale * COMMISSION_RATE
                net_sale = gross_sale - commission
                self.balance += net_sale 
                self.holdings = 0.0
                profit = (self.balance + self.holdings * actual_price) - self.last_worth
                self.position = 0 
                self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'long_close',
                        'price': actual_price,
                        'commission': commission,
                        'signals' : -1
                    })
                traded = 1
                reward += profit

            elif action == 3 and (self.position==1 or self.position ==0 ) :
                if(self.position ==1 ):
                    traded = 1
                    gross_sale = self.holdings * actual_price
                    commission = gross_sale * COMMISSION_RATE
                    net_sale = gross_sale - commission
                    self.balance += net_sale  
                    self.holdings = 0.0
                    profit = (self.balance + self.holdings * actual_price) - self.last_worth
                    self.position = 0 
                    reward += profit
                    self.net_worth = self.balance + self.holdings * actual_price
                    
                    
                if self.balance < MIN_TRADE_AMOUNT:
                    reward -= 100000000
                    done = True
                    return self._get_observation(), reward, done

                if(self.position ==0 and self.balance>MIN_TRADE_AMOUNT):
                    self.last_worth = self.net_worth
                    short_value = self.balance * MAX_SHORT_POSITION
                    gross_proceeds = short_value
                    commission_entry = gross_proceeds * COMMISSION_RATE
                    net_proceeds = gross_proceeds - commission_entry
                    units_to_short = gross_proceeds / actual_price 
                    self.holdings = -units_to_short 
                    self.position = -1 
                    self.entry_price = actual_price
                    self.balance += net_proceeds  
                    if(traded):
                        self.trades.append({
                                    'step': self.current_step,
                                    'trade_type': 'long_reversal',
                                    'price': actual_price,
                                    'commission': commission_entry,
                                    'signals': -2
                                })
                    else :
                        self.trades.append({
                                    'step': self.current_step,
                                    'trade_type': 'short',
                                    'price': actual_price,
                                    'commission': commission_entry,
                                    'signals': -1
                                })
                    traded = 2
                    reward -= commission_entry 
                    pass

            elif action == 4 and self.position == -1:
                traded = 1
                gross_purchase = -self.holdings * actual_price
                commission_exit = gross_purchase * COMMISSION_RATE
                total_cost = gross_purchase + commission_exit
                self.balance -= total_cost 
                profit = (self.balance + self.holdings * actual_price) - self.last_worth
                if self.balance < MIN_TRADE_AMOUNT:
                    self.balance = 0.0
                    self.holdings = 0.0
                    self.position = 0
                    self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'short_close',
                            'price': actual_price,
                            'commission': commission_exit,
                            'signals':1
                        })
                    reward += -100000000
                    done = True
                    return self._get_observation(), reward, done
                else:
                    self.holdings = 0.0
                    self.position = 0 
                    self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'short_close',
                            'price': actual_price,
                            'commission': commission_exit,
                            'signals':1
                        })
                    reward += profit

            if self.position == 1 and actual_price <= self.entry_price * (1 - STOP_LOSS_PERCENT) :
                    traded =1
                    gross_sale = self.holdings * actual_price
                    commission = gross_sale * COMMISSION_RATE
                    net_sale = gross_sale - commission
                    self.balance += net_sale  
                    self.holdings = 0.0
                    profit = (self.balance + self.holdings * actual_price) - self.last_worth
                    self.position = 0 
                    self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'long_close',
                            'price': actual_price,
                            'commission': commission,
                            'signals':-1
                        })
                    reward += profit

            elif self.position == -1 and actual_price >= self.entry_price * (1 + STOP_LOSS_PERCENT):
                    
                    traded = 1
                    gross_purchase = -self.holdings * actual_price
                    commission_exit = gross_purchase * COMMISSION_RATE
                    total_cost = gross_purchase + commission_exit
                    self.balance -= total_cost 
                    profit = (self.balance + self.holdings * actual_price) - self.last_worth
                    if self.balance < MIN_TRADE_AMOUNT:
                        self.balance = 0.0
                        self.holdings = 0.0
                        self.position = 0
                        self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'short_close',
                            'price': actual_price,
                            'commission': commission_exit,
                            'signals': 1
                        })
                        reward += -100000000
                        done = True
                        return self._get_observation(), reward, done
                    else:
                        self.holdings = 0.0
                        self.position = 0 
                        self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'short_close',
                            'price': actual_price,
                            'commission': commission_exit,
                            'signals':1
                        })
                        reward += profit
        
            else:
                
                if(traded == 0):
                    self.trades.append({
                        'step': self.current_step,
                        'trade_type': ' ',
                        'price': actual_price,
                        'commission': 0,
                        'signals':0
                        })

                if self.position != 0:
                    reward += (self.balance + self.holdings * actual_price) - self.net_worth
                else:
                    if(self.current_step > 0):
                        reward -= (abs(actual_price - self.actual_prices[self.current_step-1])) * self.net_worth / self.actual_prices[self.current_step-1]

            self.current_step += 1

            if self.current_step >= self.n_steps - 1:
                done = True
                if self.position ==1 :
                    self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'long_close',
                        'price': actual_price,
                        'commission': 0,
                        'signals':-1*self.position
                        }) 
                elif self.position == -1 :
                    self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'short_close',
                        'price': actual_price,
                        'commission': 0,
                        'signals':-1*self.position
                        }) 
                else :
                    self.trades.append({
                        'step': self.current_step,
                        'trade_type': ' ',
                        'price': actual_price,
                        'commission': 0,
                        'signals':0
                        }) 

            self.net_worth = self.balance + self.holdings * actual_price
            self.history.append(self.net_worth)
            self.max_net_worth = max(getattr(self, 'max_net_worth', self.initial_balance), self.net_worth)

            obs = self._get_observation()
            return obs, reward, done

    # Define parameters
    action_size = 5 
    alpha = 0.05         
    gamma = 0.95         
    epsilon = 1.0        
    epsilon_decay = 0.995
    epsilon_min = 0.1
    ep = 1030    
    n_pct_bins = 20  
    n_signal_bins = 3   
    n_holdings_states = 3 
    state_size = (
        n_signal_bins *  # Aroon Signal
        n_signal_bins *  # RSI Signal
        n_signal_bins *  # EMA Signal
        n_holdings_states *  # Holdings State
        n_pct_bins  # Percent Change
    )

    q_table = np.zeros((state_size, action_size))

    def get_price_bin(pct):
        bin_edges = np.linspace(-5, 5, n_pct_bins + 1)
        pct_clipped = np.clip(pct, -5, 5)
        pct_bin = np.digitize(pct_clipped, bins=bin_edges, right=False) - 1
        return int(np.clip(pct_bin, 0, n_pct_bins - 1))

    def get_signal_bin(signal):
        signal_mapping = {-1: 0, 0: 1, 1: 2}
        return signal_mapping.get(int(signal), 1)  


    def get_holdings_state(holdings):
        holdings_mapping = {-1: 0, 0: 1, 1: 2}
        return holdings_mapping.get(int(holdings), 1)  

    def get_state_index(aroon_signal, rsi_signal, ema_signal, holdings, pct_change_signal):
        aroon_bin = get_signal_bin(aroon_signal)
        rsi_bin = get_signal_bin(rsi_signal)
        ema_bin = get_signal_bin(ema_signal)
        holdings_bin = get_holdings_state(holdings)
        pct_bin = get_price_bin(pct_change_signal)

        state_index = (
            aroon_bin * (n_signal_bins ** 2 * n_pct_bins * n_holdings_states) + 
            rsi_bin * (n_signal_bins ** 1 * n_pct_bins * n_holdings_states) +  
            ema_bin * (n_pct_bins * n_holdings_states) +
            holdings_bin * (n_pct_bins) + 
            pct_bin
        )
        state_index = int(np.clip(state_index, 0, state_size - 1))
        return state_index
    
    train_env = TradingEnvironment(    
        actual_prices= train_prices,
        aroon_signal=normalized_train_aroon_signal,
        rsi_signal=normalized_train_rsi_signal,
        ema_signal=normalized_train_ema,
        pct_change= normalized_train_pct,
    )


    
    print("Starting Training...\n")

    for episode in range(1, ep + 1):
        state = train_env.reset()
        total_reward = 0
        step = 0

        while True:
            aroon_signal, rsi_signal, ema_signal, holdings, pct_change_signal = state
            state_index = get_state_index(aroon_signal, rsi_signal, ema_signal, holdings, pct_change_signal)

            if np.random.rand() < epsilon:
                action = np.random.choice(action_size)
            else:
                action = np.argmax(q_table[state_index]) 

            next_state, reward, done = train_env.step(action)
            total_reward += reward

            next_aroon_signal, next_rsi_signal, next_ema_signal, next_holdings, next_pct_change_signal = next_state
            next_state_index = get_state_index(next_aroon_signal, next_rsi_signal, next_ema_signal, next_holdings, next_pct_change_signal)

            old_value = q_table[state_index, action]
            next_max = np.max(q_table[next_state_index])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            if not np.isnan(new_value) and not np.isinf(new_value):
                q_table[state_index, action] = new_value
            else:
                print(f"Warning: Invalid Q-value at episode {episode}, step {step+1}. Skipping update.")

            state = next_state
            step += 1

            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)
        

    print("\nTraining Completed!\n")

    test_env = TradingEnvironment(
        actual_prices=test_prices,
        aroon_signal=normalized_test_aroon_signal,
        rsi_signal=normalized_test_rsi_signal,
        ema_signal=normalized_test_ema,
        pct_change=normalized_test_pct,
    )

    state = test_env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        aroon_signal, rsi_signal, ema_signal, holdings, pct_change_signal = state
        state_index = get_state_index(aroon_signal, rsi_signal, ema_signal, holdings, pct_change_signal)

        action = np.argmax(q_table[state_index])

        next_state, reward, done = test_env.step(action)
        total_reward += reward

        state = next_state
        step += 1

    print(f'\nFinal Net Worth on Testing Data: ${test_env.net_worth:.2f}')

    trades_df = pd.DataFrame(test_env.trades)
    test_data  = test_data.reset_index()
    test_data = pd.concat([test_data, trades_df[['signals', 'trade_type']]], axis=1)
    return test_data

def perform_backtest(csv_file_path):
    """
    Perform backtesting using the untrade SDK.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing historical price data and signals.

    Returns:
    - result (generator): Generator object that yields backtest results.
    """
    # Create an instance of the untrade client
    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        jupyter_id="team67_zelta_hpps",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=1,  # Adjust leverage as needed
        # result_type="Q"
    )

    return result

def perform_backtest_large_csv(csv_file_path):
    """
    Perform a backtest for large files using chunked uploads.
    Parameters:
    csv_file_path (str): Path to the CSV file.
    Returns:
    dict: Backtest results.
    """
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024 # 90 MB chunks
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0

    with open(csv_file_path, "rb") as f:
        while chunk_data := f.read(chunk_size):
            chunk_file_path = f"/tmp/{file_id}chunk{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)
            result = client.backtest(
                file_path=chunk_file_path,
                leverage=1,
                jupyter_id="team67_zelta_hpps",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
            )
            for value in result:
                print(value)
            os.remove(chunk_file_path)
            chunk_number += 1
    return result
    
def main():

    data = pd.read_csv("BTC_2019_2023_1h.csv")
    # Process the data
    data = process_data(data)

    # Strategize on data
    strategized_data = strat(data)

    # Save processed data to CSV file
    csv_file_path = "btc_strategy_results.csv"
    strategized_data.to_csv(csv_file_path, index=False)

    # Perform backtest on processed data
    backtest_result = perform_backtest(csv_file_path)

    # Get the last value of backtest result
    last_value = None
    for value in backtest_result:
        # print(value)  # Uncomment to see the full backtest result (backtest_result is a generator object)
        last_value = value
    print(last_value)

if __name__ == "__main__":
    main()