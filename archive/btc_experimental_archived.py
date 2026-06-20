
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import pandas_ta as ta  # Import pandas_ta for technical analysis
from fbm import FBM
from hurst import compute_Hc
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

def process_data(data_incoming):
    def add_consolidation_signal(data, window_size=10, threshold=0.5):
        # Adds a Consolidation_Signal column to the DataFrame based on price stability.

        # Calculate rolling standard deviation
        data['MaxHigh'] = data['high'].rolling(window=window_size).max()
        data['MinLow'] = data['low'].rolling(window=window_size).min()
        
        # Calculate the threshold range
        data['UpperThreshold'] = data['MinLow'] + (threshold * (data['MaxHigh'] - data['MinLow']))
        data['LowerThreshold'] = data['MaxHigh'] - (threshold * (data['MaxHigh'] - data['MinLow']))

        # Identify consolidation periods
        data['Consolidation_Signal'] = ((data['high'] < data['UpperThreshold']) & (data['low'] > data['LowerThreshold'])).astype(int)

        return data


    def load_data(df, start_datetime, end_datetime, drop_columns=None):
        """
        Load and preprocess data from a CSV file.

        Parameters:
        - file_path (str): Path to the CSV file.
        - start_datetime (str): Start datetime for filtering the data.
        - end_datetime (str): End datetime for filtering the data.
        - drop_columns (list, optional): List of columns to drop from the data.

        Returns:
        - pd.DataFrame: Preprocessed data.
        """

        # Load data from CSV file
        data = df.copy()

        
        # Drop specified columns if they exist
        if drop_columns:
            existing_cols = [col for col in drop_columns if col in data.columns]
            data.drop(existing_cols, axis=1, inplace=True)
        
        # Convert 'datetime' to datetime object and set as index
        data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')  # Adjust format as needed

        data.set_index('datetime', inplace=True)
        
        # Filter data between start and end datetime
        data = data.loc[start_datetime:end_datetime].copy()
        
        # Verify essential columns exist
        essential_columns = ['close', 'high', 'low', 'volume']
        for col in essential_columns:
            if col not in data.columns:
                raise ValueError(f"Missing essential column: '{col}' in the data.")
        
        # Handle missing values in essential columns by forward filling
        data[essential_columns] = data[essential_columns].fillna(method='ffill')
        
        # Calculate Moving Averages and MA Signal
        data['MA7'] = data['close'].rolling(window=7).mean()
        data['MA14'] = data['close'].rolling(window=14).mean()
        data['MA_Signal'] = 0
        data.loc[data['MA7'] > data['MA14'], 'MA_Signal'] = 1
        data.loc[data['MA7'] < data['MA14'], 'MA_Signal'] = -1

        # Calculate Aroon Indicator with pandas_ta
        data['Aroon_Up'] = 100 * (14 - data['high'].rolling(window=15).apply(lambda x: x.argmax())) / 14
        data['Aroon_Down'] = 100 * (14 - data['low'].rolling(window=15).apply(lambda x: x.argmin())) / 14
        data['Aroon_Signal'] = (data['Aroon_Up'] > data['Aroon_Down']).astype(int)

        # Calculate RSI
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

        # Calculate Returns
        data['returns'] = np.log(data.close / data.close.shift(1))

        # Calculate Exponential Moving Averages (EMA) and EMA Signal
        data['EMA7'] = data['close'].ewm(span=7, adjust=False).mean()
        data['EMA14'] = data['close'].ewm(span=14, adjust=False).mean()
        data['EMA28'] = data['close'].ewm(span=28, adjust=False).mean()
        data['EMA_Signal'] = 0
        data.loc[(data['EMA7'] > data['EMA14']) & (data['EMA14'] > data['EMA28']), 'EMA_Signal'] = 1
        data.loc[(data['EMA7'] < data['EMA14']) & (data['EMA14'] < data['EMA28']), 'EMA_Signal'] = -1
        
        # Ensure 'EMA_Signal' is of integer type
        data['EMA_Signal'] = data['EMA_Signal'].astype(int)
        
        # Calculate Percent Change
        data['pct_change'] = data['close'].pct_change(periods=1).fillna(0) * 100
        
        # Add Consolidation Signal
        data = add_consolidation_signal(data, window_size=10, threshold=0.5)
        
        # Drop rows where any of the EMAs are NaN
        data.dropna(subset=['EMA7', 'EMA14', 'EMA28'], inplace=True)
        
        # Handle missing values resulting from indicator calculations
        data.dropna(inplace=True)
        
        # Convert signal columns to integer types to avoid FutureWarnings
        signal_columns = ['Aroon_Signal', 'RSI_Signal', 'EMA_Signal', 'Consolidation_Signal']
        data[signal_columns] = data[signal_columns].astype(int)
        
        return data


    # Define file paths for training and testing data

    # Define date ranges for training and testing data
    train_start_datetime = '2020-01-01'
    train_end_datetime = '2022-12-31'
    test_start_datetime = '2023-01-01'
    test_end_datetime = '2024-01-01'

    # Load and preprocess training data
    train_data = load_data(data_incoming,train_start_datetime, train_end_datetime, drop_columns=['Unnamed: 0'])

    # Load and preprocess testing data, dropping unnecessary columns
    test_data = load_data(data_incoming,test_start_datetime, test_end_datetime, drop_columns=['Unnamed: 0'])

    np.random.seed(1)

    print(train_data)
    print(test_data)
    def geometric_fractional_brownian_motion(S0, mu, sigma, T, N, H):
        """ 
        Simulates a Geometric Fractional Brownian Motion path using fbm.
        
        Parameters:
        - S0 (float): Initial stock price.
        - mu (float): Drift coefficient.
        - sigma (float): Volatility coefficient.
        - T (float): Total time.
        - N (int): Number of time steps.
        - H (float): Hurst parameter.
        
        Returns:
        - np.ndarray: Simulated price path.
        """
        dt = T / N
        t = np.linspace(dt, T, N + 1)
        fBm_process = FBM(n=N, hurst=H, length=T)
        B_H = fBm_process.fbm()  # Fractional Brownian motion path
        val = (t**2*H)*(t**-1)
        prices = S0 * np.exp((mu - 0.5 * (sigma ** 2) * val) * t + sigma * B_H)
        return prices

    def compute_fgbm_signals(data, lookback=100, T=3, N=1000, num_simulations=500):
        """
        Compute fGBM signals for the given data.
        
        Parameters:
        - data (pd.DataFrame): Input data containing 'close' prices and 'returns'.
        - lookback (int): Lookback period for computing signals.
        - T (float): Total time for fGBM simulation.
        - N (int): Number of time steps in fGBM simulation.
        - num_simulations (int): Number of fGBM simulations to run.
        
        Returns:
        - pd.DataFrame: Data with added 'fgbm_signal' column.
        """
        data = data.copy()
        data['fgbm_signal'] = 0  # Initialize with neutral signals

        for i in range(lookback, len(data)):
            window_df = data.iloc[i - lookback:i+1]
            ts = window_df['close'].values
            H, c, data_reg = compute_Hc(ts)  # Compute Hurst exponent
            
            S0 = window_df['close'].iloc[-1]
            mu = window_df['returns'].mean()
            sigma = window_df['returns'].std()
            
            simulated_prices = 0
            avg_price = 0
            for _ in range(num_simulations):
                gfbm = geometric_fractional_brownian_motion(S0, mu, sigma, T, N, H)
                simulated_prices += gfbm[-1]
            avg_price = simulated_prices / num_simulations
            
            # Determine the signal based on the average forecasted price
            forecast_ratio = avg_price / S0
            if forecast_ratio > 1.02:
                signal = 1  # Predict price will go up by more than 2%
            elif forecast_ratio < 0.98:
                signal = -1  # Predict price will go down by more than 2%
            else:
                signal = 0  # No significant change
            
            data.at[data.index[i], 'fgbm_signal'] = signal

            if i % 500 == 0:
                print(f"Computed fGBM signal for index {i} / {len(data)}")

        return data


    # Precompute fGBM signals for training and testing data
    print("Precomputing fGBM signals for training data...")
    train_data = compute_fgbm_signals(train_data, lookback=100, T=3, N=96, num_simulations=20)
    print("Training data fGBM signals computed.\n")

    print("Precomputing fGBM signals for testing data...")
    test_data = compute_fgbm_signals(test_data, lookback=100, T=3, N=96, num_simulations=20)
    print("Testing data fGBM signals computed.\n")

    final_data = pd.concat([train_data, test_data])
    return final_data

def strat(data_incoming):
    train_start_datetime = '2020-01-01'
    train_end_datetime = '2022-12-31'
    test_start_datetime = '2023-01-01'
    test_end_datetime = '2024-01-01'
    train_data = data_incoming.loc[train_start_datetime:train_end_datetime].copy()
    test_data = data_incoming.loc[test_start_datetime:test_end_datetime].copy()
    train_prices = train_data['close'].values  # Use denoised prices
    train_aroon_signal = train_data['Aroon_Signal'].values
    train_rsi_signal = train_data['RSI_Signal'].values
    train_ema_signal = train_data['EMA_Signal'].values
    train_fgbm_signal = train_data['fgbm_signal'].values  # Precomputed fGBM signals
    train_pct_change = train_data['pct_change'].values
    train_consolidation_signal = train_data['Consolidation_Signal'].values 

    # Extract essential variables for testing
    test_prices = test_data['close'].values  # Use denoised prices
    test_aroon_signal = test_data['Aroon_Signal'].values
    test_rsi_signal = test_data['RSI_Signal'].values
    test_ema_signal = test_data['EMA_Signal'].values
    test_fgbm_signal = test_data['fgbm_signal'].values  # Precomputed fGBM signals
    test_pct_change = test_data['pct_change'].values
    test_consolidation_signal = test_data['Consolidation_Signal'].values  # New consolidation signal

    # Combine features for training and testing
    train_features = train_data[['close', 'Aroon_Signal', 'RSI_Signal', 'EMA_Signal', 'fgbm_signal', 'pct_change']].values
    test_features = test_data[['close', 'Aroon_Signal', 'RSI_Signal', 'EMA_Signal', 'fgbm_signal', 'pct_change']].values

    # Normalize signals for training
    normalized_train_aroon_signal = train_aroon_signal  # Already binary (0 or 1)
    normalized_train_rsi_signal = train_rsi_signal      # Binary (-1, 0, 1)
    normalized_train_ema = train_ema_signal
    normalized_train_fgbm_signal = train_fgbm_signal    # fGBM signals are already categorical (-1, 0, 1)
    normalized_train_pct = train_pct_change

    # Normalize signals for testing
    normalized_test_aroon_signal = test_aroon_signal
    normalized_test_rsi_signal = test_rsi_signal
    normalized_test_ema = test_ema_signal
    normalized_test_fgbm_signal = test_fgbm_signal
    normalized_test_pct = test_pct_change


 
    # Define constants for trading environment
    MIN_PRICE = 1e-6            # Minimum valid price to avoid division by zero
    MIN_TRADE_AMOUNT = 5000     # Minimum amount in USD to execute a trade
    COMMISSION_RATE = 0.0015    # Commission rate for each trade (0.15%)
    MAX_SHORT_POSITION = 0.75   # Maximum fraction of balance to use for short positions
    STOP_LOSS_PERCENT = 0.025   # Stop loss percentage to limit losses

   
    class TradingEnvironment:
        def __init__(self, actual_prices, aroon_signal, rsi_signal, ema_signal, fgbm_signal, pct_change, consolidation_signals):
            self.actual_prices = actual_prices
            self.aroon_signal = aroon_signal
            self.consolidation = consolidation_signals
            self.rsi_signal = rsi_signal
            self.ema_signal = ema_signal
            self.fgbm_signal = fgbm_signal
            self.pct_change = pct_change
            self.n_steps = len(actual_prices)
            self.current_step = 0
            self.position = 0          # Position: 0 (none), 1 (long), -1 (short)
            self.balance = 10000.0     # Starting balance in USD
            self.net_worth = self.balance
            self.initial_balance = self.balance
            self.trades = []
            self.entry_price = 0.0
            self.holdings = 0.0        # Quantity of asset held (positive for long, negative for short)
            self.history = [self.net_worth]  # To track net worth over time
            self.cooldown = 0          # Steps remaining before next trade
            self.last_worth = self.balance

        def _get_observation(self):
            # Return the current state observation
            return np.array([
                self.aroon_signal[self.current_step],
                self.rsi_signal[self.current_step],
                self.ema_signal[self.current_step],
                self.position,
                self.fgbm_signal[self.current_step],
                self.pct_change[self.current_step]
            ])
        
        def reset(self):
            # Reset the environment to the initial state
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
            reward = 0.0  # Initialize reward
            traded = 0
            # Execute action
            if action == 1 and self.position == 0 and self.balance > MIN_TRADE_AMOUNT:
                # Buy to enter long position
                self.last_worth = self.net_worth
                total_cost = self.balance
                commission = total_cost * COMMISSION_RATE 
                investment_amount = total_cost - commission
                self.holdings = investment_amount / actual_price
                self.position = 1  # Long position
                self.entry_price = actual_price
                self.balance = 0.0  # All balance used
                self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'long',
                        'price': actual_price,
                        'commission': commission,
                        'signals' : 1
                    })
                traded  =1 
                reward -= commission  # Small penalty for making a trade

            elif action == 2 and self.position == 1:
                # Sell to exit long position
                gross_sale = self.holdings * actual_price
                commission = gross_sale * COMMISSION_RATE
                net_sale = gross_sale - commission
                self.balance += net_sale  # Add net sale to balance
                self.holdings = 0.0
                profit = (self.balance + self.holdings * actual_price) - self.last_worth
                self.position = 0  # No position
                self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'close',
                        'price': actual_price,
                        'commission': commission,
                        'signals' : -1
                    })
                traded = 1
                reward += profit

            elif action == 3 and self.position == 0 and self.balance > MIN_TRADE_AMOUNT:
                # Short to enter short position
                self.last_worth = self.net_worth
                short_value = self.balance * MAX_SHORT_POSITION
                gross_proceeds = short_value
                commission_entry = gross_proceeds * COMMISSION_RATE
                net_proceeds = gross_proceeds - commission_entry
                units_to_short = gross_proceeds / actual_price  # Use gross proceeds to calculate units
                self.holdings = -units_to_short  # Negative holdings for short position
                self.position = -1  # Short position
                self.entry_price = actual_price
                self.balance += net_proceeds  # Add net proceeds to balance
                self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'short',
                        'price': actual_price,
                        'commission': commission_entry,
                        'signals': -1
                    })
                traded = 1
                reward -= commission_entry  # Small penalty for making a trade

            elif action == 4 and self.position == -1:
                # Cover short position
                traded = 1
                gross_purchase = -self.holdings * actual_price
                commission_exit = gross_purchase * COMMISSION_RATE
                total_cost = gross_purchase + commission_exit
                self.balance -= total_cost  # Pay to cover the short position
                profit = (self.balance + self.holdings * actual_price) - self.last_worth
                if self.balance < 0:
                    # Agent is bankrupt
                    self.balance = 0.0
                    self.holdings = 0.0
                    self.position = 0
                    self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'close',
                            'price': actual_price,
                            'commission': commission_exit,
                            'signals':1
                        })
                    reward += -100000000
                    done = True
                    return self._get_observation(), reward, done
                else:
                    self.holdings = 0.0
                    self.position = 0  # No position
                    self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'close',
                            'price': actual_price,
                            'commission': commission_exit,
                            'signals':1
                        })
                    reward += profit

            else:
                # Hold

                consolidation_val = self.consolidation[self.current_step]

                if self.position != 0:
                    reward += (self.balance + self.holdings * actual_price) - self.net_worth
                else:
                    if consolidation_val == 0:
                        reward -= (abs(actual_price - self.actual_prices[self.current_step-1])) * self.net_worth / self.actual_prices[self.current_step-1]
                    else:
                        reward += 5000

            # Check for stop loss
            if self.position == 1:
                # Long position
                if actual_price <= self.entry_price * (1 - STOP_LOSS_PERCENT):
                    # Stop loss triggered
                    traded =1
                    gross_sale = self.holdings * actual_price
                    commission = gross_sale * COMMISSION_RATE
                    net_sale = gross_sale - commission
                    self.balance += net_sale  # Add net sale to balance
                    self.holdings = 0.0
                    profit = (self.balance + self.holdings * actual_price) - self.last_worth
                    self.position = 0  # No position
                    self.trades.append({
                            'step': self.current_step,
                            'trade_type': 'close',
                            'price': actual_price,
                            'commission': commission,
                            'signals':-1
                        })
                    reward += profit

            elif self.position == -1:
                # Short position
                if actual_price >= self.entry_price * (1 + STOP_LOSS_PERCENT):
                    # Stop loss triggered
                    traded = 1
                    gross_purchase = -self.holdings * actual_price
                    commission_exit = gross_purchase * COMMISSION_RATE
                    total_cost = gross_purchase + commission_exit
                    self.balance -= total_cost  # Pay to cover the short position
                    profit = (self.balance + self.holdings * actual_price) - self.last_worth
                    if self.balance < 0:
                        # Agent is bankrupt
                        self.balance = 0.0
                        self.holdings = 0.0
                        self.position = 0
                        self.trades.append({
                            'step': self.current_step,
                            'type': 'cover',
                            'price': actual_price,
                            'commission': commission_exit
                        })
                        reward += -100000000
                        done = True
                        return self._get_observation(), reward, done
                    else:
                        self.holdings = 0.0
                        self.position = 0  # No position
                        self.trades.append({
                            'step': self.current_step,
                            'type': 'cover',
                            'price': actual_price,
                            'commission': commission_exit
                        })
                        reward += profit
            
            if(traded == 0):

                self.trades.append({
                    'step': self.current_step,
                    'trade_type': ' ',
                    'price': actual_price,
                    'commission': 0,
                    'signals':0
                    })
            
            self.current_step += 1

            if self.current_step >= self.n_steps - 1:
                done = True
                if self.position ==1 :
                    self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'close',
                        'price': actual_price,
                        'commission': 0,
                        'signals':-1
                        }) 
                elif self.position == -1:
                    self.trades.append({
                        'step': self.current_step,
                        'trade_type': 'close',
                        'price': actual_price,
                        'commission': 0,
                        'signals':1
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



    # Define the number of possible actions
    action_size = 5  # [Hold, Buy, Sell, Short, Cover Short]

    # Set the seed value for reproducibility
    seed_value = 1

    # Define hyperparameters for the Q-learning algorithm
    alpha = 0.05         # Learning rate
    gamma = 0.95         # Discount factor
    epsilon = 1.0        # Starting exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.1
    episodes = 430      # Number of episodes for training


    n_pct_bins = 50   # Number of bins for percent change
    n_signal_bins = 3   # Possible values for signals: -1, 0, 1
    n_holdings_states = 3  # Possible states for holdings: -1, 0, 1
    n_aroon_bins = 2  # Possible values for Aroon signal: 0, 1

    # Calculate total state size
    state_size = (
        n_aroon_bins *  # Aroon Signal
        n_signal_bins *  # RSI Signal
        n_signal_bins *  # EMA Signal
        n_holdings_states *  # Holdings State
        n_signal_bins *  # fGBM Signal
        n_pct_bins  # Percent Change
    )


 
    q_table_file = "my_q_table.npy"
    q_table = np.zeros((state_size, action_size))

    def get_price_bin(pct):
        # Define bin edges for percent change
        bin_edges = np.linspace(-20, 20, n_pct_bins + 1)
        # Clip pct_change_signal to the bin range to prevent NaNs
        pct_clipped = np.clip(pct, -20, 20)
        # Assign each 'pct_change_clipped' to a bin
        pct_bin = np.digitize(pct_clipped, bins=bin_edges, right=False) - 1
        # Ensure bin index is within valid range
        return int(np.clip(pct_bin, 0, n_pct_bins - 1))

    def get_signal_bin(signal):
        # Map signal values (-1, 0, 1) to bins (0, 1, 2)
        signal_mapping = {-1: 0, 0: 1, 1: 2}
        return signal_mapping.get(int(signal), 1)  # Default to neutral if unexpected value

    def get_aroon_bin(signal):
        # Map Aroon signal values (0, 1) to bins (0, 1)
        signal_mapping = {0: 0, 1: 1}
        return signal_mapping.get(int(signal), 1)  # Default to neutral if unexpected value

    def get_holdings_state(holdings):
        # Map holdings state (-1, 0, 1) to bins (0, 1, 2)
        holdings_mapping = {-1: 0, 0: 1, 1: 2}
        return holdings_mapping.get(int(holdings), 1)  # Default to neutral

    def get_state_index(aroon_signal, rsi_signal, ema_signal, holdings, fgbm_signal, pct_change_signal):
        # Map input variables to their respective bins
        aroon_bin = get_signal_bin(aroon_signal)
        rsi_bin = get_signal_bin(rsi_signal)
        ema_bin = get_signal_bin(ema_signal)
        holdings_bin = get_holdings_state(holdings)
        fgbm_bin = get_signal_bin(fgbm_signal)
        pct_bin = get_price_bin(pct_change_signal)

        # Calculate unique state index within bounds
        state_index = (
            aroon_bin * (n_signal_bins ** 3 * n_pct_bins * n_holdings_states) +  # Aroon Signal
            rsi_bin * (n_signal_bins ** 2 * n_pct_bins * n_holdings_states) +    # RSI Signal
            ema_bin * (n_signal_bins ** 1 * n_pct_bins * n_holdings_states) +    # EMA Signal
            holdings_bin * (n_signal_bins * n_pct_bins) +                        # Holdings State
            fgbm_bin * n_pct_bins +                                              # fGBM Signal
            pct_bin                                                              # Percent Change
        )

        # Ensure the state index is within valid bounds
        state_index = int(np.clip(state_index, 0, state_size - 1))

        return state_index
    train_env = TradingEnvironment(
            
        # Initializes the TradingEnvironment with the given training data.
        # Parameters:
        # actual_prices (pd.Series): Series containing the actual prices of the asset.
        # aroon_signal (pd.Series): Series containing the normalized Aroon indicator signals.
        # rsi_signal (pd.Series): Series containing the normalized Relative Strength Index (RSI) signals.
        # ema_signal (pd.Series): Series containing the normalized Exponential Moving Average (EMA) signals.
        # fgbm_signal (pd.Series): Series containing the normalized Forecasting Gradient Boosting Model (FGBM) signals.
        # pct_change (pd.Series): Series containing the normalized percentage change in prices.
        # consolidation_signals (pd.Series): Series containing the consolidation signals for the training data.
        
        actual_prices= train_prices,
        aroon_signal=normalized_train_aroon_signal,
        rsi_signal=normalized_train_rsi_signal,
        ema_signal=normalized_train_ema,
        fgbm_signal=normalized_train_fgbm_signal,
        pct_change= normalized_train_pct,
        consolidation_signals=train_consolidation_signal
    )

    
    print("Starting Training...\n")

    for episode in range(1, episodes + 1):
        # Reset the environment and initialize variables
        state = train_env.reset()
        total_reward = 0
        step = 0

        while True:
            # Unpack the current state
            aroon_signal, rsi_signal, ema_signal, holdings, fgbm_signal, pct_change_signal = state
            state_index = get_state_index(aroon_signal, rsi_signal, ema_signal, holdings, fgbm_signal, pct_change_signal)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(action_size)  # Explore: select a random action
            else:
                action = np.argmax(q_table[state_index])  # Exploit: select the best action based on Q-table

            # Execute the selected action
            next_state, reward, done = train_env.step(action)
            total_reward += reward

            # Unpack the next state
            next_aroon_signal, next_rsi_signal, next_ema_signal, next_holdings, next_fgbm_signal, next_pct_change_signal = next_state
            next_state_index = get_state_index(next_aroon_signal, next_rsi_signal, next_ema_signal, next_holdings, next_fgbm_signal, next_pct_change_signal)

            # Update Q-table using the Bellman equation
            old_value = q_table[state_index, action]
            next_max = np.max(q_table[next_state_index])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            # Update Q-table only if new_value is valid
            if not np.isnan(new_value) and not np.isinf(new_value):
                q_table[state_index, action] = new_value
            else:
                print(f"Warning: Invalid Q-value at episode {episode}, step {step+1}. Skipping update.")

            # Transition to the next state
            state = next_state
            step += 1

            if done:
                break

        # Decay epsilon to reduce exploration over time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

        # Print progress every 10 episodes
        if episode % 10 == 0 or episode == 1:
            print(f'Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Final Net Worth: ${train_env.net_worth:.2f}')
        
        # Save Q-table every 50 episodes
        if episode % 50 == 0:
            np.save(q_table_file, q_table)
            print(f"Q-table saved to {q_table_file} at episode {episode}")

    print("\nTraining Completed!\n")

    q_table = np.load(q_table_file)
    test_env = TradingEnvironment(
        actual_prices=test_prices,  # Actual prices from the test data
        aroon_signal=normalized_test_aroon_signal,  # Normalized Aroon signals
        rsi_signal=normalized_test_rsi_signal,  # Normalized RSI signals
        ema_signal=normalized_test_ema,  # Normalized EMA signals
        fgbm_signal=normalized_test_fgbm_signal,  # Normalized fGBM signals
        pct_change=normalized_test_pct,  # Normalized percentage change
        consolidation_signals=test_consolidation_signal  # Consolidation signals
    )


    # Reset the environment to the initial state for testing
    state = test_env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        # Unpack the current state
        aroon_signal, rsi_signal, ema_signal, holdings, fgbm_signal, pct_change_signal = state
        state_index = get_state_index(aroon_signal, rsi_signal, ema_signal, holdings, fgbm_signal, pct_change_signal)

        # Select action based on Q-table (no exploration during testing)
        action = np.argmax(q_table[state_index])

        # Execute the selected action
        next_state, reward, done = test_env.step(action)
        total_reward += reward

        # Transition to the next state
        state = next_state
        step += 1

    # Print the final net worth after testing
    print(f'\nFinal Net Worth on Testing Data: ${test_env.net_worth:.2f}')
    trades_df = pd.DataFrame(test_env.trades)
    test_data  = test_data.reset_index()
    test_data = pd.concat([test_data, trades_df[['signals', 'trade_type']]], axis=1)
    return test_data

if __name__ == "__main__":
    # Define file path
    data = pd.read_csv("BTC_2019_2023_1h.csv")
    print(data)
    data = process_data(data)

    # Apply strategy
    strategized_data = strat(data)

    csv_file_path = "strategy_results_final.csv"

    # Save processed data to CSV file
    strategized_data.to_csv(csv_file_path, index=False)
    

    # Perform backtest on processed data
    # backtest_result = perform_backtest(csv_file_path)

    # Get the last value of backtest result
    # last_value = None
    # for value in backtest_result:
    #     # print(value)  # Uncomment to see the full backtest result (backtest_result is a generator object)
    #     last_value = value
    # print(last_value)