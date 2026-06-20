import pandas as pd
from untrade.client import Client
from pprint import pprint

import numpy as np
from scipy.stats import pearsonr
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from colorama import Fore, Back, Style
# import pywt

def process_data(data):
    btc_data = pd.read_csv("BTC_1hr_data_file_path", parse_dates=['datetime'])
    eth_data = pd.read_csv("ETH_1hr_data_file_path", parse_dates=['datetime'])

    btc_data = btc_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(
        columns={'open': 'BTC_Open', 'high': 'BTC_High', 'low': 'BTC_Low', 'close': 'BTC_Close', 'volume': 'BTC_Volume'}
    )
    eth_data = eth_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(
        columns={'open': 'ETH_Open', 'high': 'ETH_High', 'low': 'ETH_Low', 'close': 'ETH_Close', 'volume': 'ETH_Volume'}
    )

    data = pd.merge(btc_data, eth_data, on='datetime').dropna()

    start_date = '2020-01-01'
    end_date = '2023-12-31'
    data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)].reset_index(drop=True)

    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window).mean()
        avg_loss = pd.Series(loss).rolling(window=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        short_ema = data.ewm(span=short_window, adjust=False).mean()
        long_ema = data.ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        return macd_line, signal_line

    def calculate_atr(high, low, close, window=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    # Constants 
    CORRELATION_WINDOW = 7

    data.columns = [col.lower() for col in data.columns]

    # Calculate RSI indicators
    data['eth_rsi'] = calculate_rsi(data['eth_close'])
    data['btc_rsi'] = calculate_rsi(data['btc_close'])

    # Calculate MACD indicators
    data['btc_macd'], data['btc_macd_signal'] = calculate_macd(data['btc_close'])
    data['eth_macd'], data['eth_macd_signal'] = calculate_macd(data['eth_close'])

    # Calculate correlation and ATR
    data['btc_eth_correlation'] = data['btc_close'].rolling(CORRELATION_WINDOW).corr(data['eth_close'])
    data['btc_atr'] = calculate_atr(data['btc_high'], data['btc_low'], data['btc_close'])

    # Calculate moving averages
    data['btc_ma_20'] = data['btc_close'].rolling(window=20).mean()
    data['btc_ma_50'] = data['btc_close'].rolling(window=50).mean()
    data['eth_ma_20'] = data['eth_close'].rolling(window=20).mean()
    data['eth_ma_50'] = data['eth_close'].rolling(window=50).mean()

    # Calculate Aroon indicators
    data['eth_aroon_up'] = data['eth_high'].rolling(window=24).apply(lambda x: (24 - x.argmax()) / 24, raw=True)
    data['eth_aroon_down'] = data['eth_low'].rolling(window=24).apply(lambda x: (24 - x.argmin()) / 24, raw=True)

    return data


def strat(data):
    # Initialize signals column
    data['signals'] = 0
    data['trade_type'] = ""


    # Initialize variables for trading strategy
    entry_price = None
    entry_date = None
    current_position = 0  # 0 = no position, 1 = long position, -1 = short position
    time_stop_loss_hours = 20 * 24  # Convert 20 days to hours
    trailing_stop_pct = 0.10  
    highest_since_entry = None
    lowest_since_entry = None
    rsi_threshold = 60
    correlation_threshold = 0.6

    for i in range(len(data)):
        current_price = data['eth_close'].iloc[i]
        current_time = data['datetime'].iloc[i]
        
        # Close any open positions at the end of the data
        if i == len(data) - 1 and current_position != 0:
            if current_position == 1:
                data.at[i, 'signals'] = -1
                data.at[i, 'trade_type'] = 'close'
                current_position = 0
            elif current_position == -1:
                data.at[i, 'signals'] = 1
                data.at[i, 'trade_type'] = 'close'
                current_position = 0
            continue
        
        # Handle trailing stops for long positions
        if current_position == 1:
            if highest_since_entry is None:
                highest_since_entry = current_price
            else:
                highest_since_entry = max(highest_since_entry, current_price)
            
            # Check if price dropped below trailing stop threshold
            trailing_stop_price = highest_since_entry * (1 - trailing_stop_pct)
            if current_price <= trailing_stop_price:
                data.at[i, 'signals'] = -1
                data.at[i, 'trade_type'] = 'close'
                current_position = 0
                entry_price = None
                entry_date = None
                highest_since_entry = None
                continue
                
        # Handle trailing stops for short positions    
        elif current_position == -1:
            if lowest_since_entry is None:
                lowest_since_entry = current_price
            else:
                lowest_since_entry = min(lowest_since_entry, current_price)
                
            # Check if price rose above trailing stop threshold
            trailing_stop_price = lowest_since_entry * (1 + trailing_stop_pct)
            if current_price >= trailing_stop_price:
                data.at[i, 'signals'] = 1
                data.at[i, 'trade_type'] = 'close'
                current_position = 0
                entry_price = None
                entry_date = None
                lowest_since_entry = None
                continue
        
        # Check time-based stop loss
        if current_position != 0 and entry_price is not None:
            time_since_entry = (current_time - entry_date).total_seconds() / 3600
            if time_since_entry >= time_stop_loss_hours:
                # Close position if held longer than time threshold
                data.at[i, 'signals'] = -1 if current_position == 1 else 1
                data.at[i, 'trade_type'] = 'close'
                current_position = 0
                entry_price = None
                entry_date = None
                highest_since_entry = None
                lowest_since_entry = None
                continue

        # Dyanmic ATR entry conditions
        if data['btc_atr'][i] < 0.01*data['btc_open'][i]: 
            
            # Check for bullish setup conditions
            if (data['eth_rsi'][i] > rsi_threshold and
                data['btc_eth_correlation'][i] > correlation_threshold and
                data['btc_macd'][i] > data['btc_macd_signal'][i] and
                data['eth_aroon_up'][i] > data['eth_aroon_down'][i]):

                # Square off short position if exists
                if current_position == -1:
                    data.at[i, 'signals'] = 1
                    data.at[i, 'trade_type'] = 'close'
                    current_position = 0
                    entry_price = None
                    entry_date = None
                    lowest_since_entry = None

                # Enter long position if no current position
                elif current_position == 0:
                    data.at[i, 'signals'] = 1
                    data.at[i, 'trade_type'] = 'long'
                    current_position = 1
                    entry_price = current_price
                    entry_date = data['datetime'].iloc[i]
                    highest_since_entry = current_price

            # Check for bearish setup conditions
            elif (data['eth_rsi'][i] < 30 and
                data['btc_eth_correlation'][i] > correlation_threshold and
                data['btc_macd'][i] > data['btc_macd_signal'][i] and
                data['eth_aroon_up'][i] > data['eth_aroon_down'][i]):

                # Exit long position if exists
                if current_position == 1:
                    data.at[i, 'signals'] = -1
                    data.at[i, 'trade_type'] = 'close'
                    current_position = 0
                    entry_price = None
                    entry_date = None
                    highest_since_entry = None

                # Enter short position if no current position
                elif current_position == 0:
                    data.at[i, 'signals'] = -1
                    data.at[i, 'trade_type'] = 'short'
                    current_position = -1
                    entry_price = current_price
                    entry_date = data['datetime'].iloc[i]
                    lowest_since_entry = current_price
    
    data.columns = data.columns.str.lower().str.strip()
    final_data = data[['datetime', 'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume', 'signals', 'trade_type']]
    
    final_data = final_data.rename(columns={
        'eth_open': 'open',
        'eth_high': 'high',
        'eth_low': 'low',
        'eth_close': 'close',
        'eth_volume': 'volume'
    })

    return final_data

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
        jupyter_id="your_jupyter_id",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=1,  # Adjust leverage as needed
    )

    return result

if __name__ == "__main__":

    #data loaded from csv inside process_data function 
    
    data = process_data(None)

    # Apply strategy
    strategized_data = strat(data)

    csv_file_path = "strategy_results.csv"

    # Save processed data to CSV file
    strategized_data.to_csv(csv_file_path, index=False)

    # Perform backtest on processed data
    backtest_result = perform_backtest(csv_file_path)

    # Get the last value of backtest result
    last_value = None
    for value in backtest_result:
        # print(value)  # Uncomment to see the full backtest result (backtest_result is a generator object)
        last_value = value
    print(last_value)



