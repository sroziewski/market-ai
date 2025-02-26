import os

import numpy as np
import pandas as pd
import talib
from dotenv import load_dotenv

from custom_indicators import cycle_oscillator, tc_top_bottom_finder, volume_flow_indicator, generate_signal_label, \
    one_hot_encode_column, visualize_cycle_oscillator, visualize_vfi

# Load environment variables from the `.env` file
load_dotenv()

# features = ['open', 'high', 'low', 'volume', 'close', 'macd', 'macd_signal', 'macd_histogram', 'rsi', 'stoch_rsi_k',
#             'stoch_rsi_d', 'open_sma200_diff_pct', 'high_sma200_diff_pct', 'low_sma200_diff_pct',
#             'close_sma200_diff_pct',
#             'open_sma100_diff_pct', 'high_sma100_diff_pct', 'low_sma100_diff_pct', 'close_sma100_diff_pct',
#             'open_sma50_diff_pct', 'high_sma50_diff_pct', 'low_sma50_diff_pct', 'close_sma50_diff_pct',
#             'open_sma20_diff_pct', 'high_sma20_diff_pct', 'low_sma20_diff_pct', 'close_sma20_diff_pct',
#             'sma20_sma50_angle', 'sma20_sma100_angle', 'sma20_sma200_angle', 'sma50_sma100_angle',
#             'sma50_sma200_angle', 'sma100_sma200_angle', 'omed', 'oshort', 'omed_ob', 'omed_os', 'oshort_ob',
#             'oshort_os', 'signal_label', 'vfi', 'vfima', 'd']
features = ['open', 'high', 'low', 'volume', 'close', 'macd', 'macd_signal', 'macd_histogram', 'rsi', 'stoch_rsi_k',
            'stoch_rsi_d', 'open_sma200_diff_pct', 'high_sma200_diff_pct', 'low_sma200_diff_pct',
            'close_sma200_diff_pct',
            'open_sma100_diff_pct', 'high_sma100_diff_pct', 'low_sma100_diff_pct', 'close_sma100_diff_pct',
            'open_sma50_diff_pct', 'high_sma50_diff_pct', 'low_sma50_diff_pct', 'close_sma50_diff_pct',
            'open_sma20_diff_pct', 'high_sma20_diff_pct', 'low_sma20_diff_pct', 'close_sma20_diff_pct',
            'sma20_sma50_angle', 'sma20_sma100_angle', 'sma20_sma200_angle', 'sma50_sma100_angle',
            'sma50_sma200_angle', 'sma100_sma200_angle', 'omed', 'oshort', 'omed_ob', 'omed_os', 'oshort_ob',
            'oshort_os', 'vfi', 'vfima', 'd']


def calculate_indicators(klines_df):
    print("Calculating indicators...")
    """Calculate RSI and full MACD (Line, Signal, Histogram)"""
    df = klines_df.copy()
    if 'close' not in df.columns:
        raise ValueError("Close data is required in the input DataFrame")
    sma20 = df['close'].rolling(window=20, min_periods=1).mean()
    sma50 = df['close'].rolling(window=50, min_periods=1).mean()
    sma100 = df['close'].rolling(window=100, min_periods=1).mean()
    sma200 = df['close'].rolling(window=200, min_periods=1).mean()

    # Calculate percentage differences for SMA200
    df['open_sma200_diff_pct'] = ((df['open'] - sma200) / sma200) * 100
    df['high_sma200_diff_pct'] = ((df['high'] - sma200) / sma200) * 100
    df['low_sma200_diff_pct'] = ((df['low'] - sma200) / sma200) * 100
    df['close_sma200_diff_pct'] = ((df['close'] - sma200) / sma200) * 100

    # Calculate percentage differences for SMA100
    df['open_sma100_diff_pct'] = ((df['open'] - sma100) / sma100) * 100
    df['high_sma100_diff_pct'] = ((df['high'] - sma100) / sma100) * 100
    df['low_sma100_diff_pct'] = ((df['low'] - sma100) / sma100) * 100
    df['close_sma100_diff_pct'] = ((df['close'] - sma100) / sma100) * 100

    # Calculate percentage differences for SMA50
    df['open_sma50_diff_pct'] = ((df['open'] - sma50) / sma50) * 100
    df['high_sma50_diff_pct'] = ((df['high'] - sma50) / sma50) * 100
    df['low_sma50_diff_pct'] = ((df['low'] - sma50) / sma50) * 100
    df['close_sma50_diff_pct'] = ((df['close'] - sma50) / sma50) * 100

    # Calculate percentage differences for SMA20
    df['open_sma20_diff_pct'] = ((df['open'] - sma20) / sma20) * 100
    df['high_sma20_diff_pct'] = ((df['high'] - sma20) / sma20) * 100
    df['low_sma20_diff_pct'] = ((df['low'] - sma20) / sma20) * 100
    df['close_sma20_diff_pct'] = ((df['close'] - sma20) / sma20) * 100

    # Calculate angles (approximated as arctangent of slope difference)
    df['sma20_sma50_angle'] = np.arctan((sma20 - sma20.shift(1)) - (sma50 - sma50.shift(1)))
    df['sma20_sma100_angle'] = np.arctan((sma20 - sma20.shift(1)) - (sma100 - sma100.shift(1)))
    df['sma20_sma200_angle'] = np.arctan((sma20 - sma20.shift(1)) - (sma200 - sma200.shift(1)))
    df['sma50_sma100_angle'] = np.arctan((sma50 - sma50.shift(1)) - (sma100 - sma100.shift(1)))
    df['sma50_sma200_angle'] = np.arctan((sma50 - sma50.shift(1)) - (sma200 - sma200.shift(1)))
    df['sma100_sma200_angle'] = np.arctan((sma100 - sma100.shift(1)) - (sma200 - sma200.shift(1)))

    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # MACD: Line, Signal, Histogram
    df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26,
                                                                     signalperiod=9)

    # Stochastic RSI
    rsi = df['rsi'].values
    stoch_rsi_k, stoch_rsi_d = talib.STOCHRSI(
        rsi,
        timeperiod=14,  # 14-period window
        fastk_period=3,  # %K line smoothing
        fastd_period=3,  # %D line smoothing
        fastd_matype=0  # Exponential moving average for %D
    )
    df['stoch_rsi_k'] = stoch_rsi_k  # %K line of Stochastic RSI
    df['stoch_rsi_d'] = stoch_rsi_d  # %D line of Stochastic RSI (signal line)

    if "omed" in features:
        df = cycle_oscillator(df)
        print("cycle_oscillator applied")
    if "signal_label" in features:
        df_tb = tc_top_bottom_finder(df)
        print("tc_top_bottom_finder applied")
        df = generate_signal_label(df_tb)
        print("generate_signal_label applied")
        df = one_hot_encode_column(df, 'signal_label')
        print("one_hot_encode_column applied")
    if "vfi" in features:
        df = volume_flow_indicator(df)
        print("volume_flow_indicator applied")

    print("Finished calculating indicators.")

    return df.dropna()


if __name__ == "__main__":
    BASE_DIR = os.getenv("BASE_DIR")
    file_path = f"{BASE_DIR}/data/crypto/klines/BTCUSDT/BTCUSDT_1d.csv"
    sample_data = pd.read_csv(file_path)
    df_features = calculate_indicators(sample_data)
    train_size = int(0.8 * len(df_features))
    train_df = df_features[:train_size]
    test_df = df_features[train_size:]

    visualize_cycle_oscillator(df_features)
    visualize_vfi(df_features)
    i = 1
