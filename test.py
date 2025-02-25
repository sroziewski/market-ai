import pandas as pd
import talib

from custom_indicators import cycle_oscillator, generate_signal_label, tc_top_bottom_finder, volume_flow_indicator, \
    visualize_find_tb_results


def calculate_indicators(klines_df):
    """Calculate RSI and full MACD (Line, Signal, Histogram)"""
    df = klines_df.copy()
    if 'volume' not in df.columns:
        raise ValueError("Volume data is required in the input DataFrame")

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD: Line, Signal, Histogram
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']

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

    df = cycle_oscillator(df)
    df_tb = tc_top_bottom_finder(df)
    df = generate_signal_label(df_tb)
    df = volume_flow_indicator(df)

    return df.dropna()


if __name__ == "__main__":
    sample_data = pd.read_csv("/home/simon/data/my/crypto/klines/BTCUSDT/BTCUSDT_1d.csv")
    last_1000_data = sample_data.tail(500)  # Get the last 1000 rows
    df_tb = tc_top_bottom_finder(last_1000_data)
    visualize_find_tb_results(df_tb)
    df = calculate_indicators(sample_data)
    i = 1
