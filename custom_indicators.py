import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib


def tc_top_bottom_finder(df, value_one=2, signal_strength=20):
    def nz(series, default=0):
        return series.ffill().fillna(default)

    def approximation(a, b):
        l0 = np.zeros(len(a))
        l1 = np.zeros(len(a))
        l2 = np.zeros(len(a))
        l3 = np.zeros(len(a))
        for i in range(1, len(a)):
            l0[i] = (1 - b) * a[i] + b * nz(pd.Series(l0))[i - 1]
            l1[i] = -b * l0[i] + nz(pd.Series(l0))[i - 1] + b * nz(pd.Series(l1))[i - 1]
            l2[i] = -b * l1[i] + nz(pd.Series(l1))[i - 1] + b * nz(pd.Series(l2))[i - 1]
            l3[i] = -b * l2[i] + nz(pd.Series(l2))[i - 1] + b * nz(pd.Series(l3))[i - 1]
        return (l0 + 2 * l1 + 2 * l2 + l3) / 6

    def find_indices(_list_to_check, _item_to_find):
        _indices = []
        for _idx, _value in enumerate(_list_to_check):
            if _value == _item_to_find:
                _indices.append(_idx)
        return _indices

    def get_major_indices(_data, _p):
        #  the returned indices are precisely at the time of a buy/sell signal
        return find_indices(_data, _p)  # 1 BUY / -1 SELL

    def lele(threshold_value, strength):
        n = len(df)
        bindex = np.zeros(n)
        sindex = np.zeros(n)
        result = np.zeros(n)

        for i in range(n):
            # Maintain previous indices for bindex and sindex
            if i > 0:
                bindex[i] = bindex[i - 1]
                sindex[i] = sindex[i - 1]

            # Ensure the 4-period index is valid
            if i >= 4:
                if df['close'].iloc[i] > df['close'].iloc[i - 4]:
                    bindex[i] += 1
                elif df['close'].iloc[i] < df['close'].iloc[i - 4]:
                    sindex[i] += 1

            # Ensure we have enough data for _strength periods before using slicing
            if i >= strength:
                max_high = np.max(df['high'].iloc[i - strength:i])
                min_low = np.min(df['low'].iloc[i - strength:i])

                # Check conditions for reset
                if (bindex[i] > threshold_value
                        and df['close'].iloc[i] < df['open'].iloc[i]
                        and df['high'].iloc[i] >= max_high):
                    bindex[i] = 0
                    result[i] = -1

                if (sindex[i] > threshold_value
                        and df['close'].iloc[i] > df['open'].iloc[i]
                        and df['low'].iloc[i] <= min_low):
                    sindex[i] = 0
                    result[i] = 1

        return result

    # Copy the DataFrame explicitly to avoid modification of a slice.
    df = df.copy()

    # Existing computations
    b_values = np.linspace(0.1, 0.95, 18)
    conjectures = [approximation(df['open'].values, b) for b in b_values]
    df.loc[:, 'tr'] = np.maximum.reduce([df['high'] - df['low'],
                                         (df['high'] - nz(df['close'].shift(1))).abs(),
                                         (df['low'] - nz(df['close'].shift(1))).abs()])
    inapproximability_terms = [approximation(df['tr'].values, b) for b in b_values]
    df.loc[:, 'inapproximability'] = np.mean(inapproximability_terms, axis=0)
    df.loc[:, 'amlag'] = np.mean(conjectures, axis=0)
    df.loc[:, 'upper_threshold_2'] = df['amlag'] + 2 * df['inapproximability'] * 1.618
    df.loc[:, 'lower_threshold_2'] = df['amlag'] - 2 * df['inapproximability'] * 1.618
    df.loc[:, 'sell_strong'] = ((df['high'] < df['upper_threshold_2'].shift(1)) &
                            (df['high'].shift(1) >= df['upper_threshold_2'].shift(1))).astype(int)
    df.loc[:, 'buy_strong'] = ((df['low'] > df['lower_threshold_2'].shift(1)) &
                            (df['low'].shift(1) <= df['lower_threshold_2'].shift(1))).astype(int)

    # Initialize signal columns with None
    df.loc[:, 'buy'] = None
    df.loc[:, 'sell'] = None

    # Only process if we have enough data
    if len(df) >= max(5, signal_strength):
        # Calculate signals for the last row
        major = lele(value_one, signal_strength)

        # Get indices of buy and sell signals
        buy_ind = get_major_indices(major, 1)
        sell_ind = get_major_indices(major, -1)

        # Assign 'buy' and 'sell' signals using positional indexing with .iloc
        df.iloc[buy_ind, df.columns.get_loc('buy')] = df.iloc[buy_ind, df.columns.get_loc('close')]
        df.iloc[sell_ind, df.columns.get_loc('sell')] = df.iloc[sell_ind, df.columns.get_loc('close')]

    return df


def volume_flow_indicator(df, length=130, coef=0.2, vcoef=2.5,
                     signal_length=5, smooth_vfi=False):
    """
    Calculate Volume Flow Indicator (VFI) from a DataFrame with NaN handling

    Parameters:
    df: DataFrame with columns 'high', 'low', 'close', 'volume'
    length: lookback period (default 130)
    coef: coefficient (default 0.2)
    vcoef: max volume cutoff coefficient (default 2.5)
    signal_length: signal line period (default 5)
    smooth_vfi: whether to smooth VFI with SMA (default False)

    Returns:
    DataFrame: with columns 'vfi', 'vfima', 'd'
    """
    # Create a copy to avoid modifying input
    df = df.copy()

    # Fill any NaN values in input with forward fill
    df[['high', 'low', 'close', 'volume']] = df[['high', 'low', 'close', 'volume']].fillna(method='ffill')

    # Calculate typical price (HLC3)
    typical = (df['high'] + df['low'] + df['close']) / 3

    # Calculate inter (log difference)
    inter = np.log(typical) - np.log(typical.shift(1))
    inter = inter.fillna(0)  # Fill initial NaN

    # Calculate 30-period standard deviation
    vinter = inter.rolling(window=30, min_periods=1).std().fillna(0)

    # Calculate cutoff
    cutoff = coef * vinter * df['close']

    # Calculate volume average and max
    vave = df['volume'].rolling(window=length, min_periods=1).mean().shift(1).fillna(0)
    vmax = vave * vcoef

    # Volume cutoff
    vc = np.where(df['volume'] < vmax, df['volume'], vmax)

    # Money flow
    mf = typical - typical.shift(1)
    mf = mf.fillna(0)  # Fill initial NaN

    # Volume contribution
    vcp = np.where(mf > cutoff, vc,
                   np.where(mf < -cutoff, -vc, 0))
    vcp = pd.Series(vcp, index=df.index).fillna(0)  # Ensure no NaN in vcp

    # Calculate VFI
    vfi_raw = pd.Series(vcp).rolling(window=length, min_periods=1).sum() / vave
    vfi_raw = vfi_raw.replace([np.inf, -np.inf], 0).fillna(0)  # Handle division by zero

    if smooth_vfi:
        vfi = vfi_raw.rolling(window=3, min_periods=1).mean()
    else:
        vfi = vfi_raw

    # Calculate EMA of VFI
    vfima = vfi.ewm(span=signal_length, adjust=False).mean()

    # Calculate difference
    d = vfi - vfima

    # Create result DataFrame
    result_df = pd.DataFrame({
        'vfi': vfi,
        'vfima': vfima,
        'd': d
    }, index=df.index)

    # Final NaN cleanup
    result_df = result_df.fillna(0)

    return result_df



def visualize_vfi(vfi_df, output_file="vfi_visualization.png"):
    """
    Visualize the computed Volume Flow Indicator (VFI) and its components.

    Parameters:
    vfi_df (pd.DataFrame): DataFrame containing 'vfi', 'vfima', and 'd' columns.
    output_file (str): File path to save the visualization (default: "vfi_visualization.png").
    """
    # Ensure required columns exist in the input DataFrame
    required_columns = ['vfi', 'vfima', 'd']
    if not all(col in vfi_df.columns for col in required_columns):
        raise ValueError("The provided DataFrame must contain 'vfi', 'vfima', and 'd' columns.")

    # Set up the figure
    plt.figure(figsize=(14, 8))

    # Check if the index is datetime-like, otherwise use numerical indices
    if vfi_df.index.dtype.kind == 'M':  # If the index is datetime64
        x_axis = vfi_df.index
    else:
        x_axis = range(len(vfi_df))  # Use numerical indices for the x-axis

    # Plot the VFI and its signal line
    plt.plot(x_axis, vfi_df['vfi'], label="VFI (Volume Flow Indicator)", color="blue", alpha=0.7)
    plt.plot(x_axis, vfi_df['vfima'], label="VFI Signal Line (EMA)", color="orange", linestyle="--", alpha=0.9)

    # Plot the difference bands (d)
    plt.fill_between(x_axis, vfi_df['d'], 0, where=(vfi_df['d'] > 0),
                     color='green', alpha=0.4, interpolate=True, label="VFI > EMA (Positive)")
    plt.fill_between(x_axis, vfi_df['d'], 0, where=(vfi_df['d'] <= 0),
                     color='red', alpha=0.4, interpolate=True, label="VFI <= EMA (Negative)")

    # Add title, labels, grid, and legend
    plt.grid(alpha=0.3)
    plt.title("Volume Flow Indicator (VFI) Visualization", fontsize=16)
    plt.xlabel("Index" if vfi_df.index.dtype.kind != 'M' else "Date/Time", fontsize=12)
    plt.ylabel("VFI Value", fontsize=12)
    plt.legend(loc="best", fontsize=10)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()



def visualize_results(df, output_file="output_plot.png"):
    # Plot the price (high and low), amlag (center line), and thresholds
    plt.figure(figsize=(12, 8))

    # Plot high and low prices
    plt.plot(df.index, df['high'], label='High Price', color='blue', alpha=0.5)
    plt.plot(df.index, df['low'], label='Low Price', color='orange', alpha=0.5)

    # Plot computed values
    plt.plot(df.index, df['amlag'], label='Amlag (Center Line)', color='green', linestyle='--', linewidth=1.5)
    plt.plot(df.index, df['upper_threshold_2'], label='Upper Threshold', color='red', linestyle=':')
    plt.plot(df.index, df['lower_threshold_2'], label='Lower Threshold', color='magenta', linestyle=':')

    # Scatter crossup and crossdn signals
    plt.scatter(
        df.index[df['buy_strong'] == 1],
        df['low'][df['buy_strong'] == 1],
        label='Strong Buy Signal', color='green', marker='^', s=50
    )
    plt.scatter(
        df.index[df['sell_strong'] == 1],
        df['high'][df['sell_strong'] == 1],
        label='Strong Sell Signal', color='red', marker='v', s=50
    )

    # Scatter buy and sell signals from lele
    plt.scatter(
        df.index[~df['buy'].isna()],
        df['buy'][~df['buy'].isna()],
        label='Buy Signal', color='blue', marker='o', s=70
    )
    plt.scatter(
        df.index[~df['sell'].isna()],
        df['sell'][~df['sell'].isna()],
        label='Sell Signal', color='purple', marker='x', s=70
    )

    # Add grid, legend, and labels
    plt.grid(alpha=0.3)
    plt.title('Computed Thresholds, Signals, and Price Movements', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(output_file, dpi=300)  # Change dpi for higher resolution if needed

    # Show the plot (optional)
    plt.show()


if __name__ == "__main__":
    sample_data = pd.read_csv("/home/simon/data/my/crypto/klines/BTCUSDT/BTCUSDT_1d.csv")
    last_1000_data = sample_data.tail(1000)  # Get the last 1000 rows
    # computed_df = tc_top_bottom_finder(last_1000_data)
    # visualize_results(computed_df)
    vfi_data = volume_flow_indicator(last_1000_data)
    visualize_vfi(vfi_data)