import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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


def tc_top_bottom_finder(df, value_one=2, signal_strength=20):
    def find_indices(_list_to_check, _item_to_find):
        _indices = []
        for _idx, _value in enumerate(_list_to_check):
            if _value == _item_to_find:
                _indices.append(_idx)
        return _indices

    def get_major_indices(_data, _p):
        #  the returned indices are precisely at the time of a buy/sell signal
        return find_indices(_data, _p)  # 1 BUY / -1 SELL

    def trend_reversal_signals(threshold_value, strength):
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
        major = trend_reversal_signals(value_one, signal_strength)

        # Get indices of buy and sell signals
        buy_ind = get_major_indices(major, 1)
        sell_ind = get_major_indices(major, -1)

        # Assign 'buy' and 'sell' signals using positional indexing with .iloc
        df.iloc[buy_ind, df.columns.get_loc('buy')] = df.iloc[buy_ind, df.columns.get_loc('close')]
        df.iloc[sell_ind, df.columns.get_loc('sell')] = df.iloc[sell_ind, df.columns.get_loc('close')]

    return df


def generate_signal_label(df):
    """
    Generate labels for trading signals based on the output of tc_top_bottom_finder.

    Parameters:
    df: DataFrame which includes columns ['buy', 'sell', 'buy_strong', 'sell_strong'].

    Returns:
    DataFrame with an additional 'signal_label' column containing:
    - 'buy': For buy signals.
    - 'sell': For sell signals.
    - 'buy_strong': For strong buy signals.
    - 'sell_strong': For strong sell signals.
    - None: If no signal.

    """
    # Copy the DataFrame explicitly to avoid modifying the original
    df = df.copy()

    # Initialize the signal column as None
    df['signal_label'] = None

    # Assign labels based on conditions
    df.loc[df['buy_strong'] == 1, 'signal_label'] = 'buy_strong'
    df.loc[df['sell_strong'] == 1, 'signal_label'] = 'sell_strong'
    df.loc[df['buy'].notna(), 'signal_label'] = 'buy'
    df.loc[df['sell'].notna(), 'signal_label'] = 'sell'

    return df


def one_hot_encode_column(df, column_name, drop_original=True):
    """
    One-hot encode a specified column in a DataFrame and optionally remove the original column.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame containing the column to be one-hot encoded.
    - column_name: str
        The name of the column to one-hot encode.
    - drop_original: bool, default True
        If True, the original column will be removed from the DataFrame.

    Returns:
    - pandas.DataFrame
        A modified DataFrame with the one-hot encoded columns added (and original column optionally removed).
    """
    # Ensure the column is treated as a string
    df[column_name] = df[column_name].astype(str)  # Convert to string if not already

    # Initialize the OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Perform one-hot encoding
    one_hot_encoded_array = one_hot_encoder.fit_transform(df[[column_name]])

    # Create a DataFrame for the one-hot encoded columns
    onehot_df = pd.DataFrame(one_hot_encoded_array,
                             columns=one_hot_encoder.get_feature_names_out([column_name]),
                             index=df.index)  # Ensure indices match the original DataFrame

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    df = pd.concat([df, onehot_df], axis=1)

    # Drop the original column if specified
    if drop_original:
        df = df.drop(columns=[column_name])

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
    df[['high', 'low', 'close', 'volume']] = df[['high', 'low', 'close', 'volume']].ffill()
    typical = (df['high'] + df['low'] + df['close']) / 3
    inter = np.log(typical) - np.log(typical.shift(1))
    inter = inter.fillna(0)  # Fill initial NaN
    vinter = inter.rolling(window=30, min_periods=1).std().fillna(0)
    cutoff = coef * vinter * df['close']
    vave = df['volume'].rolling(window=length, min_periods=1).mean().shift(1).fillna(0)
    vmax = vave * vcoef
    vc = np.where(df['volume'] < vmax, df['volume'], vmax)
    mf = typical - typical.shift(1)
    mf = mf.fillna(0)  # Fill initial NaN
    vcp = np.where(mf > cutoff, vc,
                   np.where(mf < -cutoff, -vc, 0))
    vcp = pd.Series(vcp, index=df.index).fillna(0)  # Ensure no NaN in vcp
    vfi_raw = pd.Series(vcp).rolling(window=length, min_periods=1).sum() / vave
    vfi_raw = vfi_raw.replace([np.inf, -np.inf], 0).fillna(0)  # Handle division by zero

    if smooth_vfi:
        vfi = vfi_raw.rolling(window=3, min_periods=1).mean()
    else:
        vfi = vfi_raw

    vfima = vfi.ewm(span=signal_length, adjust=False).mean()
    d = vfi - vfima
    result_df = pd.DataFrame({
        'vfi': vfi,
        'vfima': vfima,
        'd': d
    }, index=df.index)
    # Final NaN cleanup
    result_df = result_df.fillna(0)

    return result_df


def ehlers_smoothed_adaptive_momentum(df, source='hl2', alpha=0.07, cutoff=8.0):
    """
    Calculate Ehlers Smoothed Adaptive Momentum from DataFrame

    Parameters:
    df: DataFrame with 'high', 'low', 'close' columns
    source: 'hl2' or other price source (default 'hl2')
    alpha: alpha parameter (default 0.07)
    cutoff: cutoff parameter (default 8.0)

    Returns:
    DataFrame with 'f3' column containing the final indicator
    """
    df = df.copy()

    # Calculate source
    if source == 'hl2':
        src = (df['high'] + df['low']) / 2
    else:
        src = df['close']

    # Constants
    pi = 4 * np.arctan(1.0)
    dtr = pi / 180.0

    # Calculate s (weighted moving average)
    s = (src + 2 * src.shift(1).fillna(src) +
         2 * src.shift(2).fillna(src) +
         src.shift(3).fillna(src)) / 6.0

    # Calculate c (second order filter)
    c = pd.Series(np.zeros(len(df)), index=df.index)
    c.iloc[0] = 0
    c.iloc[1] = (src.iloc[1] - 2 * src.iloc[0] + src.iloc[0]) / 4.0 if len(df) > 1 else 0

    for i in range(2, len(df)):
        c.iloc[i] = ((1 - 0.5 * alpha) ** 2 *
                     (s.iloc[i] - 2 * s.iloc[i - 1] + s.iloc[i - 2]) +
                     2 * (1 - alpha) * c.iloc[i - 1] -
                     (1 - alpha) ** 2 * c.iloc[i - 2])

    # Calculate q1 and I1
    ip = pd.Series(np.zeros(len(df)), index=df.index)
    q1 = (0.0962 * c + 0.5769 * c.shift(2).fillna(0) -
          0.5769 * c.shift(4).fillna(0) - 0.0962 * c.shift(6).fillna(0))
    q1 = q1 * (0.5 + 0.08 * ip.shift(1))
    I1 = c.shift(3).fillna(0)

    # Calculate dp
    dp_ = pd.Series(np.zeros(len(df)), index=df.index)
    mask = (q1 != 0) & (q1.shift(1) != 0)
    dp_.loc[mask] = ((I1 / q1 - I1.shift(1) / q1.shift(1)) /
                     (1 + I1 * I1.shift(1) / (q1 * q1.shift(1))))[mask]
    dp = np.where(dp_ < 0.1, 0.1, np.where(dp_ > 1.1, 1.1, dp_))

    # Median filter
    md = pd.concat([
        pd.Series(dp, index=df.index),
        pd.Series(dp, index=df.index).shift(1),
        pd.concat([
            pd.Series(dp, index=df.index).shift(2),
            pd.Series(dp, index=df.index).shift(3),
            pd.Series(dp, index=df.index).shift(4)
        ], axis=1).median(axis=1)
    ], axis=1).median(axis=1).fillna(0)

    # Calculate periods
    dc = np.where(md == 0, 15, 2 * pi / md + 0.5)
    ip = 0.33 * dc + 0.67 * ip.shift(1).fillna(0)
    # Initialize p before using it
    p = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(len(df)):
        p.iloc[i] = 0.15 * ip.iloc[i] + 0.85 * (p.iloc[i - 1] if i > 0 else 0)
    pr = np.round(np.abs(p - 1)).astype(int)

    # Calculate velocity
    v1 = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(len(df)):
        period = min(pr.iloc[i], 75)
        if i >= period:
            v1.iloc[i] = src.iloc[i] - src.iloc[i - period]

    # Final filter coefficients
    a1 = np.exp(-pi / cutoff)
    b1 = 2.0 * a1 * np.cos((1.738 * 180 / cutoff) * dtr)
    c1 = a1 * a1
    coef2 = b1 + c1
    coef3 = -(c1 + b1 * c1)
    coef4 = c1 * c1
    coef1 = 1 - coef2 - coef3 - coef4

    # Calculate f3
    f3 = pd.Series(np.zeros(len(df)), index=df.index)
    f3.iloc[:3] = v1.iloc[:3]
    for i in range(3, len(df)):
        f3.iloc[i] = (coef1 * v1.iloc[i] +
                      coef2 * f3.iloc[i - 1] +
                      coef3 * f3.iloc[i - 2] +
                      coef4 * f3.iloc[i - 3])
    df['f3'] = f3

    return df


def cycle_oscillator(
        df,
        scl_t=10,
        mcl_t=30,
        scm=1.0,
        mcm=3.0
):
    """
    Calculate Cycle Oscillator values using pandas DataFrame.

    Parameters:
    df: DataFrame with 'high', 'low', and 'close' columns
    scl_t: Short Cycle Length (default 10)
    mcl_t: Medium Cycle Length (default 30)
    scm: Short Cycle Multiplier (default 1.0)
    mcm: Medium Cycle Multiplier (default 3.0)

    Returns:
    DataFrame with additional columns: 'omed' and 'oshort'.
    """
    df = df.copy()

    # Required columns check
    if not {'close', 'high', 'low'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'close', 'high', and 'low' columns.")

    # Calculations
    scl = scl_t / 2.0
    mcl = mcl_t / 2.0

    # Running Moving Average (RMA equivalent to EMA style in Pine Script)
    ma_scl = df['close'].ewm(span=scl, adjust=False).mean()
    ma_mcl = df['close'].ewm(span=mcl, adjust=False).mean()

    # ATR Calculation
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_scl = tr.ewm(span=scl, adjust=False).mean()
    atr_mcl = tr.ewm(span=mcl, adjust=False).mean()

    # Offsets
    scm_off = scm * atr_scl
    mcm_off = mcm * atr_mcl

    # Cycle shifts (half of the cycle lengths)
    scl_2 = int(scl / 2)
    mcl_2 = int(mcl / 2)

    # Short and Medium Cycle Tops and Bottoms
    sct = ma_scl.shift(scl_2).fillna(df['close']) + scm_off
    scb = ma_scl.shift(scl_2).fillna(df['close']) - scm_off
    mct = ma_mcl.shift(mcl_2).fillna(df['close']) + mcm_off
    mcb = ma_mcl.shift(mcl_2).fillna(df['close']) - mcm_off

    # Average of short cycle top and bottom
    scmm = (sct + scb) / 2

    # Oscillators
    omed = (scmm - mcb) / (mct - mcb)
    oshort = (df['close'] - mcb) / (mct - mcb)

    # Handle division by zero or infinite values
    omed = omed.replace([np.inf, -np.inf], 0).fillna(0)
    oshort = oshort.replace([np.inf, -np.inf], 0).fillna(0)

    # Overbought/Oversold Conditions
    omed_ob = omed.where(omed >= 1.0, np.nan)  # Medium Cycle Overbought
    omed_os = omed.where(omed <= 0.0, np.nan)  # Medium Cycle Oversold
    oshort_ob = oshort.where(oshort >= 1.0, np.nan)  # Short Cycle Overbought
    oshort_os = oshort.where(oshort <= 0.0, np.nan)  # Short Cycle Oversold

    # Add Columns to DataFrame
    df['omed'] = omed
    df['oshort'] = oshort

    df['omed_ob'] = omed_ob.notna().astype(int)
    df['omed_os'] = omed_os.notna().astype(int)
    df['oshort_ob'] = oshort_ob.notna().astype(int)
    df['oshort_os'] = oshort_os.notna().astype(int)

    return df


def visualize_cycle_oscillator(df, save_path="cycle_oscillator.png"):
    """
    Visualize the Cycle Oscillator (omed, oshort) along with Overbought (OB) and Oversold (OS) conditions.

    Parameters:
    df: DataFrame that includes columns 'omed', 'oshort', 'omed_ob', 'omed_os', 'oshort_ob', 'oshort_os'.
    save_path: Path to save the plot as a PNG file.

    Returns:
    None. The plot is saved as a file.
    """
    # Check for required columns
    required_columns = ['omed', 'oshort', 'omed_ob', 'omed_os', 'oshort_ob', 'oshort_os']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Plot omed and oshort lines
    plt.plot(df.index, df['omed'], label='Omed (Medium Cycle Oscillator)', color='blue', linewidth=1.5)
    plt.plot(df.index, df['oshort'], label='Oshort (Short Cycle Oscillator)', color='orange', linewidth=1.5)

    # Plot histograms for omed overbought (OB) and oversold (OS) conditions
    plt.bar(df.index, df['omed_ob'], width=1, color='purple', alpha=0.5, label='Omed OB (Medium Overbought)')
    plt.bar(df.index, df['omed_os'], width=1, color='purple', alpha=0.5, label='Omed OS (Medium Oversold)')

    # Plot histograms for oshort overbought (OB) and oversold (OS) conditions
    plt.bar(df.index, df['oshort_ob'], width=1, color='green', alpha=0.5, label='Oshort OB (Short Overbought)')
    plt.bar(df.index, df['oshort_os'], width=1, color='red', alpha=0.5, label='Oshort OS (Short Oversold)')

    # Add grid, title, and legend
    plt.grid(alpha=0.3)
    plt.title("Cycle Oscillator with Overbought and Oversold Conditions", fontsize=16)
    plt.xlabel("Index (or Timestamp)", fontsize=12)
    plt.ylabel("Oscillator Values", fontsize=12)
    plt.legend(fontsize=12)

    # Handle x-axis dates or indices
    if isinstance(df.index, pd.DatetimeIndex):  # If x-axis is datetime, format it
        plt.gcf().autofmt_xdate()

    # Tight layout to ensure elements fit well
    plt.tight_layout()

    # Save the plot as PNG
    plt.savefig(save_path)
    plt.show()


def visualize_ehlers_computation(df, save_path="ehlers_computation.png"):
    """
    Visualize Ehlers Smoothed Adaptive Momentum computation values and save to a PNG.

    Parameters:
    df (DataFrame): Input DataFrame containing 'f3' and other computation columns, including 'timestamp'.
    save_path (str): File path to save the plot as PNG (default is 'ehlers_computation.png').

    Returns:
    None: Saves the plot to the specified file.
    """
    # Convert 'timestamp' to datetime if it's not already datetime
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure 'timestamp' is used as the index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    plt.figure(figsize=(14, 8))
    plt.grid(alpha=0.4)

    # Plot second order filter (c)
    if 'c' in df.columns:
        plt.plot(df.index, df['c'], label='Second Order Filter (c)', color='green', alpha=0.7, linewidth=1)
    else:
        print("Column 'c' not found, skipping second order filter plot.")

    # Plot q1
    if 'q1' in df.columns:
        plt.plot(df.index, df['q1'], label='q1', color='purple', alpha=0.7, linewidth=1)
    else:
        print("Column 'q1' not found, skipping q1 plot.")

    # Plot final indicator (f3)
    if 'f3' in df.columns:
        plt.plot(df.index, df['f3'], label='Final Ehlers f3', color='red', alpha=0.8, linewidth=1.5)
    else:
        print("Column 'f3' not found, skipping final indicator plot.")

    # Format X-axis: use dates from the 'timestamp' index
    if isinstance(df.index, pd.DatetimeIndex):
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set the major ticks based on dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format major ticks as 'YYYY-MM-DD'
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())  # Add minor ticks for each month
        plt.gcf().autofmt_xdate()  # Auto-format dates to prevent overlap

    # Add title and legend
    plt.title("Ehlers Smoothed Adaptive Momentum Computation", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=12)  # Automatically handles labels from plt.plot()

    # Save to PNG
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


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


def visualize_find_tb_results(df, output_file="output_plot.png"):
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
    # vfi_data = volume_flow_indicator(last_1000_data)
    # visualize_vfi(vfi_data)
    # ehlers_data = ehlers_smoothed_adaptive_momentum(last_1000_data)
    # visualize_ehlers_computation(ehlers_data)
    cycle_oscillator_data = cycle_oscillator(last_1000_data)
    visualize_cycle_oscillator(cycle_oscillator_data)
