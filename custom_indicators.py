import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    computed_df = tc_top_bottom_finder(last_1000_data)
    visualize_results(computed_df)