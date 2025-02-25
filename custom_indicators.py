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

    def lele(_val, _strength):
        n = len(df)
        _bindex = np.zeros(n)
        _sindex = np.zeros(n)
        _ret = np.zeros(n)

        for i in range(n):
            # Maintain previous indices for _bindex and _sindex
            if i > 0:
                _bindex[i] = _bindex[i - 1]
                _sindex[i] = _sindex[i - 1]

            # Update _bindex and _sindex based on the close condition
            if i >= 4:  # Validate the 4-period condition
                if df['close'][i] > df['close'][i - 4]:
                    _bindex[i] += 1
                elif df['close'][i] < df['close'][i - 4]:
                    _sindex[i] += 1

            # Check for conditions to reset indices and set return values
            if i >= _strength:
                max_high = np.max(df['high'][i - _strength:i])
                min_low = np.min(df['low'][i - _strength:i])

                if (_bindex[i] > _val
                        and df['close'][i] < df['open'][i]
                        and df['high'][i] >= max_high):
                    _bindex[i] = 0
                    _ret[i] = -1

                if (_sindex[i] > _val
                        and df['close'][i] > df['open'][i]
                        and df['low'][i] <= min_low):
                    _sindex[i] = 0
                    _ret[i] = 1

        return _ret

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
    df.loc[:, 'crossdn'] = ((df['high'] < df['upper_threshold_2'].shift(1)) &
                            (df['high'].shift(1) >= df['upper_threshold_2'].shift(1))).astype(int)
    df.loc[:, 'crossup'] = ((df['low'] > df['lower_threshold_2'].shift(1)) &
                            (df['low'].shift(1) <= df['lower_threshold_2'].shift(1))).astype(int)

    # Initialize signal columns with None
    df.loc[:, 'buy'] = None
    df.loc[:, 'sell'] = None

    # Only process if we have enough data
    if len(df) >= max(5, signal_strength):
        # Calculate signals for the last row
        major = lele(value_one, signal_strength)

        # Get indices of buy and sell signals
        _buy_ind = get_major_indices(major, 1)
        _sell_ind = get_major_indices(major, -1)

        # Assign current price (df['close']) to 'buy' and 'sell' signal columns
        df.loc[_buy_ind, 'buy'] = df.loc[_buy_ind, 'close']
        df.loc[_sell_ind, 'sell'] = df.loc[_sell_ind, 'close']

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
        df.index[df['crossup'] == 1],
        df['low'][df['crossup'] == 1],
        label='Crossup (Buy Signal)', color='green', marker='^', s=50
    )
    plt.scatter(
        df.index[df['crossdn'] == 1],
        df['high'][df['crossdn'] == 1],
        label='Crossdn (Sell Signal)', color='red', marker='v', s=50
    )

    # Scatter buy and sell signals from lele
    plt.scatter(
        df.index[~df['buy'].isna()],
        df['buy'][~df['buy'].isna()],
        label='Buy Signal (Bullish Reversal)', color='blue', marker='o', s=70
    )
    plt.scatter(
        df.index[~df['sell'].isna()],
        df['sell'][~df['sell'].isna()],
        label='Sell Signal (Bearish Reversal)', color='purple', marker='x', s=70
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