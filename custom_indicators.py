import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def tc_top_bottom_finder(df):
    def nz(series, default=0):
        return series.ffill().fillna(default)

    def approximation(a, b):
        l0 = np.zeros(len(a))
        l1 = np.zeros(len(a))
        l2 = np.zeros(len(a))
        l3 = np.zeros(len(a))
        for i in range(1, len(a)):
            l0[i] = (1 - b) * a[i] + b * nz(pd.Series(l0))[i-1]
            l1[i] = -b * l0[i] + nz(pd.Series(l0))[i-1] + b * nz(pd.Series(l1))[i-1]
            l2[i] = -b * l1[i] + nz(pd.Series(l1))[i-1] + b * nz(pd.Series(l2))[i-1]
            l3[i] = -b * l2[i] + nz(pd.Series(l2))[i-1] + b * nz(pd.Series(l3))[i-1]
        return (l0 + 2*l1 + 2*l2 + l3) / 6

    b_values = np.linspace(0.1, 0.95, 18)
    conjectures = [approximation(df['open'].values, b) for b in b_values]
    df['tr'] = np.maximum.reduce([df['high'] - df['low'],
                                  (df['high'] - nz(df['close'].shift(1))).abs(),
                                  (df['low'] - nz(df['close'].shift(1))).abs()])
    inapproximability_terms = [approximation(df['tr'].values, b) for b in b_values]
    df['inapproximability'] = np.mean(inapproximability_terms, axis=0)
    df['amlag'] = np.mean(conjectures, axis=0)
    df['upper_threshold_2'] = df['amlag'] + 2 * df['inapproximability'] * 1.618
    df['lower_threshold_2'] = df['amlag'] - 2 * df['inapproximability'] * 1.618
    df['crossdn'] = ((df['high'] < df['upper_threshold_2'].shift(1)) &
                     (df['high'].shift(1) >= df['upper_threshold_2'].shift(1))).astype(int)
    df['crossup'] = ((df['low'] > df['lower_threshold_2'].shift(1)) &
                     (df['low'].shift(1) <= df['lower_threshold_2'].shift(1))).astype(int)
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

    # Scatter crossup and crossdn
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

    # Add grid, legend, and labels
    plt.grid(alpha=0.3)
    plt.title('Computed Thresholds and Price Movements', fontsize=14)
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
    computed_df = tc_top_bottom_finder(sample_data)
    visualize_results(computed_df)