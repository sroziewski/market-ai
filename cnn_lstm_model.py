from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bar support

from custom_indicators import cycle_oscillator, tc_top_bottom_finder, volume_flow_indicator, generate_signal_label, \
    one_hot_encode_column

features = ['open', 'high', 'low', 'volume', 'close', 'macd', 'macd_signal', 'macd_histogram', 'rsi', 'stoch_rsi_k',
            'stoch_rsi_d', 'open_sma200_diff_pct', 'high_sma200_diff_pct', 'low_sma200_diff_pct', 'close_sma200_diff_pct',
            'open_sma100_diff_pct', 'high_sma100_diff_pct', 'low_sma100_diff_pct', 'close_sma100_diff_pct',
            'open_sma50_diff_pct', 'high_sma50_diff_pct', 'low_sma50_diff_pct', 'close_sma50_diff_pct',
            'open_sma20_diff_pct', 'high_sma20_diff_pct', 'low_sma20_diff_pct', 'close_sma20_diff_pct',
            'sma20_sma50_angle', 'sma20_sma100_angle', 'sma20_sma200_angle', 'sma50_sma100_angle',
            'sma50_sma200_angle', 'sma100_sma200_angle', 'omed', 'oshort', 'omed_ob', 'omed_os', 'oshort_ob',
            'oshort_os', 'signal_label', 'vfi', 'vfima', 'd']
lookback = 130  # as for the volume_flow_indicator computation


def process_batch(args):
    """
    Process a batch of rows to create labels for the given range.

    Arguments:
        args (tuple): A tuple containing:
            - row_range (range): Indices to process
            - low_prices (np.ndarray): Array of low prices
            - high_prices (np.ndarray): Array of high prices
            - close_prices (np.ndarray): Array of close prices
            - total_rows (int): Total number of rows in the dataset

    Returns:
        list: A list of calculated labels for the rows in this batch
    """
    row_range, low_prices, high_prices, close_prices, total_rows = args
    local_y = []
    for i in row_range:
        window_20 = slice(i, min(i + 20, total_rows))
        window_50 = slice(i, min(i + 50, total_rows))
        local_y.append([
            np.min(low_prices[window_20]),
            np.max(high_prices[window_20]),
            np.mean(close_prices[window_20]),
            np.min(low_prices[window_50]),
            np.max(high_prices[window_50]),
            np.mean(close_prices[window_50])
        ])
    return local_y


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


# Custom Dataset for Price Data
class PriceDataset(Dataset):
    """Custom Dataset for price data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# HybridPriceRegressor Model Definition
class HybridPriceRegressor(nn.Module):
    def __init__(self, lookback_period=lookback, input_features=4, cnn_filters=32,
                 lstm_units=64, dropout_rate=0.3, attention_heads=4):
        super(HybridPriceRegressor, self).__init__()
        self.lookback_period = lookback_period
        self.input_features = input_features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.attention_heads = attention_heads
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Cache for labels
        self.cached_labels = None

        # CNN Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=cnn_filters,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # LSTM Temporal Processing
        self.lstm1 = nn.LSTM(cnn_filters * 2, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)

        # Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=lstm_units // 2,
                                               num_heads=attention_heads,
                                               dropout=dropout_rate,
                                               batch_first=True)

        # Fully Connected Layers for Output
        self.fc1 = nn.Linear(lstm_units // 2, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 6)  # 6 output dimensions

    def forward(self, x):
        # CNN Feature Extraction
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # LSTM Processing
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)  # Shape: (batch, seq_len, lstm_units // 2)

        # Attention Mechanism
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        x = attn_output.mean(dim=1)  # Mean across sequence length: (batch, lstm_units // 2)

        # Fully Connected Layers
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # (batch, 6)
        return x

    def prepare_data(self, klines_df):
        klines_df = calculate_indicators(klines_df)
        data = klines_df[features].values
        data_scaled = self.scaler_X.fit_transform(data)
        X = [data_scaled[i - self.lookback_period:i]
             for i in range(self.lookback_period, len(data))]
        return np.array(X), self.scaler_X

    def create_labels(self, klines_df, save_path=None):
        if self.cached_labels is not None:
            return self.cached_labels

        close_prices = klines_df['close'].values
        high_prices = klines_df['high'].values
        low_prices = klines_df['low'].values
        total_rows = len(close_prices)

        indices = range(self.lookback_period, total_rows)
        num_cores = cpu_count()
        chunk_size = len(indices) // num_cores
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

        args = [(chunk, low_prices, high_prices, close_prices, total_rows)
                for chunk in chunks]

        with Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(process_batch, args),
                                total=len(chunks), desc="Creating Labels"))

        y = [label for batch in results for label in batch]
        y_scaled = self.scaler_y.fit_transform(np.array(y))
        self.cached_labels = y_scaled

        if save_path:
            np.savez_compressed(save_path, y_scaled)
            print(f"Labels saved to {save_path}")

        return y_scaled

    def train_model(self, klines_df, epochs=200, batch_size=128, validation_split=0.2,
                    patience=10, device='cuda', save_path="hybrid_price_regressor.pth"):
        self.to(device)

        X, _ = self.prepare_data(klines_df)
        y = self.create_labels(klines_df)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = PriceDataset(X_train, y_train)
        val_dataset = PriceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                         patience=5, min_lr=1e-6)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for X_batch, y_batch in pbar:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    y_pred = self(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    pbar.set_postfix({"Train Loss": loss.item()})

            train_loss /= len(train_loader)

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = self(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
                torch.save(best_model_state, save_path)
                print(f"Model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        if best_model_state:
            self.load_state_dict(best_model_state)
            print("Restoring best model state.")

        final_model_path = "final_" + save_path
        torch.save(self.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

    def predict(self, klines_df, device='cuda'):
        self.to(device)
        self.eval()
        X, _ = self.prepare_data(klines_df)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions_scaled = self(X_tensor).cpu().numpy()
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions.tolist()

    def evaluate(self, klines_df, device='cuda'):
        self.to(device)
        self.eval()
        X, _ = self.prepare_data(klines_df)
        y = self.cached_labels if self.cached_labels is not None else self.create_labels(klines_df)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = self(X_tensor)
            mse = nn.MSELoss()(y_pred, y_tensor).item()
            mae = torch.mean(torch.abs(y_pred - y_tensor)).item()
        return [mse, mae]


# Main Entry Point
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    sample_data = pd.read_csv("/raid/sroziewski/data/crypto/klines/BTCUSDT/BTCUSDT_15m.csv")

    train_size = int(0.8 * len(sample_data))
    train_df = sample_data[:train_size]
    test_df = sample_data[train_size:]

    regressor = HybridPriceRegressor(lookback_period=50, input_features=len(features))
    regressor.train_model(train_df, batch_size=64, validation_split=0.2, patience=10)

    predictions = regressor.predict(test_df)
    print("Sample predictions (first 5):")
    for i, pred in enumerate(predictions[:500]):
        print(f"Prediction {i + 1}: {pred}")

    loss, mae = regressor.evaluate(test_df)
    print(f"Evaluation Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
