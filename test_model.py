import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bar support

load_dotenv()

features = ['open', 'high', 'low', 'close', 'volume']


def process_batch(args):
    """
    Processes a batch of data based on specific window sizes and extracts statistical
    metrics like the minimum and maximum of specified price ranges. The function
    utilizes sliding windows of 5, 10, 20, and 50 rows to compute these metrics for
    each row in the given range, making it suitable for financial data analysis or
    similar use cases.

    :param args: A tuple containing the row range, low prices, high prices, close
        prices, and the total number of rows. The parameters within the tuple
        must adhere to the following order:
        - row_range (range): Range object indicating the rows to include.
        - low_prices (numpy.ndarray): Array of low price values.
        - high_prices (numpy.ndarray): Array of high price values.
        - close_prices (numpy.ndarray): Array of close price values.
        - total_rows (int): Total number of rows in the dataset.

    :return: A list of lists where each inner list contains computed minimum and
        maximum values for the specified sliding window rules.
    :rtype: list
    """
    row_range, low_prices, high_prices, close_prices, total_rows = args
    local_y = []
    for i in row_range:
        window_5 = slice(i, min(i + 5, total_rows))
        window_10 = slice(i, min(i + 10, total_rows))
        window_20 = slice(i, min(i + 20, total_rows))
        window_50 = slice(i, min(i + 50, total_rows))
        local_y.append([
            np.min(low_prices[window_5]),
            np.max(low_prices[window_5]),
            np.min(low_prices[window_10]),
            np.max(high_prices[window_10]),
            np.min(low_prices[window_20]),
            np.max(high_prices[window_20]),
            np.min(low_prices[window_50]),
            np.max(high_prices[window_50]),
        ])
    return local_y


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
    def __init__(self, lookback_period=20, input_features=4, cnn_filters=32, lstm_units=128, dropout_rate=0.3,
                 attention_heads=4, num_outputs=8):
        super(HybridPriceRegressor, self).__init__()
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.lookback_period = lookback_period  # Reduced to 20
        self.input_features = input_features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units  # Now 128
        self.dropout_rate = dropout_rate
        self.attention_heads = attention_heads
        self.num_outputs = num_outputs  # 6 or 9, depending on labels

        # Cache for labels
        self.cached_labels = None

        # CNN Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=cnn_filters,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Less aggressive pooling
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM Temporal Processing (increased to 128 units)
        self.lstm1 = nn.LSTM(cnn_filters * 2, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)  # 128 → 64

        # Attention Layer (embed_dim now 64, since lstm_units // 2 = 128 // 2)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_units // 2,
                                               num_heads=attention_heads,
                                               dropout=dropout_rate,
                                               batch_first=True)

        # Fully Connected Layers for Output
        self.fc1 = nn.Linear(lstm_units // 2, 64)  # 64 → 64
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, num_outputs)  # 6 or 9 outputs

    def forward(self, x):
        # CNN Feature Extraction
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # LSTM Processing
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        x, _ = self.lstm1(x)  # (batch, seq_len, 128)
        x, _ = self.lstm2(x)  # (batch, seq_len, 64)
        # Attention Mechanism
        attn_output, _ = self.attention(x, x, x)  # Self-attention: (batch, seq_len, 64)
        x = attn_output[:, -1, :]  # Last time step: (batch, 64)
        # Fully Connected Layers
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # (batch, num_outputs)
        return x

    def prepare_data(self, klines_df):
        data = klines_df[features].values
        data_scaled = self.scaler_X.fit_transform(data)
        X = [data_scaled[i - self.lookback_period:i] for i in range(self.lookback_period, len(data))]
        return np.array(X), self.scaler_X

    def create_labels(self, klines_df, save_path=None):
        # Check if labels are already cached
        if self.cached_labels is not None:
            return self.cached_labels

        open_prices = klines_df['open'].values
        close_prices = klines_df['close'].values
        high_prices = klines_df['high'].values
        low_prices = klines_df['low'].values
        total_rows = len(close_prices)

        # Create indices for processing
        indices = range(self.lookback_period, total_rows)
        num_cores = cpu_count()  # Number of CPU cores
        chunk_size = len(indices) // num_cores  # Size of chunks distributed (approx evenly)
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

        # Prepare arguments for `process_batch`
        args = [(chunk, low_prices, high_prices, close_prices, total_rows) for chunk in chunks]

        # Use Pool to process in parallel
        with Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(process_batch, args), total=len(chunks), desc="Creating Labels"))

        # Combine all results into a single list
        y = [label for batch in results for label in batch]
        y_scaled = self.scaler_y.fit_transform(np.array(y))  # Scale labels
        #
        # self.cached_labels = y_scaled  # Cache the labels to avoid recomputing
        #
        # # Optionally save the labels to disk for persistent storage
        # if save_path:
        #     np.savez_compressed(save_path, y_scaled)
        #     print(f"Labels saved to {save_path}")

        return y_scaled

    def weighted_mse_loss(self, pred, target):
        """
        Weighted MSE loss prioritizing 20-step outputs over 50-step outputs.
        pred/target shape: (batch, 6) [min_low_20, max_high_20, mean_close_20, min_low_50, max_high_50, mean_close_50]
        """
        mse = nn.MSELoss(reduction='none')
        weights = torch.tensor([1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25],
                               device=pred.device)  # 10-step: 1.0,  20-step: .75, 50-step: 0.5
        loss = mse(pred, target) * weights
        return loss.mean()

    def train_model(self, klines_df, epochs=200, batch_size=64, validation_split=0.2, patience=10, device='cuda',
                    save_path="hybrid_price_regressor2.pth"):
        self.to(device)

        # Prepare data
        X, _ = self.prepare_data(klines_df)
        y = self.create_labels(klines_df)  # Called only once and cached
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = PriceDataset(X_train, y_train)
        val_dataset = PriceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.train()  # Set to training mode
            train_loss = 0
            print(f"Epoch {epoch + 1}")  # Report device

            # Retrieve the current learning rate from the optimizer
            current_lr = optimizer.param_groups[0]['lr']

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for X_batch, y_batch in pbar:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Ensure data on GPU

                    optimizer.zero_grad()
                    y_pred = self(X_batch)
                    loss = self.weighted_mse_loss(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    pbar.set_postfix({"Train Loss": loss.item()})

            train_loss /= len(train_loader)

            # Validation
            self.eval()  # Set to evaluation mode
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = self(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()

            val_loss /= len(val_loader)

            # Print Training and Validation Loss along with current learning rate and patience counter
            print(
                f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Current LR: {current_lr:.6f}, Patience Counter: {patience_counter}/{patience}"
            )

            scheduler.step(val_loss)  # Adjust learning rate

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
                # Save the best model state during training
                torch.save(best_model_state, save_path)
                print(f"Model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Restore the best model state if it was saved
        if best_model_state:
            self.load_state_dict(best_model_state)
            print("Restoring best model state.")

        # Save the final trained model at the end
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
        y = self.create_labels(klines_df)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = self(X_tensor)
            mse = self.weighted_mse_loss(y_pred, y_tensor).item()  # Use weighted loss
            mae = torch.mean(torch.abs(y_pred - y_tensor)).item()  # Keep MAE unweighted
        return [mse, mae]


# Main Entry Point
if __name__ == "__main__":
    device = os.getenv("DEVICE")
    if device is None:
        device = 'cpu'
    print(f"Using device: {device}")
    BASE_DIR = os.getenv("BASE_DIR")
    if BASE_DIR is None:
        raise ValueError("Environment variable 'BASE_DIR' not set")

    # Load data files
    file_path = f"{BASE_DIR}/data/crypto/klines/BTCUSDT/BTCUSDT_15m.csv"
    sample_data = pd.read_csv(file_path)
    file_path = f"{BASE_DIR}/data/crypto/klines/ETHUSDT/ETHUSDT_15m.csv"
    test_data = pd.read_csv(file_path)

    # Initialize the regressor
    regressor = HybridPriceRegressor(lookback_period=50, input_features=len(features))

    # Measure training time
    start_train_time = time.time()  # Record start time
    regressor.train_model(sample_data, epochs=100, batch_size=64, validation_split=0.2, patience=20, device=device,
                          save_path="hybrid_price_regressor_l2_m3.pth")
    end_train_time = time.time()  # Record end time
    print(f"Training completed in: {end_train_time - start_train_time:.2f} seconds")

    # Evaluate the model on the test dataset
    loss, mae = regressor.evaluate(test_data)
    print(f"Test Set Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
