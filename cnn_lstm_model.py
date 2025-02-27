import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bar support

from features import features, calculate_indicators, create_labels


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
    def __init__(self, lookback_period=50, input_features=4, cnn_filters=32,
                 lstm_units=64, dropout_rate=0.3, attention_heads=4):
        super(HybridPriceRegressor, self).__init__()
        self.lookback_period = lookback_period
        self.input_features = input_features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.attention_heads = attention_heads

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
        data = klines_df[features].values
        X = [data[i - self.lookback_period:i] for i in range(self.lookback_period, len(data))]
        return np.array(X)

    def train_model(self, train_df, train_labels, epochs=200, batch_size=128, validation_split=0.2,
                    patience=10, device='cuda', save_path="hybrid_price_regressor.pth"):
        self.to(device)

        split_idx = int(len(train_df) * (1 - validation_split))
        X_train, X_val = train_df[:split_idx], train_df[split_idx:]
        y_train, y_val = train_labels[:split_idx], train_labels[split_idx:]

        train_dataset = PriceDataset(X_train, y_train)
        val_dataset = PriceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

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

    def predict(self, test_df, device='cuda'):
        self.to(device)
        self.eval()
        X_tensor = torch.tensor(test_df, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = self(X_tensor).cpu().numpy()
        return predictions.tolist()

    def evaluate(self, test_df, test_labels, device='cuda'):
        self.to(device)
        self.eval()
        X_tensor = torch.tensor(test_df, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(test_labels, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = self(X_tensor)
            mse = nn.MSELoss()(y_pred, y_tensor).item()
            mae = torch.mean(torch.abs(y_pred - y_tensor)).item()
        return [mse, mae]


# Main Entry Point
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    BASE_DIR = os.getenv("BASE_DIR")
    file_path = f"{BASE_DIR}/data/crypto/klines/BTCUSDT/BTCUSDT_15m.csv"
    sample_data = pd.read_csv(file_path)
    df_features = calculate_indicators(sample_data)

    scaler_X = MinMaxScaler()
    df_features = scaler_X.fit_transform(df_features)
    labels = create_labels(sample_data)

    train_size = int(0.8 * len(df_features))
    train_df = df_features[:train_size]
    test_df = df_features[train_size:]

    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    regressor = HybridPriceRegressor(lookback_period=50, input_features=len(features))
    regressor.train_model(train_df, train_labels, batch_size=64, validation_split=0.2, patience=10)

    predictions = regressor.predict(test_df)
    print("Sample predictions (first 500):")
    for i, pred in enumerate(predictions[:500]):
        print(f"Prediction {i + 1}: {pred}")

    loss, mae = regressor.evaluate(test_df, test_labels)
    print(f"Evaluation Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
