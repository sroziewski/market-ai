import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Custom Dataset
class PriceDataset(Dataset):
    """Custom Dataset for price data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model Definition
class HybridPriceRegressor(nn.Module):
    def __init__(self, lookback_period=50, input_features=4, cnn_filters=32, lstm_units=64, dropout_rate=0.3):
        super(HybridPriceRegressor, self).__init__()
        self.lookback_period = lookback_period
        self.input_features = input_features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # CNN Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # LSTM Temporal Processing
        self.lstm1 = nn.LSTM(cnn_filters * 2, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)

        # Dense Regression Layers
        self.fc1 = nn.Linear(lstm_units // 2, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 6)  # [low_20, high_20, avg_20, low_50, high_50, avg_50]

    def forward(self, x):
        x = x.transpose(1, 2)  # Reshape for Conv1d: (batch, features, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.transpose(1, 2)  # Reshape for LSTM: (batch, seq_len, features)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take the last output of the LSTM
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Final output
        return x

    def prepare_data(self, klines_df):
        if not all(col in klines_df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame must include 'open', 'high', 'low', 'close' columns.")
        data = klines_df[['open', 'high', 'low', 'close']].values
        data_scaled = self.scaler_X.fit_transform(data)
        X = [data_scaled[i - self.lookback_period:i] for i in range(self.lookback_period, len(data))]
        return np.array(X), self.scaler_X

    def create_labels(self, klines_df):
        close_prices = klines_df['close'].values
        high_prices = klines_df['high'].values
        low_prices = klines_df['low'].values
        y = []
        for i in range(self.lookback_period, len(close_prices)):
            window_20 = slice(i, min(i + 20, len(close_prices)))
            window_50 = slice(i, min(i + 50, len(close_prices)))
            y.append([
                np.min(low_prices[window_20]),
                np.max(high_prices[window_20]),
                np.mean(close_prices[window_20]),
                np.min(low_prices[window_50]),
                np.max(high_prices[window_50]),
                np.mean(close_prices[window_50])
            ])
        y_scaled = self.scaler_y.fit_transform(np.array(y))
        return y_scaled

    def train(self, klines_df, epochs=50, batch_size=32, validation_split=0.2, patience=10,
              device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.to(device)
        X, _ = self.prepare_data(klines_df)
        y = self.create_labels(klines_df)

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = PriceDataset(X_train, y_train)
        val_dataset = PriceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = self(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        self.load_state_dict(best_model_state)

    def predict(self, klines_df, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.to(device)
        self.eval()
        X, _ = self.prepare_data(klines_df)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions_scaled = self(X_tensor).cpu().numpy()
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions.tolist()

    def evaluate(self, klines_df, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.to(device)
        self.eval()
        X, _ = self.prepare_data(klines_df)
        y = self.create_labels(klines_df)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = self(X_tensor)
            mse = nn.MSELoss()(y_pred, y_tensor).item()
            mae = torch.mean(torch.abs(y_pred - y_tensor)).item()
        return [mse, mae]


# Main Program Execution
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    sample_data = pd.read_csv("/path/to/BTCUSDT_15m.csv")  # Replace with your data path

    regressor = HybridPriceRegressor(lookback_period=50, input_features=4)
    regressor.train(sample_data, epochs=50, batch_size=32, validation_split=0.2, patience=10)

    predictions = regressor.predict(sample_data)
    print("Sample predictions (first 5):")
    for i, pred in enumerate(predictions[:5]):
        print(f"Timestep {i + 50}: {pred}")

    loss, mae = regressor.evaluate(sample_data)
    print(f"Evaluation Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
