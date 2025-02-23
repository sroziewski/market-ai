import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class HybridPriceRegressor:
    def __init__(self, lookback_period=50, input_features=4, cnn_filters=32, lstm_units=64, dropout_rate=0.3):
        """
        Initialize the hybrid CNN-LSTM regressor for predicting future price values.

        Parameters:
        - lookback_period (int): Number of time steps to look back
        - input_features (int): Number of features per time step (OHLC = 4)
        - cnn_filters (int): Number of filters in CNN layers
        - lstm_units (int): Number of units in LSTM layer
        - dropout_rate (float): Dropout rate for regularization
        """
        self.lookback_period = lookback_period
        self.input_features = input_features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self):
        """Build the improved hybrid CNN-LSTM architecture for regression"""
        model = Sequential([
            # CNN Feature Extraction
            Conv1D(filters=self.cnn_filters, kernel_size=3, activation='relu',
                   input_shape=(self.lookback_period, self.input_features), padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=self.cnn_filters * 2, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            # LSTM Temporal Processing
            LSTM(self.lstm_units, return_sequences=True),
            BatchNormalization(),
            LSTM(self.lstm_units // 2),

            # Dense Regression Layers
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(6)  # 6 outputs: low_20, high_20, avg_20, low_50, high_50, avg_50
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mae'])
        return model

    def prepare_data(self, klines_df):
        """
        Prepare and scale input data (X) for the model.

        Returns:
        - X (array): Shape (samples, lookback_period, input_features)
        - scaler_X: Fitted scaler for input features
        """
        if not all(col in klines_df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close' columns.")

        data = klines_df[['open', 'high', 'low', 'close']].values
        data_scaled = self.scaler_X.fit_transform(data)
        X = [data_scaled[i - self.lookback_period:i] for i in range(self.lookback_period, len(data))]

        return np.array(X), self.scaler_X

    def create_labels(self, klines_df):
        """
        Create and scale target values for next 20 and 50 timesteps.

        Returns:
        - y (array): Shape (samples, 6) with scaled [low_20, high_20, avg_20, low_50, high_50, avg_50]
        """
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

    def train(self, klines_df, epochs=50, batch_size=32, validation_split=0.2, patience=10):
        """
        Train the model with early stopping and learning rate reduction.
        """
        X, _ = self.prepare_data(klines_df)
        y = self.create_labels(klines_df)

        # Callbacks for training optimization
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return history

    def predict(self, klines_df):
        """
        Predict future price values and inverse-transform to original scale.

        Returns:
        - predictions (list): List of [low_20, high_20, avg_20, low_50, high_50, avg_50]
        """
        X, _ = self.prepare_data(klines_df)
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions.tolist()

    def evaluate(self, klines_df):
        """
        Evaluate model performance.

        Returns:
        - metrics (list): Loss (MSE) and MAE
        """
        X, _ = self.prepare_data(klines_df)
        y = self.create_labels(klines_df)
        return self.model.evaluate(X, y)


# Example usage
if __name__ == "__main__":
    dates = pd.date_range(start='2023-01-01', periods=300, freq='1H')
    base_trend = np.linspace(100, 120, 300) + np.random.random(300) * 5 - 2.5
    sample_data = pd.DataFrame({
        'open': base_trend + np.random.random(300) * 2 - 1,
        'high': base_trend + np.random.random(300) * 3,
        'low': base_trend - np.random.random(300) * 3,
        'close': base_trend
    }, index=dates)

    regressor = HybridPriceRegressor(lookback_period=50, input_features=4)
    history = regressor.train(sample_data, epochs=50, batch_size=32, validation_split=0.2, patience=10)

    predictions = regressor.predict(sample_data)
    print("Sample predictions (first 5):")
    for i, pred in enumerate(predictions[:5]):
        print(f"Timestep {i + 50}: Low_20={pred[0]:.2f}, High_20={pred[1]:.2f}, Avg_20={pred[2]:.2f}, "
              f"Low_50={pred[3]:.2f}, High_50={pred[4]:.2f}, Avg_50={pred[5]:.2f}")

    loss, mae = regressor.evaluate(sample_data)
    print(f"Evaluation Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")