import os

import torch
import pandas as pd
import time

from dotenv import load_dotenv

from test_model import HybridPriceRegressor

load_dotenv()

# Load your test data
BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    raise ValueError("Environment variable 'BASE_DIR' not set")
file_path = f"{BASE_DIR}/data/crypto/klines/ETHUSDT/ETHUSDT_15m.csv"
test_data = pd.read_csv(file_path)

# Convert necessary columns to scalar types if needed
required_columns = ['timestamp', 'open', 'high', 'low', 'close']
for col in required_columns:
    if col not in test_data.columns:
        raise ValueError(f"'{col}' column not found in the test data")
    # Ensure no lists/arrays; consider selecting the first value if a column has lists
    test_data[col] = test_data[col].apply(lambda x: x if not isinstance(x, (list, tuple)) else x[0])

# Instantiate your model class
regressor = HybridPriceRegressor(lookback_period=50, input_features=4)

# Load the saved model
model_path = "final_hybrid_price_regressor2.pth"
regressor.load_state_dict(torch.load(model_path))  # Load the model's weights
regressor.eval()  # Set model to evaluation mode

# Measure prediction time
start_prediction_time = time.time()  # Record start time
regressor.create_labels(test_data)  # Generate labels if necessary
predictions = regressor.predict(test_data)  # Generate predictions for test_data
end_prediction_time = time.time()  # Record end time

# Output predictions with corresponding timestamps and OHLC values
print("All predictions with timestamps and OHLC values:")
for i, (timestamp, open_, high, low, close, pred) in enumerate(zip(
        test_data['timestamp'], test_data['open'], test_data['high'], test_data['low'], test_data['close'], predictions
)):
    # Ensure all numerical values are printed correctly
    open_, high, low, close, pred = float(open_), float(high), float(low), float(close), float(pred)
    print(
        f"{i + 1} | Timestamp: {timestamp} | Open: {open_:.4f} | High: {high:.4f} | Low: {low:.4f} | Close: {close:.4f} | Prediction: {pred:.4f}")

print(f"Prediction completed in: {end_prediction_time - start_prediction_time:.2f} seconds")

# Evaluate the model on the test dataset
loss, mae = regressor.evaluate(test_data)
print(f"Test Set Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
