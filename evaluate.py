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
file_path = f"{BASE_DIR}/data/crypto/klines/ETHUSDT/ETHUSDT_1d.csv"
test_data = pd.read_csv(file_path)

# Ensure OHLC columns contain scalar values
for col in ['open', 'high', 'low', 'close', 'volume']:
    test_data[col] = test_data[col].apply(lambda x: x[0] if isinstance(x, list) else x).astype(float)

# Ensure 'timestamp' is a scalar value (optional: convert to datetime)
if 'timestamp' in test_data.columns:
    test_data['timestamp'] = test_data['timestamp'].apply(lambda x: x[0] if isinstance(x, list) else x)

# Instantiate your model class
regressor = HybridPriceRegressor(lookback_period=50, input_features=5)

# Load the saved model
model_path = "final_hybrid_price_regressor_l1_m2.pth"
regressor.load_state_dict(torch.load(model_path))  # Load the model's weights
regressor.eval()  # Set model to evaluation mode

# Measure prediction time
start_prediction_time = time.time()  # Record start time
regressor.create_labels(test_data)  # Generate labels if necessary
predictions = regressor.predict(test_data)  # Generate predictions for test_data
end_prediction_time = time.time()  # Record end time

# Output predictions with corresponding timestamps, OHLC values, and full predictions
print("All predictions with timestamps, OHLC values, and full predictions:")

window_size = 50  # Prediction window size

for i, (pred) in enumerate(predictions):
    # The corresponding index in test_data based on the window size
    index = i + window_size

    # Ensure that we don't exceed the bounds of test_data
    if index < len(test_data['timestamp']):
        timestamp = test_data['timestamp'][index]
        open_ = test_data['open'][index]
        high = test_data['high'][index]
        low = test_data['low'][index]
        close = test_data['close'][index]
    else:
        raise IndexError(f"Index {index} out of range for test_data (length: {len(test_data['timestamp'])})")

    # Ensure predictions have 6 elements
    if isinstance(pred, (list, tuple)) and len(pred) == 6:
        pred_values = ', '.join([f"{float(p):.4f}" for p in pred])  # Format all 6 elements of prediction
    else:
        raise ValueError(f"Prediction does not have 6 elements: {pred}")

    print(
        f"{i + 1} | Timestamp: {timestamp} | Open: {open_:.4f} | High: {high:.4f} | Low: {low:.4f} | Close: {close:.4f} | Predictions: [{pred_values}]"
    )



print(f"Prediction completed in: {end_prediction_time - start_prediction_time:.2f} seconds")

# Evaluate the model on the test dataset
loss, mae = regressor.evaluate(test_data)
print(f"Test Set Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
