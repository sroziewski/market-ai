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

# Instantiate your model class (ensure features match your structure)
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

# Output predictions with corresponding timestamps
if 'timestamp' not in test_data.columns:
    raise ValueError("'timestamp' column not found in the test data")

print("All predictions with timestamps:")
for i, (timestamp, pred) in enumerate(
        zip(test_data['timestamp'], predictions)):  # Iterate over timestamps and predictions
    print(f"{i + 1} | Timestamp: {timestamp} | Prediction: {pred}")

print(f"Prediction completed in: {end_prediction_time - start_prediction_time:.2f} seconds")

# Evaluate the model on the test dataset
loss, mae = regressor.evaluate(test_data)
print(f"Test Set Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
