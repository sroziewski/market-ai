import os
import sys
import time
import torch
import pandas as pd
from dotenv import load_dotenv
from test_model import HybridPriceRegressor


def main():
    # Load environment variables
    load_dotenv()

    # Validate program arguments
    if len(sys.argv) != 3:
        print("Usage: python program.py <model_path> <data_file_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_file = sys.argv[2]

    # Validate the base directory from environment
    BASE_DIR = os.getenv("BASE_DIR")
    if BASE_DIR is None:
        raise ValueError("Environment variable 'BASE_DIR' not set")

    # Construct the full file path
    file_path = f"{BASE_DIR}/data/crypto/klines/{data_file}"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    # Load the test data
    test_data = pd.read_csv(file_path)

    # Ensure OHLC columns contain scalar values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        test_data[col] = test_data[col].apply(lambda x: x[0] if isinstance(x, list) else x).astype(float)

    # Ensure 'timestamp' is a scalar value (optional: convert to datetime)
    if 'timestamp' in test_data.columns:
        test_data['timestamp'] = test_data['timestamp'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Instantiate the model
    regressor = HybridPriceRegressor(lookback_period=50, input_features=5)

    # Validate and load the saved model checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    regressor.load_state_dict(torch.load(model_path))
    regressor.eval()

    # Measure the prediction time
    start_prediction_time = time.time()
    regressor.create_labels(test_data)  # Generate labels if needed
    predictions = regressor.predict(test_data)  # Generate predictions for test_data
    end_prediction_time = time.time()

    # Output predictions with timestamps, OHLC values, and full predictions
    print("All predictions with timestamps, OHLC values, and full predictions:")

    window_size = 50  # Prediction window size

    # Process and print predictions
    for i, (pred) in enumerate(predictions):
        # The corresponding index in test_data based on the window size
        index = i + window_size

        # Ensure valid index range for data retrieval
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
            pred_values = ', '.join([f"{float(p):.4f}" for p in pred])  # Format prediction values
        else:
            raise ValueError(f"Prediction does not have 6 elements: {pred}")

        print(
            f"{i + 1} | Timestamp: {timestamp} | Open: {open_:.4f} | High: {high:.4f} | Low: {low:.4f} | Close: {close:.4f} | Predictions: [{pred_values}]"
        )

    print(f"Prediction completed in: {end_prediction_time - start_prediction_time:.2f} seconds")

    # Evaluate the model on the test dataset
    loss, mae = regressor.evaluate(test_data)
    print(f"Test Set Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
