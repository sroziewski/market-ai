import torch
import pandas as pd
import time

from test_model import HybridPriceRegressor

# Load your test data
file_path = "path_to_your_test_data_file.csv"
test_data = pd.read_csv(file_path)

# Instantiate your model class (ensure features match your structure)
# Assuming your model is called HybridPriceRegressor
regressor = HybridPriceRegressor(lookback_period=50, input_features=4)

# Load the saved model
model_path = "final_hybrid_price_regressor2.pth"
regressor.load_state_dict(torch.load(model_path))  # Load the model's weights
regressor.eval()  # Set model to evaluation mode

# Measure prediction time
start_prediction_time = time.time()  # Record start time
predictions = regressor.predict(test_data)  # Generate predictions for test_data
end_prediction_time = time.time()  # Record end time

# Output predictions
print("Sample predictions (first 50):")
for i, pred in enumerate(predictions[:50]):  # Print first 5 predictions
    print(f"Prediction {i + 1}: {pred}")
print(f"Prediction completed in: {end_prediction_time - start_prediction_time:.2f} seconds")

# Evaluate the model on the test dataset
loss, mae = regressor.evaluate(test_data)
print(f"Test Set Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")
