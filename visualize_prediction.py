import matplotlib.pyplot as plt
import pandas as pd
import os

# Read the CSV data
data = pd.read_csv("predictions/final_hybrid_price_regressor_l3_m8_true_ETHUSDT_1d.csv")


# Function to draw and save the plot
def plot_columns_and_save():
    # Define the output directory and file name
    output_dir = "visualizations"
    file_name = "final_hybrid_price_regressor_l3_m8_true_ETHUSDT_1d_window_50.png"
    output_path = os.path.join(output_dir, file_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data["window_50_min"], label="window_50_min", color="blue")
    plt.plot(data["window_50_max"], label="window_50_max", color="green")
    plt.plot(data["close"], label="close", color="red")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Visualization of window_50_min, window_50_max, and close")
    plt.legend()
    plt.grid()

    # Save the plot to the specified file
    plt.savefig(output_path, format="png")
    print(f"Plot saved to: {output_path}")

    # Display the plot (optional)
    plt.show()


# Call the function
plot_columns_and_save()
