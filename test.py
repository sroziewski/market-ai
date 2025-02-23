import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Test basic TensorFlow functionality
# Create a simple computation: a constant matrix multiplication
try:
    a = tf.constant([[2, 3]])
    b = tf.constant([[1], [4]])
    result = tf.matmul(a, b)

    print("Matrix multiplication result:")
    print(result.numpy())  # Output the result
except Exception as e:
    print(f"An error occurred while testing TensorFlow: {e}")
