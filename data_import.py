# Import Fashion MNIST dataset using Keras
from keras.datasets import fashion_mnist
import pandas as pd
import numpy as np
import os

# Load the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Convert to DataFrame for easier viewing
# Flatten the images for the dataframe
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create DataFrames
train_df = pd.DataFrame(X_train_flat)
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_flat)
test_df['label'] = y_test

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Save to CSV files
train_csv_path = os.path.join(data_dir, 'fashion_mnist_train.csv')
test_csv_path = os.path.join(data_dir, 'fashion_mnist_test.csv')

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print(f"\nData saved to:")
print(f"  Training: {train_csv_path}")
print(f"  Test: {test_csv_path}")
print("\nFirst 5 records of training data:")
print(train_df.head())