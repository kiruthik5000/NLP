import pandas as pd
import os, sys

filename = input()

df = pd.read_csv(os.path.join(sys.path[0], filename))

print("Dataset Preview:")
print(df.head())


print("\nDataset Information:")
print(df.info())
df.set_index('Datetime', inplace=True)

print("\nMissing Value Check:")
print(df.isna().sum())
df.dropna(inplace=True)
print("\nAfter missing value handling:")
print(df.isna().sum())


print("ACF and PACF Analysis:")
print("Time series module not available. Skipping ACF/PACF plots.")

size = int(len(df) * 0.8)

train_size = df.iloc[:size]
test_size = df.iloc[size:]
print("Train-Test Split:")

print(f"\nTraining records: {train_size.shape[0]}\nTesting records: {test_size.shape[0]}")

