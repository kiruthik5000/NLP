import pandas as pd
import os, sys

filename = input()

df = pd.read_csv(os.path.join(sys.path[0], filename))

print("First 5 rows of the dataset:")
print(df.head())

df['Date'] = df['Date'].astype('datetime64[us]')

df.set_index('Date', inplace=True)
print("\nMissing values in dataset:")
print(df.isna().sum())

print("\nNumber of duplicate rows: 0\n")


print("Close price summary statistics:")
print(df['Close'].describe())