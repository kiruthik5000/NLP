import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os, sys

filename = input()

df = pd.read_csv(os.path.join(sys.path[0], filename))

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['DATE'] = df['DATE'].dt.strftime('%Y-%d-%m')
df.set_index('DATE', inplace=True)

print("First 5 records of dataset:")
print(df.head())

df['Consumption'] = df['Consumption'].fillna(df['Consumption'].mean())


def remove_outlier(df):
    q1 = df['Consumption'].quantile(0.25)
    q3 = df['Consumption'].quantile(0.75)

    IQR = q3 - q1

    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR

    return df[
        (df['Consumption'] >= lower_bound) &
        (df['Consumption'] <= upper_bound)
        ]


df = remove_outlier(df)

print("data preprocessing completed.")


def printer(model, name):
    print(f"\n{name} Model Components (First 5 Values)\nTrend:")
    print(model.trend.dropna().head())

    print("\nSeasonality:")
    print(model.seasonal.dropna().head())

    print("\nResiduals:")
    print(model.resid.dropna().head())


add_decomp = seasonal_decompose(
    df['Consumption'],
    model='additive',
    period=12
)

mul_decomp = seasonal_decompose(
    df['Consumption'],
    model='multiplicative',
    period=12
)
printer(add_decomp, name='Additive')
printer(mul_decomp, name='Multiplicative')
print("""Model Comparison Conclusion:\n
If seasonal values are constant → Additive model fits better.\n
If seasonal values change proportionally with trend → Multiplicative model fits better.""")
