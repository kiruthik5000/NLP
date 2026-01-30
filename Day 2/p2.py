import pandas as pd
import numpy as np
import os, sys
import warnings

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

filename = input()
df = pd.read_csv(os.path.join(sys.path[0], filename))

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df.asfreq('M')

series = df['Close_diff'].dropna()

train_size = int(len(series) * 0.8)

train_data = series.iloc[:train_size]
test_data = series.iloc[train_size:]

print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}\n")

models = {}

for p in range(1, 6):
    try:
        model = ARIMA(train_data, order=(p, 0, 0))
        result = model.fit()
        models[f'{p}'] = result.aic
        print(f"AR({p}) AIC: {result.aic}")
    except:
        print('error')

for q in range(1, 6):
    try:
        model = ARIMA(train_data, order=(0, 0, q))
        result = model.fit()
        print(f"MA({q}) AIC: {result.aic}")
    except:
        print("error")

best_model = min(models, key=models.get)
best_aic = models[best_model]

print(f"Best Model: AR({best_model})")

final_model = ARIMA(train_data, order=(int(best_model), 0, 0))
final_result = final_model.fit()
print(final_result.summary())

residuals = final_result.resid
ljung_box = acorr_ljungbox(residuals, lags=[1], return_df=True)

print("\nLjung-Box Test Results:")
print(ljung_box)



