import pandas as pd
import os, sys
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

filename = input().strip()

df = pd.read_csv(os.path.join(sys.path[0], filename), parse_dates=['Datetime'])

df.sort_values('Datetime', inplace=True)
df.set_index('Datetime', inplace=True)
df = df.asfreq('MS')   # Month Start
y = df['Power_Consumption_diff'].dropna()

train_size = int(len(y) * 0.8)

train = y.iloc[:train_size]
test = y.iloc[train_size:]

print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}")
print()

results = {}

for p in range(1, 6):
    model = ARIMA(train, order=(p, 0, 0))
    result = model.fit()
    results[f'AR({p})'] = result.aic
    print(f"AR({p}) AIC: {result.aic}")

for q in range(1, 6):
    model = ARIMA(train, order=(0, 0, q))
    result = model.fit()
    results[f'MA({q})'] = result.aic
    print(f"MA({q}) AIC: {result.aic}")

print()

best_model = min(results, key=results.get)
print("Best Model Selected:")
print(best_model)
print()

if best_model.startswith('AR'):
    p = int(best_model.split('(')[1].split(')')[0])
    final_model = ARIMA(train, order=(p, 0, 0))
else:
    q = int(best_model.split('(')[1].split(')')[0])
    final_model = ARIMA(train, order=(0, 0, q))

final_result = final_model.fit()
print(final_result.summary())

lb_test = acorr_ljungbox(final_result.resid, lags=[1], return_df=True)

print("\nLjung-Box Test Results:")
print(lb_test)