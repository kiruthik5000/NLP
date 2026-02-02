import sys, os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
filename = input()
file_path = os.path.join(sys.path[0], filename)

df = pd.read_csv(file_path)
print("Dataset Preview:")
print(df.head())
print()
print("Dataset Information:")
info = df.info()
print(info)
print()
new_df = df.iloc[:, :-3]
new_df.set_index('Date', inplace=True)
print("Missing Value Check:")
numeric_cols = new_df.select_dtypes(include=['float64', 'int64']).columns
print(new_df.isnull().sum())
new_df = new_df.dropna()
print("After missing value handling:")
print(new_df.isnull().sum())
print()
split_index = int(len(new_df) * 0.8)
train = new_df['Close'][:split_index]
test = new_df['Close'][split_index:]
print("Train-Test Split:")
print(f"Training records: {len(train)}")
print(f"Testing records: {len(test)}")

print()
print("SARIMA Model Summary:")
try:
    from pmdarima import auto_arima
    model = auto_arima(
        train,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    print(model.summary())
except Exception:
    print("pmdarima not available. SARIMA modeling skipped.")
