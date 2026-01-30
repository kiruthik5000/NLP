import os
import sys
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def main():
    filename = input("").strip()
    file_path = os.path.join(sys.path[0], filename)

    df = pd.read_csv(file_path)

    date_col = 'Datetime'

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.set_index(date_col, inplace=True)

    df['Power_Consumption_diff'] = pd.to_numeric(
        df['Power_Consumption_diff'], errors='coerce'
    )

    df = df.dropna(subset=['Power_Consumption_diff'])

    if len(df) < 5:
        print("Insufficient data for AR(2) and MA(1) modeling.")
        print("At least 5 non-null differenced observations are required.")
        return

    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    print("Training data size:", len(train))
    print("Testing data size:", len(test))
    print()


    ar_model = ARIMA(train["Power_Consumption_diff"], order=(2, 0, 0))
    ar_fitted = ar_model.fit()

    print("AR(2) Model Summary:")
    print(ar_fitted.summary())
    print()

    ma_model = ARIMA(train["Power_Consumption_diff"], order=(0, 0, 1))
    ma_fitted = ma_model.fit()

    print("MA(1) Model Summary:")
    print(ma_fitted.summary())


if __name__ == "__main__":
    main()
