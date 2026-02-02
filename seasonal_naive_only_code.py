import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def seasonal_naive_forecast(series, season_length, forecast_horizon):
    if len(series) < season_length:
        raise ValueError("Series length must be at least equal to season_length.")

    last_season_values = series.iloc[-season_length:]
    forecast_values = []
    for i in range(forecast_horizon):
        forecast_values.append(last_season_values.iloc[i % season_length])

    if isinstance(series.index, pd.DatetimeIndex):
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            if season_length == 12:
                freq = 'MS'
            elif season_length == 7:
                freq = 'D'
            else:
                freq = 'D'
        
        # Adjusting the start date for date_range to correctly follow the last date of the series
        # For 'D' frequency, add 1 day to start the forecast immediately after the last observation
        # For other frequencies like 'MS', `start` itself determines the first point of the next period.
        if freq == 'D' or freq == 'B': # Assuming 'B' (business day) also needs explicit day increment
             forecast_start_date = last_date + pd.Timedelta(days=1)
        elif freq == 'W-SUN' or freq == 'W-MON' or freq == 'W-TUE' or freq == 'W-WED' or freq == 'W-THU' or freq == 'W-FRI' or freq == 'W-SAT':
            # For weekly frequencies, the next period starts on the specified day of the week
            forecast_start_date = last_date + pd.Timedelta(weeks=1)
        elif freq in ['MS', 'M', 'QS', 'Q', 'AS', 'A']:
            # For monthly, quarterly, annual starts, pd.date_range with `start=last_date` and `periods=forecast_horizon+1` will correctly determine the next period.
            # However, we need to make sure the start date is indeed *after* the last historical date.
            # A simpler way is to just let date_range figure it out, then slice from the second element.
            # Or, we can use `pd.tseries.frequencies.to_offset(freq).rollforward(last_date)` to get the start of the next period.
            try:
                offset = pd.tseries.frequencies.to_offset(freq)
                forecast_start_date = last_date + offset
            except ValueError:
                # Fallback if to_offset fails for some reason
                forecast_start_date = last_date + pd.Timedelta(days=1) # generic fallback
        else: # Generic fallback for other frequencies or if detection fails
            forecast_start_date = last_date + pd.Timedelta(days=1)


        # The date_range will generate one extra point at the beginning if using `start=last_date` with `freq` that starts on a specific point *within* the last period.
        # So we create `forecast_horizon + 1` periods and then take from the second one.
        # This handles cases where freq like 'MS' or 'M' would include the last_date if not handled carefully.
        forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_horizon, freq=freq)
    else:
        forecast_index = range(len(series), len(series) + forecast_horizon)

    return pd.Series(forecast_values, index=forecast_index)

if __name__ == "__main__":
    dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
    np.random.seed(42)
    trend = np.linspace(0, 5, 36)
    season = np.tile([10, 12, 15, 18, 20, 22, 25, 23, 20, 17, 14, 11], 3)
    noise = np.random.normal(0, 1.5, 36)
    data = trend + season + noise
    time_series = pd.Series(data, index=dates)

    season_length = 12
    forecast_horizon = 24

    train_size = len(time_series) - season_length
    train_series = time_series.iloc[:train_size]
    test_series = time_series.iloc[train_size:]

    forecast = seasonal_naive_forecast(train_series, season_length, forecast_horizon)

    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series.values, label='Training Data', color='blue')
    plt.plot(test_series.index, test_series.values, label='Actual Future Data (for comparison)', color='green', linestyle='--')
    plt.plot(forecast.index, forecast.values, label='Seasonal Naive Forecast', color='red', linestyle=':')
    plt.title('Seasonal Naive Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    common_index = forecast.index.intersection(test_series.index)
    if not common_index.empty:
        forecast_aligned = forecast[common_index]
        test_aligned = test_series[common_index]
        mae = np.mean(np.abs(forecast_aligned - test_aligned))
        rmse = np.sqrt(np.mean((forecast_aligned - test_aligned)**2))
        print(f"\nMean Absolute Error (MAE) on test set: {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE) on test set: {rmse:.2f}")
    else:
        print("\nNo overlapping dates between forecast and test series for direct comparison.")
