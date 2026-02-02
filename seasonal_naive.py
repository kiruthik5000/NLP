import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def seasonal_naive_forecast(series, season_length, forecast_horizon):
    """
    Applies a seasonal naive forecast to a time series.

    Args:
        series (pd.Series): The input time series.
        season_length (int): The length of one season (e.g., 12 for monthly data with yearly seasonality).
        forecast_horizon (int): The number of future steps to forecast.

    Returns:
        pd.Series: A Series containing the seasonal naive forecasts.
    """
    if len(series) < season_length:
        raise ValueError("Series length must be at least equal to season_length.")

    # Get the last 'season_length' values from the original series
    # These will be used as the "naive" predictions for the next season
    last_season_values = series.iloc[-season_length:]

    forecast_values = []
    for i in range(forecast_horizon):
        # The forecast for the i-th step is the value from 'season_length' steps ago
        forecast_values.append(last_season_values.iloc[i % season_length])

    # Create a new index for the forecast
    # Assuming the original series has a DatetimeIndex
    if isinstance(series.index, pd.DatetimeIndex):
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            # Fallback for irregular frequencies, though a proper frequency is ideal
            # Here, we'll just add a generic time unit for demonstration
            if season_length == 12: # Assume monthly if season_length is 12 and freq not inferred
                freq = 'MS'
            elif season_length == 7: # Assume daily if season_length is 7
                freq = 'D'
            else:
                freq = 'D' # Default to daily if no obvious frequency

        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1) if freq == 'D' else last_date,
                                       periods=forecast_horizon + 1, freq=freq)[1:]
    else:
        # For non-datetime index, just use a range
        forecast_index = range(len(series), len(series) + forecast_horizon)


    return pd.Series(forecast_values, index=forecast_index)

if __name__ == "__main__":
    # --- 1. Generate Sample Seasonal Data ---
    # Create a time series with yearly seasonality (e.g., monthly sales data)
    dates = pd.date_range(start='2020-01-01', periods=36, freq='MS') # 3 years of monthly data
    np.random.seed(42)
    # Base trend
    trend = np.linspace(0, 5, 36)
    # Seasonal component (e.g., higher sales in summer/fall)
    season = np.tile([10, 12, 15, 18, 20, 22, 25, 23, 20, 17, 14, 11], 3)
    # Noise
    noise = np.random.normal(0, 1.5, 36)

    data = trend + season + noise
    time_series = pd.Series(data, index=dates)

    print("Original Time Series Head:")
    print(time_series.head())
    print("\nOriginal Time Series Tail:")
    print(time_series.tail())

    # --- 2. Apply Seasonal Naive Forecast ---
    season_length = 12  # Yearly seasonality for monthly data
    forecast_horizon = 24 # Forecast for the next 24 months (2 years)

    # Split data into training and testing (for demonstration)
    train_size = len(time_series) - season_length
    train_series = time_series.iloc[:train_size]
    test_series = time_series.iloc[train_size:]

    print(f"\nTrain Series End Date: {train_series.index[-1]}")
    print(f"Test Series Start Date: {test_series.index[0]}")

    forecast = seasonal_naive_forecast(train_series, season_length, forecast_horizon)

    print("\nSeasonal Naive Forecast Head:")
    print(forecast.head())
    print("\nSeasonal Naive Forecast Tail:")
    print(forecast.tail())

    # --- 3. Plotting the Results ---
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

    # --- 4. Evaluate (Optional) ---
    # For a more robust evaluation, you'd compare the forecast to actual future values
    # Here, we'll compare the first part of the forecast with the test_series
    
    # Ensure forecast and test_series have overlapping indices for comparison
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
