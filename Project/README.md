# Project: Time Series Forecasting with ARIMA

This directory contains a sub-project focused on time series forecasting.

## Objective
The primary goal of this project is to build and evaluate a time series forecasting model using the ARIMA (AutoRegressive Integrated Moving Average) algorithm. The project involves data loading, exploratory data analysis, stationarity testing, model training, prediction, and performance evaluation.

## Contents

-   `Model.ipynb`: This Jupyter Notebook details the entire workflow of the time series forecasting project. It includes:
    -   Importing necessary libraries (pandas, matplotlib, statsmodels, sklearn).
    -   Loading the dataset from `Data/data.csv`.
    -   Initial data exploration and visualization.
    -   Performing the Augmented Dickey-Fuller (ADF) test for stationarity.
    -   Generating Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to determine ARIMA parameters.
    -   Splitting data into training and testing sets.
    -   Training an ARIMA(1, 2, 1) model.
    -   Generating predictions on the test set.
    -   Visualizing actual vs. predicted values.
    -   Evaluating model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

-   `Data/`: This subdirectory contains the dataset used for the time series analysis.
    -   `data.csv`: The primary dataset for this forecasting task, which is read and processed by the `Model.ipynb` notebook.

## Getting Started

To run this project, ensure you have Python and Jupyter Notebook installed. You can open `Model.ipynb` in a Jupyter environment to execute the code cells and reproduce the analysis. Make sure the `data.csv` file is correctly placed in the `Data` subdirectory relative to `Model.ipynb`.

## Dependencies

The key libraries used in this project include:
-   pandas
-   matplotlib
-   statsmodels
-   scikit-learn (sklearn)

These can typically be installed via pip:
`pip install pandas matplotlib statsmodels scikit-learn`
