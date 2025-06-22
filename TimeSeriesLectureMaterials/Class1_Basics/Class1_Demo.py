# Class 1: Introduction to the Basics of Time Series Analysis - Python Demonstrations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore") # Ignore harmless warnings

# --- 1. Load and Initial Preprocessing ---
print("--- 1. Loading and Preprocessing Data ---")
data_file = "/home/ubuntu/aapl_stock_data_10y.csv"
df = pd.read_csv(data_file, index_col='Date', parse_dates=True)

# Select Adjusted Close price
ts = df['Adj Close'].copy()

# Check for missing values (should be none after fetch script)
print(f"Missing values: {ts.isnull().sum()}")

# Plot the raw time series
plt.figure(figsize=(12, 6))
plt.plot(ts)
plt.title('AAPL Adjusted Close Price (10 Years)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.savefig("/home/ubuntu/plot_01_raw_data.png")
plt.close()
print("Saved plot: plot_01_raw_data.png")

# --- 2. Moving Averages --- 
print("\n--- 2. Calculating Moving Averages ---")
# Calculate 50-day and 200-day Simple Moving Averages (SMA)
rolling_mean_50 = ts.rolling(window=50).mean()
rolling_mean_200 = ts.rolling(window=200).mean()

# Plot the data with moving averages
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Adj Close')
plt.plot(rolling_mean_50, label='50-Day SMA', color='orange')
plt.plot(rolling_mean_200, label='200-Day SMA', color='red')
plt.title('AAPL Adj Close with 50 & 200 Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig("/home/ubuntu/plot_02_moving_averages.png")
plt.close()
print("Saved plot: plot_02_moving_averages.png")

# --- 3. Exponential Smoothing --- 
print("\n--- 3. Applying Exponential Smoothing ---")
# Simple Exponential Smoothing (SES)
# Note: SES is best for data without trend/seasonality, applying here for demonstration
ses_model = SimpleExpSmoothing(ts).fit(smoothing_level=0.2)
ses_fitted = ses_model.fittedvalues

# Holt's Linear Trend
holt_model = Holt(ts).fit()
holt_fitted = holt_model.fittedvalues

# Holt-Winters Seasonal Smoothing
# Note: Daily stock data might not have strong yearly seasonality, using 252 trading days as approx period
# Using additive trend and multiplicative seasonality as an example
hw_model = ExponentialSmoothing(ts, trend='add', seasonal='mul', seasonal_periods=252).fit()
hw_fitted = hw_model.fittedvalues

# Plotting Smoothing Results (Example: Holt's)
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original Adj Close')
plt.plot(holt_fitted, label='Holt\'s Linear Trend Fit', color='red')
plt.title('AAPL Adj Close with Holt\'s Linear Trend Smoothing')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig("/home/ubuntu/plot_03_holt_smoothing.png")
plt.close()
print("Saved plot: plot_03_holt_smoothing.png")
# Note: Could plot SES and HW similarly if needed for lecture

# --- 4. Decomposition --- 
print("\n--- 4. Decomposing Time Series ---")
# Using multiplicative model as stock prices often exhibit multiplicative seasonality/trends
# Using period=252 for approximate annual seasonality in trading days
decomposition_result = seasonal_decompose(ts, model='multiplicative', period=252)

trend = decomposition_result.trend
seasonal = decomposition_result.seasonal
residual = decomposition_result.resid

# Plot decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.suptitle('Multiplicative Decomposition (Period=252)', y=1.02)
plt.savefig("/home/ubuntu/plot_04_decomposition.png")
plt.close()
print("Saved plot: plot_04_decomposition.png")

# --- 5. Stationarity Check (Visual + ADF Test) --- 
print("\n--- 5. Checking for Stationarity ---")
# Visual check: Rolling statistics
rolling_mean = ts.rolling(window=252).mean()
rolling_std = ts.rolling(window=252).std()

plt.figure(figsize=(12, 6))
plt.plot(ts, color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean (252 days)')
plt.plot(rolling_std, color='black', label='Rolling Std Dev (252 days)')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.grid(True)
plt.savefig("/home/ubuntu/plot_05_rolling_stats.png")
plt.close()
print("Saved plot: plot_05_rolling_stats.png")
print("Visual inspection: Mean is clearly trending upwards, indicating non-stationarity.")

# Augmented Dickey-Fuller (ADF) Test
print("\nPerforming Augmented Dickey-Fuller Test on original series:")
adf_result = adfuller(ts.dropna()) # dropna just in case
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'{key:>8}: {value:.4f}')

if adf_result[1] <= 0.05:
    print("Result: Reject the null hypothesis (H0). Series is likely stationary.")
else:
    print("Result: Fail to reject the null hypothesis (H0). Series is likely non-stationary.")

# Try differencing once to achieve stationarity
ts_diff = ts.diff().dropna()

print("\nPerforming Augmented Dickey-Fuller Test on first-differenced series:")
adf_result_diff = adfuller(ts_diff)
print(f'ADF Statistic: {adf_result_diff[0]:.4f}')
print(f'p-value: {adf_result_diff[1]:.4f}')
print('Critical Values:')
for key, value in adf_result_diff[4].items():
    print(f'{key:>8}: {value:.4f}')

if adf_result_diff[1] <= 0.05:
    print("Result: Reject the null hypothesis (H0). Differenced series is likely stationary.")
else:
    print("Result: Fail to reject the null hypothesis (H0). Differenced series is likely non-stationary.")

# Plot differenced series
plt.figure(figsize=(12, 6))
plt.plot(ts_diff)
plt.title('AAPL Adjusted Close Price (First Difference)')
plt.xlabel('Date')
plt.ylabel('Price Difference')
plt.grid(True)
plt.savefig("/home/ubuntu/plot_06_differenced_data.png")
plt.close()
print("Saved plot: plot_06_differenced_data.png")

# --- 6. ACF and PACF Plots --- 
print("\n--- 6. Plotting ACF and PACF ---")
# Plot ACF and PACF for the original series
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(ts, ax=axes[0], lags=40, title='ACF - Original Series')
plot_pacf(ts, ax=axes[1], lags=40, title='PACF - Original Series')
plt.tight_layout()
plt.savefig("/home/ubuntu/plot_07_acf_pacf_original.png")
plt.close()
print("Saved plot: plot_07_acf_pacf_original.png")
print("ACF for original series shows slow decay, typical of non-stationary data.")

# Plot ACF and PACF for the differenced series
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(ts_diff, ax=axes[0], lags=40, title='ACF - Differenced Series')
plot_pacf(ts_diff, ax=axes[1], lags=40, title='PACF - Differenced Series')
plt.tight_layout()
plt.savefig("/home/ubuntu/plot_08_acf_pacf_differenced.png")
plt.close()
print("Saved plot: plot_08_acf_pacf_differenced.png")
print("ACF/PACF for differenced series can help suggest orders (p, q) for ARMA/ARIMA models.")

print("\nClass 1 Demonstrations Complete.")

