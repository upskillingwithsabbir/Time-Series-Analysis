# Class 2: Time Series Analysis with Statistical Modeling - Python Demonstrations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings("ignore") # Ignore harmless warnings

# --- 1. Load Data and Prepare --- 
print("--- 1. Loading Data ---")
data_file = "/home/ubuntu/aapl_stock_data_10y.csv"
df = pd.read_csv(data_file, index_col='Date', parse_dates=True)

# Use Adjusted Close price
ts = df['Adj Close'].copy()
# Use Volume as an exogenous variable example
exog = df['Volume'].copy()

# Use log returns for GARCH modeling (common practice)
log_returns = np.log(ts / ts.shift(1)).dropna()

# Split data: Train (first 9 years), Test (last 1 year approx)
# ~252 trading days per year
train_size = len(ts) - 252
train_ts, test_ts = ts[:train_size], ts[train_size:]
train_exog, test_exog = exog[:train_size], exog[train_size:]
train_log_returns, test_log_returns = log_returns[:train_size], log_returns[train_size:]

print(f"Train set size: {len(train_ts)}")
print(f"Test set size: {len(test_ts)}")

# Initialize variables for metrics to avoid NameError if a model fails
arima_rmse, arima_mae = np.nan, np.nan
auto_arima_rmse, auto_arima_mae = np.nan, np.nan
sarimax_rmse, sarimax_mae = np.nan, np.nan
best_order = (0,0,0) # Default order

# --- 2. ARIMA Model --- 
print("\n--- 2. Fitting ARIMA Model ---")
# Based on Class 1, the series needed differencing (d=1).
# ACF/PACF of differenced series can suggest p, q. Let's try ARIMA(1,1,1) as a starting point.
# Note: Order selection is iterative. ACF/PACF gives hints.

try:
    arima_model = ARIMA(train_ts, order=(1, 1, 1))
    arima_fit = arima_model.fit()
    print(arima_fit.summary())

    # Forecast
    arima_pred = arima_fit.predict(start=len(train_ts), end=len(ts)-1)

    # Plot forecast vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(train_ts.index, train_ts, label='Train')
    plt.plot(test_ts.index, test_ts, label='Test')
    plt.plot(test_ts.index, arima_pred, label='ARIMA(1,1,1) Forecast')
    plt.title('ARIMA(1,1,1) Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/plot_09_arima_forecast.png")
    plt.close()
    print("Saved plot: plot_09_arima_forecast.png")

    # Performance Metrics
    arima_rmse = np.sqrt(mean_squared_error(test_ts, arima_pred))
    arima_mae = mean_absolute_error(test_ts, arima_pred)
    print(f"ARIMA(1,1,1) RMSE: {arima_rmse:.4f}")
    print(f"ARIMA(1,1,1) MAE: {arima_mae:.4f}")

except Exception as e:
    print(f"Error fitting ARIMA(1,1,1): {e}")

# --- 3. AUTO ARIMA --- 
print("\n--- 3. Fitting AUTO ARIMA Model ---")
# Automatically find best ARIMA model
try:
    auto_arima_model = pm.auto_arima(train_ts, 
                                     start_p=1, start_q=1,
                                     test='adf', # use adf test to find optimal 'd'
                                     max_p=3, max_q=3, # maximum p and q
                                     m=1, # Non-seasonal
                                     d=None, # let model determine 'd'
                                     seasonal=False, # No Seasonality
                                     start_P=0, D=0, 
                                     trace=True,
                                     error_action='ignore', 
                                     suppress_warnings=True, 
                                     stepwise=True) # Use stepwise algorithm

    print(auto_arima_model.summary())
    best_order = auto_arima_model.order # Store the best order found

    # Forecast with Auto ARIMA
    auto_arima_pred = auto_arima_model.predict(n_periods=len(test_ts))
    auto_arima_pred = pd.Series(auto_arima_pred, index=test_ts.index)

    # Plot forecast vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(train_ts.index, train_ts, label='Train')
    plt.plot(test_ts.index, test_ts, label='Test')
    plt.plot(test_ts.index, auto_arima_pred, label='Auto ARIMA Forecast')
    plt.title('Auto ARIMA Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/plot_10_auto_arima_forecast.png")
    plt.close()
    print("Saved plot: plot_10_auto_arima_forecast.png")

    # Performance Metrics
    auto_arima_rmse = np.sqrt(mean_squared_error(test_ts, auto_arima_pred))
    auto_arima_mae = mean_absolute_error(test_ts, auto_arima_pred)
    print(f"Auto ARIMA RMSE: {auto_arima_rmse:.4f}")
    print(f"Auto ARIMA MAE: {auto_arima_mae:.4f}")

except Exception as e:
    print(f"Error fitting Auto ARIMA: {e}")
    # Use default order if auto_arima fails
    best_order = (1, 1, 1) # Fallback if auto arima fails
    print(f"Falling back to order {best_order} for SARIMAX due to Auto ARIMA error.")

# --- 4. SARIMAX Model (Example with Exogenous Variable) --- 
print("\n--- 4. Fitting SARIMAX Model (with Exogenous Variable) ---")
# Using orders from Auto ARIMA (or fallback), add Volume as exogenous variable
# Note: Seasonality (P,D,Q,m) is set to 0 here as auto_arima found none.

try:
    # Ensure exog variables are numpy arrays
    train_exog_np = train_exog.values.reshape(-1, 1)
    test_exog_np = test_exog.values.reshape(-1, 1)

    sarimax_model = SARIMAX(train_ts, 
                            exog=train_exog_np, 
                            order=best_order, 
                            seasonal_order=(0, 0, 0, 0), # No seasonality assumed here
                            enforce_stationarity=False, 
                            enforce_invertibility=False)
    sarimax_fit = sarimax_model.fit(disp=False)
    print(sarimax_fit.summary())

    # Forecast with exogenous variables
    sarimax_pred = sarimax_fit.predict(start=len(train_ts), end=len(ts)-1, exog=test_exog_np)

    # Plot forecast vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(train_ts.index, train_ts, label='Train')
    plt.plot(test_ts.index, test_ts, label='Test')
    plt.plot(test_ts.index, sarimax_pred, label='SARIMAX Forecast')
    plt.title(f'SARIMAX{best_order} Forecast vs Actuals (Exog: Volume)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/plot_11_sarimax_forecast.png")
    plt.close()
    print("Saved plot: plot_11_sarimax_forecast.png")

    # Performance Metrics
    sarimax_rmse = np.sqrt(mean_squared_error(test_ts, sarimax_pred))
    sarimax_mae = mean_absolute_error(test_ts, sarimax_pred)
    print(f"SARIMAX{best_order} RMSE: {sarimax_rmse:.4f}")
    print(f"SARIMAX{best_order} MAE: {sarimax_mae:.4f}")

except Exception as e:
    print(f"Error fitting SARIMAX: {e}")

# --- 5. ARCH/GARCH Model for Volatility --- 
print("\n--- 5. Fitting GARCH Model on Log Returns ---")
# Model volatility of log returns
# Common choice: GARCH(1,1)
try:
    garch_model = arch_model(train_log_returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off') # Turn off verbose fitting output
    print(garch_fit.summary())

    # Plot conditional volatility
    plt.figure(figsize=(12, 6))
    plt.plot(garch_fit.conditional_volatility, label='Conditional Volatility')
    plt.title('GARCH(1,1) Conditional Volatility of AAPL Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/plot_12_garch_volatility.png")
    plt.close()
    print("Saved plot: plot_12_garch_volatility.png")
except Exception as e:
    print(f"Error fitting GARCH: {e}")

# --- 6. Model Diagnostics (Example: Auto ARIMA) --- 
print("\n--- 6. Diagnostic Checks (Example: Auto ARIMA) ---")
# Check if auto_arima_model exists and has residuals
if 'auto_arima_model' in locals() and hasattr(auto_arima_model, 'resid'):
    residuals = auto_arima_model.resid()
    
    # Plot Residuals
    plt.figure(figsize=(12, 4))
    plt.plot(residuals)
    plt.title('Auto ARIMA Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual Value')
    plt.grid(True)
    plt.savefig("/home/ubuntu/plot_13_auto_arima_residuals.png")
    plt.close()
    print("Saved plot: plot_13_auto_arima_residuals.png")

    # ACF/PACF of Residuals
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(residuals, ax=axes[0], lags=40, title='ACF of Residuals')
    plot_pacf(residuals, ax=axes[1], lags=40, title='PACF of Residuals')
    plt.tight_layout()
    plt.savefig("/home/ubuntu/plot_14_residual_acf_pacf.png")
    plt.close()
    print("Saved plot: plot_14_residual_acf_pacf.png")
    print("Ideally, ACF/PACF of residuals should show no significant spikes.")

    # Ljung-Box Test
    try:
        ljung_box_result = acorr_ljungbox(residuals, lags=[20], return_df=True)
        print("\nLjung-Box Test on Residuals:")
        print(ljung_box_result)
        print("If p-value > 0.05, we fail to reject H0 (residuals are independent/white noise).")
    except Exception as e:
        print(f"Could not perform Ljung-Box test: {e}")
else:
    print("Skipping Auto ARIMA diagnostics as the model did not fit successfully.")

# --- 7. Model Comparison (Metrics) --- 
print("\n--- 7. Model Performance Comparison (Test Set) ---")
print(f"ARIMA(1,1,1)   RMSE: {arima_rmse:.4f}, MAE: {arima_mae:.4f}")
print(f"Auto ARIMA     RMSE: {auto_arima_rmse:.4f}, MAE: {auto_arima_mae:.4f}")
print(f"SARIMAX{best_order}    RMSE: {sarimax_rmse:.4f}, MAE: {sarimax_mae:.4f}")
print("Note: Lower RMSE/MAE indicates better forecast accuracy on the test set.")

print("\nClass 2 Demonstrations Complete.")

