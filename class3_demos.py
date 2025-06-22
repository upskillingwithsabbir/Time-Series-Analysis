# Class 3: Time Series Analysis with ML Approach - Python Demonstrations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore") # Ignore harmless warnings

# --- 1. Load Data and Prepare --- 
print("--- 1. Loading Data ---")
data_file = "/home/ubuntu/aapl_stock_data_10y.csv"
df = pd.read_csv(data_file, index_col='Date', parse_dates=True)

# Use Adjusted Close price
ts = df['Adj Close'].copy()

# Split data: Train (first 9 years), Test (last 1 year approx)
# ~252 trading days per year
train_size = len(ts) - 252
train_ts, test_ts = ts[:train_size], ts[train_size:]

print(f"Train set size: {len(train_ts)}")
print(f"Test set size: {len(test_ts)}")

# Initialize variables for metrics
prophet_rmse, prophet_mae = np.nan, np.nan
xgb_rmse, xgb_mae = np.nan, np.nan

# --- 2. Facebook Prophet --- 
print("\n--- 2. Fitting Prophet Model ---")

# Prepare data for Prophet (requires columns 'ds' and 'y')
prophet_train_df = train_ts.reset_index()
prophet_train_df.columns = ['ds', 'y']

try:
    # Instantiate and fit Prophet model
    # Prophet automatically detects trend changes and seasonality
    prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, 
                            changepoint_prior_scale=0.05) # Default is 0.05
    prophet_model.fit(prophet_train_df)

    # Create future dataframe for predictions
    future_dates = prophet_model.make_future_dataframe(periods=len(test_ts), freq='B') # 'B' for business day frequency
    # Filter future_dates to match the test set index exactly
    future_dates = future_dates[future_dates['ds'].isin(test_ts.index)]

    # Make predictions
    prophet_forecast = prophet_model.predict(future_dates)

    # Extract prediction ('yhat')
    prophet_pred = prophet_forecast['yhat'].values
    prophet_pred = pd.Series(prophet_pred, index=test_ts.index)

    # Plot forecast vs actual
    fig = prophet_model.plot(prophet_forecast)
    plt.plot(test_ts.index, test_ts, '.r', label='Actual Test Data')
    plt.title('Prophet Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/plot_15_prophet_forecast.png")
    plt.close(fig) # Close the figure created by prophet
    print("Saved plot: plot_15_prophet_forecast.png")

    # Plot components (optional, good for lecture)
    fig_comp = prophet_model.plot_components(prophet_forecast)
    plt.savefig("/home/ubuntu/plot_16_prophet_components.png")
    plt.close(fig_comp)
    print("Saved plot: plot_16_prophet_components.png")

    # Performance Metrics
    prophet_rmse = np.sqrt(mean_squared_error(test_ts, prophet_pred))
    prophet_mae = mean_absolute_error(test_ts, prophet_pred)
    print(f"Prophet RMSE: {prophet_rmse:.4f}")
    print(f"Prophet MAE: {prophet_mae:.4f}")

except Exception as e:
    print(f"Error fitting Prophet: {e}")

# --- 3. XGBoost --- 
print("\n--- 3. Fitting XGBoost Model ---")

# Feature Engineering for XGBoost
def create_features(df, label=None):
    """ Creates time series features from datetime index. """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour # Will be 0 for daily data
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Add Lag features
    for lag in [1, 5, 10, 21]: # Lag by 1 day, 1 week, 2 weeks, 1 month (approx)
        df[f'lag_{lag}'] = df['Adj Close'].shift(lag)
        
    # Add Rolling Mean features
    for window in [5, 21]: # Rolling mean over 1 week, 1 month
        df[f'rolling_mean_{window}'] = df['Adj Close'].shift(1).rolling(window=window).mean()

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
              'dayofyear', 'dayofmonth', 'weekofyear'] + [f'lag_{lag}' for lag in [1, 5, 10, 21]] + \
             [f'rolling_mean_{window}' for window in [5, 21]]]
    if label:
        y = df[label]
        return X, y
    return X

# Create features for the entire dataset first to handle lags correctly
df_with_features = df.copy()
df_with_features['Adj Close'] = ts # Ensure the target column exists
X_all, y_all = create_features(df_with_features, label='Adj Close')

# Split features into train/test based on original index
X_train, y_train = X_all.loc[train_ts.index], y_all.loc[train_ts.index]
X_test, y_test = X_all.loc[test_ts.index], y_all.loc[test_ts.index]

# Drop rows with NaNs created by lag/rolling features (mostly at the beginning of train set)
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

# Check if test set has NaNs (shouldn't if lags are smaller than test set size)
if X_test.isnull().values.any():
    print("Warning: NaNs found in X_test, potentially due to lag features. Dropping NaNs.")
    test_nan_indices = X_test[X_test.isnull().any(axis=1)].index
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]
    # Adjust original test_ts to match the rows kept in X_test/y_test
    test_ts = test_ts.drop(test_nan_indices)
    print(f"Adjusted test set size after dropping NaNs: {len(test_ts)}")

try:
    # Instantiate and fit XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000, # Number of boosting rounds
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50, # Stop if validation score doesn't improve
        n_jobs=-1 # Use all available CPU cores
    )

    # Use last part of training set as validation for early stopping
    # Ensure validation set size matches test set size for consistency if possible
    val_size = len(y_test) # Use size of potentially reduced test set
    X_train_part, X_val_part = X_train[:-val_size], X_train[-val_size:]
    y_train_part, y_val_part = y_train[:-val_size], y_train[-val_size:]

    xgb_model.fit(X_train_part, y_train_part, 
                  eval_set=[(X_val_part, y_val_part)], 
                  verbose=False) # Set verbose=True to see training progress

    # Make predictions
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred = pd.Series(xgb_pred, index=y_test.index) # Use y_test index which might have dropped NaNs

    # Plot forecast vs actual
    plt.figure(figsize=(12, 6))
    # Plot original train/test for context, using potentially adjusted test_ts
    plt.plot(train_ts.index, train_ts, label='Train') 
    plt.plot(test_ts.index, test_ts, label='Test') 
    plt.plot(xgb_pred.index, xgb_pred, label='XGBoost Forecast', alpha=0.8)
    plt.title('XGBoost Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/plot_17_xgboost_forecast.png")
    plt.close()
    print("Saved plot: plot_17_xgboost_forecast.png")

    # Performance Metrics
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    print(f"XGBoost MAE: {xgb_mae:.4f}")

except Exception as e:
    print(f"Error fitting XGBoost: {e}")

# --- 4. Comparison (Metrics) --- 
print("\n--- 4. ML Model Performance Comparison (Test Set) ---")
print(f"Prophet RMSE: {prophet_rmse:.4f}, MAE: {prophet_mae:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}")
print("Compare these metrics with those from Class 2 (Statistical Models).")

print("\nClass 3 Demonstrations Complete.")

