# Class 2: Interpretation of Statistical Model Results

This section provides an interpretation of the results obtained from running the statistical models (ARIMA, Auto ARIMA, SARIMAX, GARCH) on the AAPL adjusted close price data.

## 1. Model Performance Comparison (Test Set Forecasting)

The primary goal of these models was to forecast the adjusted close price for the last year of the data (test set). We evaluated the forecasts using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Lower values indicate better accuracy.

*   **ARIMA(1,1,1):**
    *   RMSE: 35.0100
    *   MAE: 31.5656
*   **Auto ARIMA:**
    *   The `pmdarima.auto_arima` function successfully identified an optimal order (1,1,2) based on the AIC criterion during its search process. However, it encountered an error ("Input contains NaN") during the prediction phase on the test set. This resulted in NaN values for RMSE and MAE. This failure often indicates issues with how the model handles data transformations (like differencing) at the prediction boundary or potential NaNs introduced during feature engineering (though none were explicitly added here). Further investigation would be needed in a real-world scenario (e.g., checking internal steps, trying different `pmdarima` versions or settings).
*   **SARIMAX(1,1,1) (with Volume as Exogenous Variable):**
    *   Since Auto ARIMA failed to provide a reliable forecast, we used the fallback order (1,1,1) for the SARIMAX model, incorporating the daily trading volume as an external predictor.
    *   RMSE: 35.2970
    *   MAE: 31.8694

**Interpretation:**
Comparing the successful models, the simple ARIMA(1,1,1) model performed slightly better than the SARIMAX(1,1,1) model with volume as an exogenous variable on this specific test set, having marginally lower RMSE and MAE. This suggests that, for this particular dataset and model configuration, adding volume did not improve the one-step-ahead forecast accuracy for the price itself. It is important to note that stock price forecasting is notoriously difficult, and these relatively high error values (compared to the price scale) are common. The value of exogenous variables might be more apparent in different contexts or with different modeling approaches.

## 2. Model Summaries Interpretation

*   **ARIMA(1,1,1) Summary:**
    *   The summary showed significant coefficients for both the AR1 (autoregressive term at lag 1) and MA1 (moving average term at lag 1) components (p-values << 0.05). This supports the inclusion of these terms in the model based on the training data.
    *   The `sigma2` value represents the estimated variance of the residuals.
*   **Auto ARIMA Summary:**
    *   The `auto_arima` trace showed the stepwise search process, comparing different model orders based on AIC. It converged to ARIMA(1,1,2) as the best model for the training data (AIC: 9225.391).
    *   The summary for the selected ARIMA(1,1,2) model showed significant coefficients for AR1, MA1, and MA2 terms.
*   **SARIMAX(1,1,1) Summary:**
    *   The coefficient for the exogenous variable `x1` (Volume) was statistically significant (p=0.005). This indicates that, within the training data, volume had a statistically significant relationship with the differenced adjusted close price, even though it didn't improve the test set forecast accuracy in this instance. The coefficient was negative, suggesting a slight inverse relationship in this specific model context.
    *   AR1 and MA1 terms were also highly significant, similar to the ARIMA model.
    *   A warning about the covariance matrix being singular or near-singular suggests potential multicollinearity or numerical instability issues, which might warrant further investigation or model simplification.
*   **GARCH(1,1) Summary (on Log Returns):**
    *   The model was fitted to the log returns to analyze volatility.
    *   The `mu` (mean) coefficient was significant but small, indicating a slight positive drift in daily log returns.
    *   The volatility parameters `alpha[1]` (ARCH term) and `beta[1]` (GARCH term) were both highly significant (p-values << 0.05).
    *   The sum of `alpha[1]` (0.10) and `beta[1]` (0.88) is close to 1 (0.98). This indicates high persistence in volatility – meaning that periods of high volatility tend to be followed by more high volatility, and periods of low volatility by low volatility (volatility clustering), a well-known characteristic of financial time series.
    *   The plot `plot_12_garch_volatility.png` visually confirms this, showing periods where the estimated conditional volatility spikes and remains elevated.

## 3. Diagnostic Checks (Auto ARIMA Residuals)

We examined the residuals from the fitted Auto ARIMA model (ARIMA(1,1,2)) to assess its adequacy on the training data.

*   **Residual Plot (`plot_13_auto_arima_residuals.png`):** The residuals appear somewhat randomly distributed around zero, without obvious trends or strong patterns, which is desirable.
*   **ACF/PACF Plots (`plot_14_residual_acf_pacf.png`):** Ideally, the ACF and PACF plots of the residuals should show no significant spikes outside the confidence bands for all lags greater than 0. While many spikes are within the bands, there might be a few borderline or slightly significant spikes, suggesting some autocorrelation might remain.
*   **Ljung-Box Test:** This test formally checks if the residuals are independently distributed (i.e., resemble white noise). The null hypothesis (H0) is that the residuals are independent.
    *   The test yielded a p-value of approximately 0.011 for lags up to 20.
    *   Since the p-value (0.011) is less than the common significance level of 0.05, we **reject the null hypothesis**. This indicates that there is significant autocorrelation remaining in the residuals, and they do not behave like white noise.

**Interpretation:** The diagnostic checks, particularly the Ljung-Box test, suggest that the Auto ARIMA(1,1,2) model, while being the best fit according to AIC on the training data, did not fully capture all the temporal dependencies. There is still structure left in the residuals. This could mean a more complex model (perhaps higher orders, different differencing, or incorporating seasonality if relevant) might be needed, or that the underlying process has complexities not easily captured by standard ARIMA models.

## 4. Overall Findings & Recommendations for Lecture

*   **Process:** The demonstration successfully walked through loading, splitting, modeling (ARIMA, Auto ARIMA, SARIMAX, GARCH), forecasting, evaluating (RMSE, MAE), and diagnosing time series models.
*   **Stationarity:** The initial ADF test confirmed the non-stationarity of the raw price series, necessitating differencing (d=1) for ARIMA-based models.
*   **Model Selection:** Auto ARIMA is a useful tool for automatically finding potentially good ARIMA orders based on information criteria (AIC/BIC), but it's not foolproof (as seen by the prediction failure and residual issues). Manual inspection of ACF/PACF plots remains important.
*   **Exogenous Variables:** SARIMAX allows incorporating external factors. While Volume was statistically significant in the training fit, it didn't improve test set price forecasts here. The impact of exogenous variables can be highly context-dependent.
*   **Volatility Modeling:** GARCH models are essential for analyzing and forecasting the *volatility* (risk) of financial returns, capturing the common phenomenon of volatility clustering.
*   **Diagnostics:** Residual analysis is crucial. Even if a model looks good on paper (e.g., low AIC) or performs reasonably on forecasts, checking residuals (plots, Ljung-Box test) reveals if the model assumptions are met and if predictable patterns remain.
The failure of the Auto ARIMA residuals to pass the Ljung-Box test is a key teaching point about the limitations of models and the importance of diagnostics.
*   **Forecasting Difficulty:** Emphasize that stock price forecasting is inherently challenging due to market efficiency and noise. Simple models often perform similarly to more complex ones for point forecasts.
*   **Next Steps (in a real project):** Investigate the Auto ARIMA prediction failure. Explore higher-order models or different structures based on residual diagnostics. Consider non-linear models or Machine Learning approaches (Class 3).

