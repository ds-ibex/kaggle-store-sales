# Standard Imports
import importlib
import os
import numpy as np
import pandas as pd
import sys

# ARIMA Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.visualisation import plot_time_series_preds


def arima_trial(X_train, X_valid=None, predict_steps=0, pdq_order=(1, 0, 0), seasonal_order=None, plots=False, date_series=None, show_summary=False):

    if seasonal_order is not None:
        model = SARIMAX(X_train, order=pdq_order, seasonal_order=seasonal_order)
        fitted_arima = model.fit(disp=0)
    else:
        model = ARIMA(X_train, order=pdq_order)
        fitted_arima = model.fit()

    if show_summary:
        print(fitted_arima.summary())
        if plots:
            _ = fitted_arima.plot_diagnostics(figsize=(16,10))
    
    n_train = len(X_train)
    n_valid = len(X_valid)
    # predict the training values and validation values
    train_predictions = fitted_arima.predict(start=0, end=n_train-1)
    valid_predictions = fitted_arima.predict(start=n_train, end=n_train+n_valid-1)
    
    valid_start_date = date_series.iloc[-1] + pd.DateOffset(days=1)
    valid_date_series = pd.date_range(start=valid_start_date, periods=n_valid)
    
    if plots and date_series is not None:
        plot_time_series_preds(date_series, preds=[X_train, train_predictions], col='diff_sales')
        plot_time_series_preds(valid_date_series, preds=[X_valid, valid_predictions], col='diff_sales')
        
    res = pd.DataFrame({
        'date': valid_date_series,
        'diff_sales': X_valid,
        'pred_diff_sales': valid_predictions,
    })
    return res