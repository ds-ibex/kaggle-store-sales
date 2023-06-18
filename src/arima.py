# Standard Imports
import numpy as np
import pandas as pd

# ARIMA Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# visualisation
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from src.data_setup import create_day_of_week
from src.visualisation import plot_time_series_preds
from src.model_evaluation import eval_hypothesis_test


class ibex_ARIMA:
    def __init__(self, train: pd.DataFrame) -> None:
        """Class implementing an arima model

        Args:
            train (pd.DataFrame): dataframe with training data. Must have both 'date' and 'sales' as columns
        """
        assert all(col in train.columns for col in ['date', 'sales'])
        self.model = None
        self.train = train  # dataframe with training data
        self.train_daily_sales = train.groupby('date')['sales'].sum().reset_index()
        self.train_daily_sales['day_of_week'] = create_day_of_week(self.train_daily_sales.date)

        # add differenced sales
        self.train_daily_sales['diff_sales'] = self.train_daily_sales['sales'].diff()
        self.train_daily_sales = self.train_daily_sales.dropna()
        self.n = self.train_daily_sales.shape[0]


    def plot_autocorrelation(self, plot_size=(10, 6), lags=30) -> None:
        # plot the autocorrelation 
        my_plot_size = (12, 8)

        plot_cols = ['sales']
        if 'diff_sales' in self.train_daily_sales.columns:
            plot_cols.append('diff_sales')

        for col in plot_cols:
            sales_series = self.train_daily_sales[col]
            # plot the autocorrelation
            _, ax = plt.subplots(figsize=my_plot_size)
            plot_acf(sales_series, lags=lags, ax=ax, title=f'{col} Autocorrelation ({sales_series.autocorr():.4f})')
            # plot the partial autocorrelation
            _, ax = plt.subplots(figsize=my_plot_size)
            plot_pacf(sales_series, lags=lags, method='ywm', ax=ax, title=f'{col} Partial Autocorrelation')   # added method to get rid of a future warning


    def test_stationarity(self, col='sales', plot_rolling_average=True, window_size=7) -> None:
        """_summary_

        Args:
            col (str, optional): _description_. Defaults to 'sales'.
            plot_rolling_average (bool, optional): _description_. Defaults to True.
            window_size (int, optional): _description_. Defaults to 7.
        """
        stationarity_hypotheses = [
            f'{col} data is not stationary',   # null hypothesis
            f'{col} data is stationary'        # alternate hypothesis
        ]
        # Dickey-Fuller Test for Stationarity
        dickey_fuller_result = adfuller(self.train_daily_sales[col])
        reject_h0 = eval_hypothesis_test(stationarity_hypotheses, dickey_fuller_result[1])

        # plot the rolling average
        if plot_rolling_average:
            tmp = self.train_daily_sales.copy()
            tmp[f'{col} Rolling Average'] = tmp[col].rolling(7).mean()
            fig = px.line(tmp, x='date', y=f'{col} Rolling Average')
            fig.update_layout(title=f'{col} {window_size} Day Rolling Average - Stationary: {reject_h0}')
            fig.show()
        # if we fail to reject the null hypothesis that the data is not stationary, try the test again with the differenced sales
        if not reject_h0:
            self.test_stationarity('diff_sales', plot_rolling_average, window_size)


    def fit(self, pdq_order=(1, 0, 0), seasonal_order=None, col='diff_sales', plot_summary=True):
        if seasonal_order is not None:
            model = SARIMAX(self.train_daily_sales[col].values, order=pdq_order, seasonal_order=seasonal_order)
            self.model = model.fit(disp=0)
        else:
            model = ARIMA(self.train_daily_sales[col].values, order=pdq_order)
            self.model = model.fit()

        if plot_summary:
            print(self.model.summary())
            self.model.plot_diagnostics(figsize=(16,10))

        # predict values on training data
        self.train_daily_sales[f'pred_{col}'] = self.model.predict(start=0, end=self.n-1)
        # undo differencing
        if col == 'diff_sales':
            self.train_daily_sales['pred_sales'] = self.train_daily_sales[f'pred_diff_sales'].cumsum()# + self.train_daily_sales['sales'].iloc[0]
        
        plot_time_series_preds(self.train_daily_sales['date'], preds=[self.train_daily_sales['diff_sales'], self.train_daily_sales['pred_diff_sales']], col='diff_sales')


    def evaluate(self, validation=None) -> list:
        
        # start_date = self.train_daily_sales['date'].iloc[-1] + pd.DateOffset(days=1)
        # pred_daily_sales = pd.DataFrame({
        #     'date': pd.date_range(start=start_date, periods=steps),
        #     'pred_diff_sales': pred_diff_sales,
        # })
        steps = len(validation)

        validation['day_of_week'] = create_day_of_week(validation['date'])
        validation['pred_diff_sales'] = self.model.predict(start=self.n, end=self.n + steps -1)
        validation['pred_sales'] = validation['pred_diff_sales'].cumsum() + self.train_daily_sales['sales'].iloc[-1]
        plot_time_series_preds(validation['date'], preds=[validation['sales'], validation['pred_sales']], col='sales')
        
        return validation

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