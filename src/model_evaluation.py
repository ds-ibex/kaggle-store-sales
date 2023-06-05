import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


def calc_root_mean_squared_error(y_true, y_pred):
    """Calculate the root mean squared error for the predicted values

    Args:
        y_true (list): ground truth values
        y_pred (list): predicted values

    Returns:
        float: rmse score for the predictions
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calc_root_mean_squared_log_error(y_true, y_pred):
    """Calculate the root mean squared log error for the predicted values

    Args:
        y_true (list): ground truth values
        y_pred (list): predicted values

    Returns:
        float: rmsle score for the predictions
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def model_eval_pipeline(y_true, y_pred):
    """Evaluate the performance of the predictions of a model

    Args:
        y_true (list): ground truth values
        y_pred (list): predicted values

    Returns:
        dict: scores in the tuple (mae, mse, rmse, rmsle, r2)
    """
    metrics = {
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
        'rmse': calc_root_mean_squared_error,
        'rmsle': calc_root_mean_squared_log_error,
        'r2': r2_score,
    }
    return {metric: metric_function(y_true, y_pred) for metric, metric_function in metrics.items()}