import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

def model_eval_pipeline(y_true, y_pred):
    """Evaluate the performance of the predictions of a model

    Args:
        y_true (list): ground truth values
        y_pred (list): predicted values

    Returns:
        tuple: scores in the tuple (mae, mse, rmse, rmsle, r2)
    """
    metrics = {}
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['rmsle'] = np.sqrt(mean_squared_log_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)
    return metrics
