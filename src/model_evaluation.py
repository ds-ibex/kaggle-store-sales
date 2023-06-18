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


def eval_hypothesis_test(hypotheses: list, p_value: float, alpha=0.05) -> bool:
    """ Evaluate a hypothesis test

    Args:
        hypotheses (list): [null hypothesis as a string, alternate hypothesis as a string]
        p_value (float): p_value that is a result of the test
        alpha (float, optional): confidence level that you require p to be less than to reject the null. Defaults to 0.05.

    Returns:
        bool: whether we reject the null hypothesis
    """
    if p_value < alpha:
        print(f'Reject the null hypothesis, accept alternate hypothesis: "{hypotheses[1]}" (p-value: {p_value:.4f})')
        return True
    else:
        print(f'Fail to reject the null hypothesis: "{hypotheses[0]}" (p-value: {p_value:.4f})')
        return False
        

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