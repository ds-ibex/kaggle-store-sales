import numpy as np
import pandas as pd
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
  
 
def transform_daily_sales_predictions(pred_df: pd.DataFrame, train: pd.DataFrame, cols=['day_of_week'], target='sales'):
    all_cols = cols + ['store_nbr', 'family']
    assert all(col in train.columns for col in cols), 'Error transform_daily_sales_predictions() - not all cols in train df'
    train[target] = train[target].astype('float64')
    train_grouped = train.groupby(cols)[target].sum().reset_index()
    # calculate percent of target, controlled for the first column
    train_grouped[f'pct_{target}'] = train_grouped[target] / train_grouped.groupby(cols)[target].transform(sum)
    train_grouped = train_grouped.drop(columns=[target])    
    # merge percentages with the predicted values
    pred_df = pd.merge(pred_df, train_grouped, on=cols)
    assert all(pred_df.groupby('date')[f'pct_{target}'].sum() == 1.0)
    # scale predicted daily values by their percentage of daily 
    pred_df[f'transformed_{target}'] = pred_df[f'pred_{target}'].mul(pred_df[f'pct_{target}'])
    return pred_df


## Leo's code - Potential duplicate of above
def rmsle_func(y_true, y_pred):
    # Define your RMSLE calculation here
    # For example:
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

def rmsle_lgbm(y_pred, data):
    y_true = np.array(data.get_label())
    for i in range(len(y_pred)):
        y_pred[i]=max(0,y_pred[i])
    score = rmsle_func(y_true, y_pred)
    return 'rmsle', score, False