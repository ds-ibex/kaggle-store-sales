import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.model_evaluation import calc_root_mean_squared_error

def plot_heatmap(df: pd.DataFrame, rows: str, cols: str, values='sales', normalize_rows=True, decimals=2) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        rows (str): _description_
        cols (str): _description_
        values (str, optional): _description_. Defaults to 'sales'.
        normalize_rows (bool, optional): _description_. Defaults to True.
        decimals (int, optional): _description_. Defaults to 2.
    """
    # heatmap of sales by family by day of the week, normalized by row
    heatmap_df = df.pivot_table(values=values, index=rows, columns=cols)
    if normalize_rows:
        # normalize across the rows
        heatmap_df = heatmap_df.div(heatmap_df.sum(axis=1), axis=0).round(decimals=decimals)
    fig = px.imshow(heatmap_df, text_auto=True, aspect='auto', height=(50 * heatmap_df.shape[0]))
    if cols == 'day_of_week':
        # Set xticks
        fig.update_layout(
            xaxis=dict(
                tickvals=list(range(1, 8)),  # Specify tick values
                ticktext=['mon', 'tues', 'weds', 'thurs', 'fri', 'sat', 'sun']  # Specify tick labels
            )
        )
    title_str = f'{values.capitalize()} by {rows.capitalize()} and {cols.capitalize()}'
    if normalize_rows:
        title_str = 'Normalized ' + title_str
    fig.update_layout(title=title_str)
    fig.show()
    

def plot_time_series_preds(time_series, preds, col):
    fig = go.Figure()
    fig.add_scatter(x=time_series['date'], y=time_series[col].values, name=f'{col} True')
    fig.add_scatter(x=time_series['date'], y=preds, name=f'{col} Pred')
    fig.update_layout(title=f'Time series {col} predictions (RMSE: {calc_root_mean_squared_error(time_series[col].values, preds):,.2f})')
    fig.show()