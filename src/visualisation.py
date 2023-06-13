import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
# import from model_evaluation.py, need to include the 'src.' so that the interpreter knows where to read from if this is called from a notebook
from src.model_evaluation import calc_root_mean_squared_error


def plot_heatmap(df: pd.DataFrame, rows: str, cols: str, values='sales', normalize_rows=True, decimals=2) -> None:
    """Plot a heatmap of a dataframe 
    
    @Griffin complete this docstring

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
    

def plot_time_series_preds(time_series: pd.DataFrame, preds: list, col: str) -> None:
    """_summary_

    @Griffin complete this docstring

    Args:
        time_series (pd.DataFrame): _description_
        preds (list): _description_
        col (str): _description_
    """
    fig = go.Figure()
    fig.add_scatter(x=time_series['date'], y=time_series[col].values, name=f'{col} True')
    fig.add_scatter(x=time_series['date'], y=preds, name=f'{col} Pred')
    fig.update_layout(title=f'Time series {col} predictions (RMSE: {calc_root_mean_squared_error(time_series[col].values, preds):,.2f})')
    fig.show()
    

def plot_rolling_average_stdev(time_series: pd.DataFrame, col='sales', windowsize=7) -> None:
    """_summary_

    @Griffin complete this docstring

    Args:
        time_series (pd.DataFrame): _description_
        col (str, optional): _description_. Defaults to 'sales'.
        windowsize (int, optional): _description_. Defaults to 7.
    """
    df = time_series.copy()
    # Calculate rolling average and rolling standard deviation
    df['rolling_average'] = df[col].rolling(windowsize).mean()
    df['rolling_std'] = df[col].rolling(windowsize).std()

    # Create the line plot with rolling average and rolling standard deviation
    fig = go.Figure()
    fig.add_scatter(x=df['date'], y=df['rolling_average'], name='Rolling Average')
    fig.add_scatter(x=df['date'], y=df['rolling_std'], name='Rolling St. Dev')
    fig.update_layout(title='Rolling Average and Standard Deviation')
    fig.show()


def plot_sales_by(df: pd.DataFrame, col: str) -> None:
    """Plot a bar chart of sales grouped by the col.

    @Griffin complete this docstring

    Args:
        df (pd.DataFrame): _description_
        col (str): _description_
    """
    # group the data by the column and sum the sales
    grouped = df.groupby(col)['sales'].sum().reset_index().sort_values(by='sales', ascending=False)
    # turn the column into a string type so that it will get treated as categorical
    grouped[col] = grouped[col].astype('str')
    # create a bar chart from the grouped data
    fig = px.bar(grouped, x=col, y='sales')
    fig.show()
    

def seaborn_plot_sales_by(df: pd.DataFrame, col: str) -> None:
    """Plot a bar chart of sales grouped by the col.
    
    This is an alternate method in case you would rather use seaborn than plotly

    @Griffin complete this docstring

    Args:
        df (pd.DataFrame): _description_
        col (str): _description_
    """
    grouped = df.groupby('family')['sales'].sum().reset_index().sort_values(by='sales',ascending=False)
    # Create a bar chart using seaborn
    sns.barplot(x=col, y='sales', data=grouped)

    # Set the chart title and labels
    plt.title(f'Total Sales by {col}')
    plt.xlabel(col)
    plt.ylabel('Sales')

    # Display the chart
    plt.show()

def generate_interactive_treemap(df,top):
    """Plot a treemap chart of sales grouped by the store_nbr,family,month.
    Create a hierarchical datafram to use in the treemap

    Args:
        df (pd.DataFrame): _description_
        top (int): How much stores you want to put
    """
    top = min(top, len(df['store_nbr'].unique().tolist()))
    top_store =df.groupby(["store_nbr"]).sales.sum().reset_index()[:top]

    hierarchical_data = df.groupby(['store_nbr', 'family', 'month']).sum().reset_index()
    hierarchical_data=hierarchical_data[['store_nbr', 'family', 'month','sales']]
    hierarchical_data_level_1=df.groupby(['store_nbr', 'family']).sum().reset_index()
    hierarchical_data_level_1=hierarchical_data_level_1[['store_nbr', 'family','sales']]
    hierarchical_data_level_0=df.groupby(['store_nbr']).sum().reset_index()
    hierarchical_data_level_0=hierarchical_data_level_0[['store_nbr', 'sales']]

    hierarchical_data = hierarchical_data[hierarchical_data['store_nbr'].isin(top_store['store_nbr'])]
    hierarchical_data_level_1 = hierarchical_data_level_1[hierarchical_data_level_1['store_nbr'].isin(top_store['store_nbr'])]
    hierarchical_data_level_0 = hierarchical_data_level_0[hierarchical_data_level_0['store_nbr'].isin(top_store['store_nbr'])]

    hierarchical_data['total_sales']=hierarchical_data.groupby(['store_nbr', 'family'])['sales'].transform('sum')
    hierarchical_data_level_1['total_sales']=hierarchical_data_level_1.groupby(['store_nbr'])['sales'].transform('sum')
    hierarchical_data_level_0['total_sales']=hierarchical_data_level_0['sales'].sum()

    # Creating the hierarchical dataframe
    df_hierarchy = pd.DataFrame(columns=['ids', 'labels', 'parents', 'sales','sales_perc'])

    df_hierarchy = df_hierarchy.append({
            'ids': 'Portfolio',
            'labels': 'Portfolio'
        }, ignore_index=True)

    for index, row in hierarchical_data_level_0.iterrows():
        store_nbr = int(row['store_nbr'])
        sales = row['sales']
        sales_perc =  round(row['sales']/row['total_sales'],4)*100
        
        # Constructing the id, labels, and parents values
        id_value = f"{store_nbr}"
        labels_value = f"Store {store_nbr}"
        parents_value = 'Portfolio'
        
        # Appending the row to the hierarchical dataframe
        df_hierarchy = df_hierarchy.append({
            'ids': id_value,
            'labels': labels_value,
            'parents': parents_value,
            'sales': sales,
            'sales_perc': sales_perc
        }, ignore_index=True)


    for index, row in hierarchical_data_level_1.iterrows():
        store_nbr = int(row['store_nbr'])
        family = row['family']
        sales = row['sales']
        sales_perc = round(row['sales']/row['total_sales'],4)*100
        
        # Constructing the id, labels, and parents values
        id_value = f"{store_nbr}-{family}"
        labels_value = f"{family}"
        parents_value = f"{store_nbr}"
        
        # Appending the row to the hierarchical dataframe
        df_hierarchy = df_hierarchy.append({
            'ids': id_value,
            'labels': labels_value,
            'parents': parents_value,
            'sales': sales,
            'sales_perc': sales_perc
        }, ignore_index=True)

    # Iterating over the rows of the hierarchical data
    for index, row in hierarchical_data.iterrows():
        store_nbr = int(row['store_nbr'])
        family = row['family']
        month = row['month']
        sales = row['sales']
        try:
            sales_perc = round(row['sales']/row['total_sales'],4)*100
        except:
            sales_perc =0
        
        # Constructing the id, labels, and parents values
        id_value = f"{store_nbr}-{family}-{month}"
        labels_value = f"Month {month}"
        parents_value = f"{store_nbr}-{family}"
        
        # Appending the row to the hierarchical dataframe
        df_hierarchy = df_hierarchy.append({
            'ids': id_value,
            'labels': labels_value,
            'parents': parents_value,
            'sales': sales,
            'sales_perc': sales_perc
        }, ignore_index=True)


    fig = go.Figure()

    fig.add_trace(go.Treemap(
        ids = df_hierarchy.ids,
        labels = df_hierarchy.labels,
        parents = df_hierarchy.parents,
        maxdepth=2,
        root_color="lightgrey",
        values=df_hierarchy.sales,
        textinfo='label+value+percent parent',
        branchvalues="total",
        marker_colorscale = 'Blues'

    ))

    fig.update_layout(
        uniformtext=dict(minsize=10, mode='hide'),
        margin = dict(t=50, l=25, r=25, b=25)
    )

    fig.show()