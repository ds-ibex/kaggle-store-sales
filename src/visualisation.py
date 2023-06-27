import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
# import from model_evaluation.py, need to include the 'src.' so that the interpreter knows where to read from if this is called from a notebook
from src.model_evaluation import calc_root_mean_squared_error
from sklearn.preprocessing import LabelEncoder


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
    

def plot_time_series_preds(date_series, preds: list, names=['True', 'Pred'], col='Sales') -> None:
    """_summary_

    @Griffin complete this docstring

    Args:
        time_series (pd.DataFrame): _description_
        preds (list): _description_
        col (str): _description_
    """
    # fig = go.Figure()
    # fig.add_scatter(x=time_series['date'], y=time_series[col].values, name=f'{col} True')
    # fig.add_scatter(x=time_series['date'], y=preds, name=f'{col} Pred')
    # fig.update_layout(title=f'Time series {col} predictions (RMSE: {calc_root_mean_squared_error(time_series[col].values, preds):,.2f})')
    # fig.show()
    fig = go.Figure()
    for pred, name in zip(preds, names):
        fig.add_scatter(x=date_series, y=pred, name=f'{col} {name}')
    title_str = f'Time series {col.capitalize()} predictions'
    if len(preds) == 2:
        title_str += f' (RMSE: {calc_root_mean_squared_error(preds[0], preds[1]):,.2f})'
    fig.update_layout(title=title_str)
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
    fig = px.bar(grouped, x=col, y='sales', color=col)
    fig.update_layout(title=f'Sales by {col}')
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

def build_hierarchical_dataframe(df:pd.DataFrame, levels, value_column, color_columns=None):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_all_trees = pd.DataFrame(columns=['id', 'label','parent', 'value', 'color'])
    for i, level in enumerate(levels):
        dfg = df.groupby(levels[i:]).sum()
        try:
            dfg_inv=df.groupby(levels[i+1:]).sum()
            dfg_inv=dfg_inv.drop(columns=['month'])
            dfg_inv=dfg_inv.rename(columns={value_column: 'total'})
            dfg=dfg.join(dfg_inv,on=levels[i+1:],rsuffix='_other')
        except:
            dfg_inv=df[value_column].sum()
            dfg['total']=dfg_inv
        dfg = dfg.reset_index()
        for index, row in dfg.iterrows():
            if i < len(levels) - 1:
                parent = row[levels[i+1]]
                label= row[levels[i]]
                id = str(row[levels[i]])
                for j in range(i+1,len(levels)):
                    try:
                        parent=str(row[levels[j+1]])+"-"+str(parent)
                    except:
                        parent=parent
                    id=str(row[levels[j]])+"-"+str(id)
            else:
                parent = 'All_Stores'
                try:
                    label= str(int(row[levels[i]]))
                    id = str(int(row[levels[i]]))
                except:
                    label= str(row[levels[i]])
                    id = str(row[levels[i]])
            value = row[value_column]
            try:
                color = row[color_columns] / row['total']
            except:
                color=0
            df_all_trees = df_all_trees.append({
                'id': id,
                'label': label,
                'parent': parent,
                'value': value,
                'color': color
            }, ignore_index=True)
    total = pd.Series(dict(id='All_Stores', label='All_Stores',
                              value=df[value_column].sum(),
                              color=1))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    df_all_trees['color'].fillna(0,inplace=True)
    df_all_trees=df_all_trees.reindex(index=df_all_trees.index[::-1])
    df_all_trees=df_all_trees.reset_index()
    df_all_trees=df_all_trees.drop(columns=['index'])
    return df_all_trees



def generate_interactive_treemap(df,top,levels,color_columns,value_column,depth,colorscale):
    """
    Plot a treemap chart of sales grouped by the store_nbr,family,month.
    Create a hierarchical datafram to use in the treemap

    Args:
        df (pd.DataFrame): _description_
        top (int): How much stores you want to put
        level (list of string): list of the levels, the first one in the list is the level at the bottom
        color_columns (string): column you want to use as for your color
        value_column (string): column you want to use as for your value and width scale
        depth (int): depth of your tree map
        colorscale (string): Color use for the color scale
    """
    top_20_store =df.groupby(["store_nbr"]).sales.sum().reset_index()[:top]
    hierarchical_data = df.groupby(['store_nbr', 'family', 'month']).sum().reset_index()
    hierarchical_data = hierarchical_data[hierarchical_data['store_nbr'].isin(top_20_store['store_nbr'])]
    hierarchical_data=hierarchical_data[['store_nbr', 'family', 'month','sales']]
    df_all_trees = build_hierarchical_dataframe(hierarchical_data, levels, value_column,color_columns)
    max_score = df_all_trees['color'].max()
    min_score = df_all_trees['color'].min()
    fig2 = go.Figure()

    fig2.add_trace(go.Treemap(
        ids = df_all_trees.id,
        labels = df_all_trees.label,
        parents = df_all_trees.parent,
        maxdepth=depth,
        root_color="grey",
        values=df_all_trees.value,
        #textinfo='label+value+percent parent',
        branchvalues="total",
        textinfo='label+value+percent parent',
        hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Percentage of Sales: %{color:.2%}',
        marker=dict(
            colors=df_all_trees['color'],
            colorscale=colorscale,
            cmin=min_score,
            cmax=max_score*0.75,
            showscale=True)

    ))

    fig2.update_layout(
        uniformtext=dict(minsize=10, mode='hide'),
        margin = dict(t=50, l=25, r=25, b=25)
    )

    fig2.show()

def comparison_val_pred(train,df_validation, pred, mode, dim):
    """
    Plot an histogram to compare your validation data and your predictions.

    Args:
        training (pd.Dataframe): initial training data
        df_validation (pd.Dataframe): validation data
        pred (list): list of prediction from your model
        mode (String): values or percentages to show. Need to be either 'value' or 'percentage'
        dim (String): group by dimension
    """
    # Check if the mode is the right one

    if mode in ['value', 'percentage']:
        #Create base data
        val=df_validation.rename(columns={'target':'sales'})
        Prediction = val.copy()
        Prediction['sales']=pred
        family_list =train['family'].unique()
        fam_le = pd.DataFrame(family_list, columns=['family'])
        le=LabelEncoder()
        fam_le['family_le']=le.fit_transform(fam_le['family'])     

        ## Create data to use for Viz
        merged_data = pd.concat([val,Prediction], axis=0,keys=['Validation','pred'])
        merged_data = merged_data.reset_index().rename(columns={'level_0': 'Dataset','level_1': 'id'})
        merged_data=merged_data.merge(fam_le, left_on='family', right_on='family_le')
        merged_data=merged_data.drop(columns=['family_x','family_le'])
        merged_data=merged_data.rename(columns={'family_y':'family'})
        data =merged_data.groupby(['Dataset',dim]).sum().reset_index()
        data =data [['Dataset',dim,'sales']]
        if dim == 'store_nbr':
            data.astype({'store_nbr':str})

        # Define colors for each category
        colors = {'Validation': 'cornflowerblue', 'pred': 'coral'}

        # Create the grouped bar chart
        fig = go.Figure()

        if mode == 'value':
        
            for category in data['Dataset'].unique():
                category_data = data[data['Dataset'] == category]
                fig.add_trace(go.Bar(
                    x=category_data[dim],
                    y=category_data['sales'],
                    name=category,
                    marker=dict(color=colors[category])
                ))

            # Customize the layout
            fig.update_layout(
                title=f"Sales by {dim} and Model",
                xaxis_title=f"{dim}",
                yaxis_title="Sales",
                barmode='group'
            )
            
        elif mode == 'percentage':
            # Create a pivot table to calculate the sales for each category within each family
            pivot_table = data .pivot_table(values='sales', index=dim, columns='Dataset', aggfunc='sum')

            # Calculate the percentage of sales compared to category A within each family
            for col in pivot_table.columns:
                if col!='Validation':
                    pivot_table[col] = pivot_table[col] / pivot_table['Validation']
            pivot_table['Validation']=1
            for category in pivot_table.columns:
                fig.add_trace(go.Bar(
                    x=pivot_table.index,
                    y=pivot_table[category], 
                    name=f"{category}",
                    marker=dict(color=colors[category])
                ))
            # Customize the layout
            fig.update_layout(
                title=f"Sales percentage by {dim} and Model",
                xaxis_title=f"{dim}",
                yaxis_title="Sales",
                barmode='group'
            )
            # Format the y-axis tick labels as percentages
            fig.update_yaxes(tickformat=".1%")

        # Show the plot
        fig.show()
    else:
        return("Please, select the correct mode value. Either 'value' or 'percentage'.")
    