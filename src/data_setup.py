# Standard imports
import gc
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder

# Define gloabl path variables
ROOT_PATH = Path(os.path.dirname(os.getcwd()))
DATA_PATH = ROOT_PATH / 'data'
assert 'raw' in os.listdir(DATA_PATH), 'Data directory not structured properly: kaggle-store-sales/data/raw does not exist, see readme.md for proper structure'
RAW_PATH = DATA_PATH / 'raw'
PROCESSED_PATH = DATA_PATH / 'processed'
SUBMISSION_PATH = DATA_PATH / 'submissions'

# if the processed directory does not exist, create it
if 'processed' not in os.listdir(DATA_PATH):
    print(f'Creating directory: {PROCESSED_PATH}')
    os.mkdir(PROCESSED_PATH)

# if the submissions directory does not exist, create it
if 'submissions' not in os.listdir(DATA_PATH):
    print(f'Creating directory: {SUBMISSION_PATH}')
    os.mkdir(SUBMISSION_PATH)
    

def get_data():
    """Load processed dataframes for train, test, stores, transactions
    
    On first load, loads them from csv files and processed the dataframes to be memory efficient.
    Stores them in pickle files in the processed directory for faster access in the future.

    On later loads, reads the dataframes from processed pickle files. 

    Returns:
        tuple: the four dataframes (train, test, stores, transactions)
    """
    
    fnames = ['train', 'test', 'stores', 'transactions']
    
    # check if the files are in the processed directory
    if all(f'{fname}.pkl' in os.listdir(PROCESSED_PATH) for fname in fnames):
        print('loading pickled dataframes...')
        return tuple(pd.read_pickle(PROCESSED_PATH / f'{fname}.pkl') for fname in fnames)

    print('loading dataframes from csv files...')
    # Read data files into dataframes
    train, test, stores, transactions = tuple(pd.read_csv(RAW_PATH / f'{fname}.csv') for fname in fnames)
    
    # sort transactions by store number and date
    transactions = transactions.sort_values(['store_nbr', 'date'])

    # Convert to more memory efficient datatypes

    # Transactions has a max of 8359, int32 < 4 million
    transactions['transactions'] = transactions['transactions'].astype('int32')
    stores['cluster'] = stores.cluster.astype('int8')

    # Store Number has a max of 54, int8 < 256
    store_nbr_list = [train, test, stores]
    for df in store_nbr_list:
        df['store_nbr'] = df['store_nbr'].astype('int8')
    # python is pass by assignment so pass them back to the original objects
    train, test, stores = store_nbr_list

    # list to process dates
    dfs_with_date = [train, test, transactions]
    
    # add date features
    train, test, transactions = tuple(create_date_features(df) for df in dfs_with_date)

    # smaller floats
    train['onpromotion'] = train.onpromotion.astype('float32')
    test['onpromotion'] = test.onpromotion.astype('float32')
    train['sales'] = train.sales.astype('float32')
    
    # set indexes
    train = train.set_index('id')
    test = test.set_index('id')
    stores = stores.set_index('store_nbr')
    
    dfs = (train, test, stores, transactions)
    
    # store the dataframes as pickle files in the processed directory
    print('pickling data files...')
    for df, fname in zip(dfs, fnames):
        df.to_pickle(PROCESSED_PATH / f'{fname}.pkl')
    
    # return a tuple of the dataframes (train, test, stores, transactions)
    return dfs


def train_val_split(train=None, val_weeks=4):
    """ Split the training data into train and validation data.
        Validation data will be the last val_weeks number of weeks from the training data.

    Args:
        train (df, optional): training data as a dataframe. Defaults to None and loads the dataframe using get_data().
        val_weeks (int, optional): number of weeks to set as validation data. Defaults to 4.

    Returns:
        tuple: train df, validation df
    """
    
    # if train was not passed, take it from the get data function
    if train is None:
        train = get_data()[0]
    
    cutoff = train['date'].max() - pd.DateOffset(weeks=val_weeks)
    
    train_cutoff = train[train['date'] < cutoff]
    val = train[train['date'] >= cutoff]
    return train_cutoff, val


def create_date_features(df):
    """ Create date features in a dataframe

    Args:
        df (dataframe): dataframe to add features to

    Returns:
        dataframe: a copy of the dataframe with the date features
    """
    # turn date column into a datetime object
    df['date'] = pd.to_datetime(df['date']) 
    df['year'] = df.date.dt.isocalendar().year.astype("int32")
    df['month'] = df.date.dt.month.astype("int8")
    df['week'] = df.date.dt.isocalendar().week.astype("int8")
    df['day'] = df.date.dt.dayofyear.astype("int16")
    df['quarter'] = df.date.dt.quarter.astype("int8")
    # day of the week (1 - 7)
    df['day_of_week'] = df.date.dt.isocalendar().day.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['week_of_month'] = ((df['day_of_month']-1) // 7 + 1).astype("int8")
    df['is_weekend'] = (df.date.dt.weekday // 4).astype("int8")
    df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype("int8")
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    # 0: Winter, 1: Spring, 2: Summer, 3: Fall
    df['season'] = np.where(df.month.isin([12,1,2]), 0, 1)
    df['season'] = np.where(df.month.isin([6,7,8]), 2, df['season'])
    df['season'] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df['season'])).astype("int8")
    return df


def process_holiday_events():
    """_summary_

    Returns:
        _type_: _description_
    """
    train, test, stores, transactions = get_data()
    
    #Import holiday data
    holidays = pd.read_csv(RAW_PATH / 'holidays_events.csv')
    holidays["date"] = pd.to_datetime(holidays.date)
    holidays

    # Transferred Holidays
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis = 1).reset_index(drop = True)
    tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis = 1).reset_index(drop = True)
    tr = pd.concat([tr1,tr2], axis = 1)
    tr = tr.iloc[:, [5,1,2,3,4]]
    # TODO @Eoin can you add comments here

    holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis = 1)
    holidays = holidays.append(tr).reset_index(drop = True)

    # Additional Holidays
    holidays["description"] = holidays["description"].str.replace("-", "").str.replace("+", "").str.replace('\d+', '')
    holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])

    # Bridge Holidays
    holidays["description"] = holidays["description"].str.replace("Puente ", "")
    holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])
    
    # Work Day Holidays, that is meant to payback the Bridge.
    work_day = holidays[holidays.type == "Work Day"]  
    holidays = holidays[holidays.type != "Work Day"]  

    # Split

    # Events are national
    events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis = 1).rename({"description":"events"}, axis = 1)

    holidays = holidays[holidays.type != "Event"].drop("type", axis = 1)
    regional = holidays[holidays.locale == "Regional"].rename({"locale_name":"state", "description":"holiday_regional"}, axis = 1).drop("locale", axis = 1).drop_duplicates()
    national = holidays[holidays.locale == "National"].rename({"description":"holiday_national"}, axis = 1).drop(["locale", "locale_name"], axis = 1).drop_duplicates()
    local = holidays[holidays.locale == "Local"].rename({"description":"holiday_local", "locale_name":"city"}, axis = 1).drop("locale", axis = 1).drop_duplicates()

    # TODO can this be refactored to be a bool? (will make processing faster)
    test['test/train'] = 'test'
    train['test/train'] = 'train'

    d = pd.merge(train.append(test), stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")

    # National Holidays & Events
    d = pd.merge(d, national, how = "left")
    # Regional
    d = pd.merge(d, regional, how = "left", on = ["date", "state"])
    # Local
    d = pd.merge(d, local, how = "left", on = ["date", "city"])


    # Work Day: It will be removed when real work day colum created
    d = pd.merge(d,  work_day[["date", "type"]].rename({"type":"IsWorkDay"}, axis = 1),how = "left")

    # EVENTS
    events["events"] =np.where(events.events.str.contains("futbol"), "Futbol", events.events)

    # is there a reason we are not using an existing one hot encoder like sklearn's?
    def one_hot_encoder(df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
        # categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        df.columns = df.columns.str.replace(" ", "_")
        return df, df.columns.tolist()

    # TODO add comments to this
    events, events_cat = one_hot_encoder(events, nan_as_category=False)
    events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1,events["events_Dia_de_la_Madre"])
    events = events.drop(239)

    d = pd.merge(d, events, how = "left")
    d[events_cat] = d[events_cat].fillna(0)

    # New features
    # NOTE - could we just use fillna here?
    d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
    d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
    d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)

    # 
    d["national_independence"] = np.where(d.holiday_national.isin(['Batalla de Pichincha',  'Independencia de Cuenca', 'Independencia de Guayaquil', 'Independencia de Guayaquil', 'Primer Grito de Independencia']), 1, 0)
    d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
    d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
    d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)

    holidays, holidays_cat = one_hot_encoder(d[["holiday_national","holiday_regional","holiday_local"]], nan_as_category=False)
    d = pd.concat([d.drop(["holiday_national","holiday_regional","holiday_local"], axis = 1),holidays], axis = 1)

    he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist()+ d.columns[d.columns.str.startswith("local")].tolist()
    d[he_cols] = d[he_cols].astype("int8")

    d[["test/train","family", "city", "state", "type"]] = d[["test/train","family", "city", "state", "type"]].astype("category")

    # what are we doing here?
    del holidays, holidays_cat, work_day, local, regional, national, events, events_cat, tr, tr1, tr2, he_cols
    gc.collect()


    # Inegrate Holidays data

    d = create_date_features(d)

    # Workday column
    d["workday"] = np.where((d.holiday_national_binary == 1) | (d.holiday_local_binary==1) | (d.holiday_regional_binary==1) | (d['day_of_week'].isin([6,7])), 0, 1)
    d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
    d.drop("IsWorkDay", axis = 1, inplace = True)

    # Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. 
    # Supermarket sales could be affected by this.
    d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

    return d


def oil_setup():
    # Import 
    oil = pd.read_csv(RAW_PATH / 'oil.csv')
    oil["date"] = pd.to_datetime(oil.date)
    # Resample
    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    # Interpolate
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()
    oil['dcoilwtico_interpolated']  = oil['dcoilwtico_interpolated'].rolling( 3,center=True,min_periods=1).mean()

    return oil


def get_oil_holiday_data():
    d = process_holiday_events()
    oil = oil_setup()
    d = pd.merge(d, oil, how = "left", on = ["date"])
    return d
        
    
def get_daily_sales(df):
    """Take a dataframe, group it by date and aggregate sales

    Args:
        df (dataframe): dataframe with sales data

    Returns:
        df: aggregated dataframe with daily sales 
    """
    df = df.groupby("date").sales.sum().reset_index()
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df = df.set_index('date')
    return df

def Transform_Data_For_DT(df,SIZE:60, enable_encode:False):
    """
    Take a dataframe, and transform it to train Decision Tree models
    -> Moving from 1 long line with time series data to many training samples with target values
    link: https://towardsdatascience.com/approaching-time-series-with-a-tree-based-model-87c6d1fb6603

    Encode the family column to be used by Decision Tree (if enable)
    Args:
        df (dataframe): dataframe with daily sales, family and store_nbr
        SIZE (integer): number of prior data points you want to use to train your model
        enable_encode (bool): enable the label encoder to encode the family columns

    Returns:
        df_train: new data frame with prior data points and a target sales
    """
    COLUMNS = ['t{}'.format(x) for x in range(SIZE)] + ['target']
    df_train= []
    fam_list= []
    sto_list = []
    date_list=[]
    family_list =df['family'].unique()
    store_list =df['store_nbr'].unique()
    for fam in family_list:
        for sto in store_list:
            tmp= df[(df['family']==fam) & (df['store_nbr']==sto)]
            tmp=tmp.reset_index()
            if tmp.shape[0]>0:
                for i in range(SIZE, tmp.shape[0]):
                    df_train.append(tmp.loc[i-SIZE:i, 'sales'].tolist())
                    fam_list.append(fam)
                    sto_list.append(sto)
                    date_list.append(tmp.loc[i, 'date'])
    df_train = pd.DataFrame(df_train, columns=COLUMNS)
    df_train['family']=fam_list
    df_train['store_nbr']=sto_list
    df_train['date']=date_list
    df_train['date']= pd.to_datetime(df_train['date'])
    df_train["year"] = df_train.date.dt.year
    df_train["month"] = df_train.date.dt.month
    df_train["daynumber"] = df_train.date.dt.day
    df_train['day_of_week'] = df_train['date'].dt.dayofweek
    columns= ['family', 'store_nbr']
    if enable_encode:
        columns= ['family']
        for col in columns:
            le=LabelEncoder()
            df_train[col]=le.fit_transform(df_train[col])
    
    return(df_train)

def clean_train(train):
    """
    Removed data points filled with 0 for stores that were not opened yet

    Args:
        train (dataframe): training dataset with daily sales and store_nbr
    Returns:
        train: cleaned dataframe
    """
    train = train[~((train.store_nbr == 52) & (train.date < "2017-04-20"))]
    train = train[~((train.store_nbr == 22) & (train.date < "2015-10-09"))]
    train = train[~((train.store_nbr == 42) & (train.date < "2015-08-21"))]
    train = train[~((train.store_nbr == 21) & (train.date < "2015-07-24"))]
    train = train[~((train.store_nbr == 29) & (train.date < "2015-03-20"))]
    train = train[~((train.store_nbr == 20) & (train.date < "2015-02-13"))]
    train = train[~((train.store_nbr == 53) & (train.date < "2014-05-29"))]
    train = train[~((train.store_nbr == 36) & (train.date < "2013-05-09"))]
    train = train.reset_index()
    return(train)

def DT_features(df, enable_encode:False):
    """
    Create a dataframe with features for the DT modeling such as average, median, min, max,...
    Add seasonality metrics as well (week only for the moment, more could be added at a later time)

    Args:
        df (dataframe): transformed dataframe (output of the Transform_Data_For_DT function)
        enable_encode (bool): enable the label encoder to encode the family columns
    Returns:
        df_feats: new dataframe with features used to train DT

    """
    temp=df.drop(columns={'family','store_nbr','date','year','month','daynumber','day_of_week'})
    df_feats=pd.DataFrame()
    df_feats['prev_1'] = temp.iloc[:,-2] #Here -2 as -1 is a target
    for win in [2, 3, 5, 7, 10, 14, 21, 28, 56]: 
        #Interval of study, big window are good against noisy data, small window perform best for sharp trend
        tmp = temp.iloc[:,-1-win:-1]
        #General statistics for base level
        df_feats['mean_prev_{}'.format(win)] = tmp.mean(axis=1)
        df_feats['median_prev_{}'.format(win)] = tmp.median(axis=1)
        df_feats['min_prev_{}'.format(win)] = tmp.min(axis=1)
        df_feats['max_prev_{}'.format(win)] = tmp.max(axis=1)
        df_feats['std_prev_{}'.format(win)] = tmp.std(axis=1)
        #Capturing trend
        df_feats['mean_ewm_prev_{}'.format(win)] = tmp.T.ewm(com=9.5).mean().T.mean(axis=1)
        df_feats['last_ewm_prev_{}'.format(win)] = tmp.T.ewm(com=9.5).mean().T.iloc[:,-1]
        
        df_feats['avg_diff_{}'.format(win)] = (tmp - tmp.shift(1, axis=1)).mean(axis=1)
        #df_feats['avg_div_{}'.format(win)] = (tmp / tmp.shift(1, axis=1)).mean(axis=1) --Not sure of this one
        del tmp
    for win in [2, 3, 4, 8]:
        tmp = temp.iloc[:,-1-win*7:-1:7] #7 for week
        #Features for weekly seasonality
        df_feats['week_mean_prev_{}'.format(win)] = tmp.mean(axis=1)
        df_feats['week_median_prev_{}'.format(win)] = tmp.median(axis=1)
        df_feats['week_min_prev_{}'.format(win)] = tmp.min(axis=1)
        df_feats['week_max_prev_{}'.format(win)] = tmp.max(axis=1)
        df_feats['week_std_prev_{}'.format(win)] = tmp.std(axis=1)
        del tmp
    
    df_feats['family']=df['family']
    df_feats['store_nbr']=df['store_nbr']
    df_feats['date']=df['date']
    df_feats['year']=df['year']
    df_feats['month']=df['month']
    df_feats['day']=df['daynumber']
    df_feats['day_of_week']=df['day_of_week']
    if enable_encode:
        columns= ['family']
        for col in columns:
            le=LabelEncoder()
            df_feats[col]=le.fit_transform(df_feats[col])
    return(df_feats)


def load_model(path:Path, name:str):
    """
    Load a previously saved pickled model
    Args:
        path (string): path where the model is saved
        name (string): name of the model you want to loan
    Returns:
        model: loaded model
    """
    model_path=os.path.join(path.resolve().as_posix(), name)
    #model_path=path.joinpath(path, name)
    model = pickle.load(open(model_path, 'rb'))
    return model

def save_model(path, model, name):
    """
    Save a model under the specified location in a pickle format
    Args:
        path (string): path where the model is saved
        model: model to save
        name (string): name of the model you want to loan
    Returns:
        
    """
    model_path=os.path.join(path.resolve().as_posix(), name)
    pickle.dump(model, open(model_path, 'wb'))
    print("Model has been saved in the following location",model_path)
    return()