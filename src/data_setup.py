# Standard imports
import gc
import numpy as np
import os
import pandas as pd
from pathlib import Path

# Define Gloabl Path Variables
# current_directory = os.getcwd()
# parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

ROOT_PATH = Path(os.path.dirname(os.getcwd()))
DATA_PATH = ROOT_PATH / 'data'
assert 'raw' in os.listdir(DATA_PATH), 'Data directory not structured properly, see readme.md'

RAW_PATH = DATA_PATH / 'raw'
PROCESSED_PATH = DATA_PATH / 'processed'
SUBMISSION_PATH = DATA_PATH / 'submissions'

TRAIN_PATH = RAW_PATH / 'train.csv'
TEST_PATH = RAW_PATH / 'test.csv'
HOLIDAYS_PATH = RAW_PATH / 'holidays_events.csv'
TRANSACTIONS_PATH = RAW_PATH / 'transactions.csv'
STORES_PATH = RAW_PATH / 'stores.csv'
OIL_PATH = RAW_PATH / 'oil.csv'

""" Old code to downlaod data from kaggle, works on mac but needs changes to work on windows
if 'data' not in os.listdir('..') or 'raw' not in os.listdir('../data'):
    print('\ndownloading kaggle data...')
    ! mkdir ../data
    ! mkdir ../data/raw
    ! mkdir ../data/processed
    ! kaggle competitions download -c store-sales-time-series-forecasting
    ! unzip store-sales-time-series-forecasting.zip
    ! mv *.csv ../data/raw
    ! rm store-sales-time-series-forecasting.zip
    ! mkdir ../data/submissions
    ! mv ../data/raw/*_submission* ../data/submissions/
else:
    print('kaggle data already downloaded in ../data')"""

def get_data():
    """Load processed dataframes for train, test, stores, transactions

    Returns:
        tuple: four dataframes (train, test, stores, transactions)
    """
    
    # TODO refactor this to use pickle and the processed data sections

    # Read data files into dataframes
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    stores = pd.read_csv(STORES_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH).sort_values(['store_nbr', 'date'])

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
    for df in dfs_with_date:
        # convert to a datetime object
        df['date'] = pd.to_datetime(train.date)     
        # add in year and month
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
    train, test, transactions = dfs_with_date

    # smaller floats
    train['onpromotion'] = train.onpromotion.astype('float32')
    test['onpromotion'] = test.onpromotion.astype('float32')
    train['sales'] = train.sales.astype('float32')
    
    # set indexes
    train = train.set_index('id')
    test = test.set_index('id')
    stores = stores.set_index('store_nbr')
    
    return train, test, stores, transactions


def get_daily_sales(df):
    """Take a dataframe, group it by date and aggregate sales

    Args:
        df (dataframe): dataframe with sales data

    Returns:
        df: aggregated dataframe with daily sales 
    """
    df = df.groupby(["date"]).sales.sum().reset_index()
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df = df.set_index('date')
    return df


# Time Related Features
def create_date_features(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    #df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['week_of_month'] = ((df['day_of_month']-1) // 7 + 1).astype("int8")


    df['week_of_year'] = (df.date.dt.weekofyear).astype("int8")
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
    df['year'] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday // 4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype("int8")
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df

def process_holiday_events():
    """_summary_

    Returns:
        _type_: _description_
    """
    train, test, stores, transactions = get_data()
    
    #Import holiday data
    holidays = pd.read_csv(HOLIDAYS_PATH)
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
    oil = pd.read_csv(OIL_PATH)
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

# d = get_data()   
# print(d.shape)
# print('complete')