# Standard imports
import gc
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

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
    if SIZE <=15:
        print('SIZE parameter is too small, pick a bigger integer')
        return()
    else:
        COLUMNS = ['t{}'.format(x) for x in range(SIZE-15)] + ['target']
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
                        df_train.append(tmp.loc[i-SIZE:i-15, 'sales'].tolist())
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
    for win in [2, 3, 5, 7, 10, 14, 21, 28]: 
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
    for win in [2, 3, 4]:
        #Check ADF test for seasonality
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
    df_feats['date']= pd.to_datetime(df_feats['date'])
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


