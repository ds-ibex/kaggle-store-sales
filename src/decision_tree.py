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
                        #df_train.append(tmp.loc[i-SIZE:i-15, 'sales'].tolist())
                        df_train.append(tmp.loc[i-SIZE:i, 'sales'].tolist())
                        fam_list.append(fam)
                        sto_list.append(sto)
                        date_list.append(tmp.loc[i, 'date'])
        df_train = pd.DataFrame(df_train, columns=COLUMNS)
        df_train['family']=fam_list
        df_train['store_nbr']=sto_list
        df_train['date']=date_list
        df_train['date']= pd.to_datetime(df_train['date'])
        df_train=df_train.merge(df, on=['date','family','store_nbr'])
        #df_train=df_train.drop(columns={'sales','id'})
        #LabelEncode the categorical column

        columns= ['family','city','state','type']
        if enable_encode:
            columns= ['family','city','state','type']
            for col in columns:
                le=LabelEncoder()
                df_train[col]=le.fit_transform(df_train[col])
        df_train = df_train.astype({'is_test': 'int8'})
        return(df_train)
    
def DT_features(df, enable_encode:False): ##@Leo Update format to have f''
    """
    Create a dataframe with features for the DT modeling such as average, median, min, max,...
    Add seasonality metrics as well (week only for the moment, more could be added at a later time)

    Args:
        df (dataframe): transformed dataframe (output of the Transform_Data_For_DT function)
        enable_encode (bool): enable the label encoder to encode the family columns
    Returns:
        df_feats: new dataframe with features used to train DT

    """
    temp=df.drop(columns={'family','store_nbr','date'})
    df_feats=pd.DataFrame()
    df_feats['prev_1'] = temp.iloc[:,-2] #Here -2 as -1 is a target
    df_feats = df_feats.astype({'prev_1':temp.iloc[:,-2].dtype})
    for win in [2, 3, 5, 7, 10, 14, 21, 28]: 
        #Interval of study, big window are good against noisy data, small window perform best for sharp trend
        tmp = temp.iloc[:,-1-win:-1]
        #General statistics for base level
        df_feats['mean_prev_{}'.format(win)] = tmp.mean(axis=1)
        df_feats = df_feats.astype({'mean_prev_{}'.format(win):tmp.mean(axis=1).dtype})
        df_feats['median_prev_{}'.format(win)] = tmp.median(axis=1)
        df_feats = df_feats.astype({'median_prev_{}'.format(win):tmp.median(axis=1).dtype})
        df_feats['min_prev_{}'.format(win)] = tmp.min(axis=1)
        df_feats = df_feats.astype({'min_prev_{}'.format(win):tmp.min(axis=1).dtype})
        df_feats['max_prev_{}'.format(win)] = tmp.max(axis=1)
        df_feats = df_feats.astype({'max_prev_{}'.format(win):tmp.max(axis=1).dtype})
        df_feats['std_prev_{}'.format(win)] = tmp.std(axis=1)
        df_feats = df_feats.astype({'std_prev_{}'.format(win):tmp.std(axis=1).dtype})
        #Capturing trend
        df_feats[f'mean_ewm_prev_{win}'] = tmp.T.ewm(com=9.5).mean().T.mean(axis=1)
        df_feats = df_feats.astype({f'mean_ewm_prev_{win}':tmp.T.ewm(com=9.5).mean().T.mean(axis=1).dtype})
        df_feats['last_ewm_prev_{}'.format(win)] = tmp.T.ewm(com=9.5).mean().T.iloc[:,-1]
        df_feats = df_feats.astype({'last_ewm_prev_{}'.format(win):tmp.T.ewm(com=9.5).mean().T.iloc[:,-1].dtype})
        
        df_feats['avg_diff_{}'.format(win)] = (tmp - tmp.shift(1, axis=1)).mean(axis=1)
        df_feats = df_feats.astype({'avg_diff_{}'.format(win):(tmp - tmp.shift(1, axis=1)).mean(axis=1).dtype})
        #df_feats['avg_div_{}'.format(win)] = (tmp / tmp.shift(1, axis=1)).mean(axis=1) --Not sure of this one
        del tmp
    for win in [2, 3, 4]:
        #Check ADF test for seasonality
        tmp = temp.iloc[:,-1-win*7:-1:7] #7 for week
        #Features for weekly seasonality
        df_feats['week_mean_prev_{}'.format(win)] = tmp.mean(axis=1)
        df_feats = df_feats.astype({'week_mean_prev_{}'.format(win):tmp.mean(axis=1).dtype})
        df_feats['week_median_prev_{}'.format(win)] = tmp.median(axis=1)
        df_feats = df_feats.astype({'week_median_prev_{}'.format(win):tmp.median(axis=1).dtype})
        df_feats['week_min_prev_{}'.format(win)] = tmp.min(axis=1)
        df_feats = df_feats.astype({'week_min_prev_{}'.format(win):tmp.min(axis=1).dtype})
        df_feats['week_max_prev_{}'.format(win)] = tmp.max(axis=1)
        df_feats = df_feats.astype({'week_max_prev_{}'.format(win):tmp.max(axis=1).dtype})
        df_feats['week_std_prev_{}'.format(win)] = tmp.std(axis=1)
        df_feats = df_feats.astype({'week_std_prev_{}'.format(win):tmp.std(axis=1).dtype})
        del tmp
    
    df_feats['family']=df['family']
    df_feats['store_nbr']=df['store_nbr']
    df_feats['date']=df['date']
    df_feats['date']= pd.to_datetime(df_feats['date'])
    df_feats=df_feats.merge(df, on=['date','family','store_nbr'])
    tmp=df_feats.copy()
    COLUMNS = [col for col in df_feats.columns if col not in ['t{}'.format(x) for x in range(60)]]
    df_feats=df_feats[COLUMNS]
    tmp=df_feats.copy()
    """if enable_encode:
        columns= ['family']
        for col in columns:
            le=LabelEncoder()
            df_feats[col]=le.fit_transform(df_feats[col])"""
    #MinMaxScale on continuous columns
    columns=[col for col in df_feats.columns if col not in ['family','city','state','type','onpromotion','date','sales','store_nbr']]
    minVec = df_feats[columns].min().copy()
    maxVec = df_feats[columns].max().copy()
    df_feats[columns]=(df_feats[columns]-minVec)/(maxVec-minVec)
    #for col in columns:
    #    df_feats = df_feats.astype({col:tmp[col].dtype})
    #change 0 to -1 on the onpromotion columns
    df_feats.loc[df_feats.onpromotion==0,'onpromotion']=-1
    df_feats=reduce_mem_usage(df_feats)
    return(df_feats)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


