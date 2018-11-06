# Import Packages
####################################################################
import numpy as np, pandas as pd, tensorflow as tf, xgboost as xgb
from random import shuffle
from tqdm import tqdm
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
import re, pprint, re, string, sklearn, gc
from sklearn.model_selection import train_test_split
from functools import reduce
from datetime import datetime
import sklearn.preprocessing

# Import Data
####################################################################
folder_path = "...."
train = pd.read_csv(folder_path + "training_set.csv")
train_meta = pd.read_csv(folder_path + "training_set_metadata.csv")

# Define Functions
####################################################################
def mjd_to_unix(df, mjd_col):
    temp_list = []
    for mjd in df[mjd_col]:
        temp_list.append((mjd - 40587) * 86400)
    return temp_list

def plastic_ts_agg(dat, metadat):
    df_copy = dat.\
    groupby(['object_id', 'mjd', 'passband'], axis = 0, as_index = False).\
    agg({'flux': [np.min, np.max, np.mean],
         'flux_err': [np.min, np.max, np.mean],
         'detected': [np.max]}).\
    sort_values(['object_id', 'mjd', 'passband'], axis = 0).\
    fillna(0)
    df_copy.columns = ['object_id', 'mjd', 'passband', 'min_flux', 'max_flux', 
                       'mean_flux', 'min_flux_err', 'max_flux_err', 'mean_flux_err', 'max_detected']
    df_copy['passband'] = df_copy['passband'].astype(int)
    df_copy['object_id'] = df_copy['object_id'].astype(int)
    output = df_copy.sort_values(['object_id', 'mjd', 'passband'])
    start_tm = output[['object_id', 'mjd']]
    start_tm.columns = ['object_id', 'mjd_start']
    start_tm = start_tm.\
    groupby(['object_id'], as_index = False).\
    agg({'mjd_start':'min'})
    output = pd.merge(output, start_tm, 'left', 'object_id')
    output['unix_mjd'] = mjd_to_unix(output, 'mjd')
    output['unix_mjd_start'] = mjd_to_unix(output, 'mjd_start')
    output['tm_elapsed'] = [np.float64(i) for i in output['unix_mjd'] - output['unix_mjd_start']]
    output['dt'] = [datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S') for i in output['unix_mjd']]
    output['doy'] = [int(datetime.strptime(i,'%Y-%m-%d %H:%M:%S').strftime('%j')) for i in output['dt']]
    output.drop(['mjd_start', 'unix_mjd', 'unix_mjd_start', 'dt', 'mjd'], inplace = True, axis = 1)
    output = pd.merge(output, metadat, 'inner', 'object_id').fillna(0)
    return output

def pd_to_array_bycol(df, bycol, ycol):
    x_list = []
    y_list = []
    uniq_colvals = [ucv for ucv in set(df[bycol])]
    for ucv in uniq_colvals:
        x_list.append(df[df[bycol] == ucv].drop([bycol, ycol], axis = 1).values.astype('float32'))
        y_list.append(int(df[df[bycol] == ucv].iloc[0,:][ycol]))
    sklb = sklearn.preprocessing.LabelBinarizer()
    sklb.fit(np.unique(y_list))
    dummy_y = sklb.transform(y_list)
    return dummy_y, np.array(x_list)

# Execute Data Prep Functions
####################################################################
train_xy = plastic_ts_agg(train, train_meta)
train_y, train_x = pd_to_array_bycol(train_xy, 'object_id', 'target')
