### Import Packages
######################################################################################################
import collections
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import random
import time
import xgboost as xgb



### File Paths (config_fp...)
######################################################################################################
# Raw Data
config_fp_raw_data_dir = 'D:/insurance_data/'
config_fp_raw_train_file = 'train.csv'
config_fp_raw_test_file = 'test.csv'

# Processed Data
config_fp_proc_data_dir = 'D:/processed_insurance_data/'



### Define Functions
######################################################################################################
def print_timestamp_message(message, timestamp_format = '%Y-%m-%d %H:%M:%S'):
    """
    Print formatted timestamp followed by custom message
    
    Args:
        message (str): string to concatenate with timestamp
        timestamp_format (str): format for datetime string. defaults to '%Y-%m-%d %H:%M:%S'
    """
    ts_string = datetime.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    print(f'{ts_string}: {message}')


def common_list_values(list_one, list_two):
    """
    Return a list of unique values shared by two lists
    Args:
        list_one (list): first list of values
        list_two (list): second list of values
    """
    return list(set(list_one).intersection(list_two))


def actual_vs_pred_deciles(actual, predicted):
    """
    Calculate average predicted vs. actual loss for every
    decile of actual loss values. Returns a pandas DataFrame
    with columns 'Actual_Decile', 'Actual_Value', 'Predicted_Value'
    Args:
        actual : actual values
        predicted : predicted values
    """
    actual_deciles = [ad for ad in pd.qcut(actual, 10, labels = False)]
    output_df = pd.DataFrame({'Actual_Decile' : actual_deciles,
                              'Actual_Value' : actual,
                              'Predicted_Value' : predicted}).\
    groupby(['Actual_Decile'], as_index = False).\
    agg({'Actual_Value' : 'mean',
         'Predicted_Value' : 'mean'})
    return output_df


def create_partition_column(pandas_df, identifier_col, n_partitions = 10, partition_colname = 'partition'):
    """
    Add a partition column to a pandas DataFrame that ensures identifiers only occur in a single fold
    Args:
        pandas_df (pandas.DataFrame): DataFrame object to modify
        identifier_col (str): existing column with identifiers (e.g. policy number)
        n_partitions (int): number of partitions to create
        partition_colname (str): new column name to create
    """
    unique_ids = list(set(pandas_df[identifier_col]))
    random.shuffle(unique_ids)
    unique_partitions = list(range(1, n_partitions + 1, 1))
    partitions = unique_partitions * int(np.ceil(len(unique_ids) / len(unique_partitions)))
    id_partition_dict = dict(zip(unique_ids, partitions[:len(unique_ids)]))
    print(f'Creating field {partition_colname} with {n_partitions} partitions')
    pandas_df[partition_colname] = [id_partition_dict.get(x) for x in pandas_df[identifier_col]]
    

def recode_sparse_values(lst, min_freq = 0.01, recode_value = 'RECODE'):
    """
    Recode values in iterable object below some percentage frequency
    Args:
        lst (list): ... or other iterable object
        min_freq (float): minimum percentage frequency to not be recoded
        recode_value (str): string to replace values below specified frequency
    Returns:
        list
    """
    key_values = collections.Counter(lst).keys()
    percent_count_values = [v / len(lst) for v in collections.Counter(lst).values()]
    keep_values = [kv for i, kv in enumerate(key_values) if percent_count_values[i] > min_freq]
    return_values = [x if x in keep_values else recode_value for x in lst]
    return return_values


class TrainTestTransformer:
    def __init__(self,
                 df_train, df_test, y_col, categ_cols, cont_cols,
                 categ_min_freq = 0.01, categ_recode_value = 'RECODE'):
        self.df_train = df_train
        self.df_test = df_test
        self.y_col = y_col
        self.categ_cols = categ_cols
        self.cont_cols = cont_cols
        self.categ_min_freq = categ_min_freq
        self.categ_recode_value = categ_recode_value
        
    def recode_sparse_categoricals(self):
        temp_train_categ_df = pd.DataFrame()
        temp_test_categ_df = pd.DataFrame()
        for c in self.categ_cols:
            temp_train_categ_df[c] = recode_sparse_values(self.df_train[c],
                                                          min_freq = self.categ_min_freq,
                                                          recode_value = self.categ_recode_value)
            train_vals = set(list(temp_train_categ_df[c]))
            temp_test_categ_df[c] = [v if v in train_vals else self.categ_recode_value for v in self.df_test[c]]
            
        return temp_train_categ_df, temp_test_categ_df[temp_train_categ_df.columns]
        
    def label_encode_categoricals(self):
        recoded_train_categ_df, recoded_test_categ_df = self.recode_sparse_categoricals()
        dummy_train = pd.get_dummies(recoded_train_categ_df)
        dummy_test = pd.get_dummies(recoded_test_categ_df)
        common_cols = common_list_values(dummy_train.columns, dummy_test.columns)
        return dummy_train[common_cols], dummy_test[common_cols]
    
    def scale_continuous_fields(self):
        temp_train_contin_df = pd.DataFrame()
        temp_test_contin_df = pd.DataFrame()
        for c in self.cont_cols:
            train_mean = np.mean(self.df_train[c])
            train_stdev = np.std(self.df_train[c])
            temp_train_contin_df[c] = [(x - train_mean) / train_stdev if not np.isnan(x) else 0 for x in self.df_train[c]]
            temp_test_contin_df[c] = [(x - train_mean) / train_stdev if not np.isnan(x) else 0 for x in self.df_test[c]]
        return temp_train_contin_df, temp_test_contin_df
    
    def transform_train_test_xcols(self):
        cont_train, cont_test = self.scale_continuous_fields()
        categ_train, categ_test = self.label_encode_categoricals()
        xy_cols = [self.y_col] + self.categ_cols + self.cont_cols
        non_xy_cols = [c for c in self.df_train.columns if c not in xy_cols]
        train_x = pd.concat([cont_train, categ_train], axis = 1)
        test_x = pd.concat([cont_test, categ_test], axis = 1)
        return train_x, test_x



### Read Data and Define Columns
######################################################################################################
# Read Dataset
loss_cost_df = pd.read_csv(f'{config_fp_raw_data_dir}{config_fp_raw_train_file}')

# List of Categorical, Continuous, and Dependent Fields
categ_x_cols = [c for c in loss_cost_df.columns if 'cat' in c.lower()]
cont_x_cols = [c for c in loss_cost_df.columns if 'cont' in c.lower()]
y_col = 'loss'
partition_col = 'partition'
id_col = 'id'



### Apply Transformations After Splitting Train and Test
######################################################################################################
# Create 10-fold Partitions
create_partition_column(loss_cost_df, 'id')

# Apply Transformations to Train and Test
transformer = TrainTestTransformer(df_train = loss_cost_df[loss_cost_df.partition < 8],
                                   df_test = loss_cost_df[loss_cost_df.partition >= 8],
                                   y_col = y_col,
                                   categ_cols = categ_x_cols,
                                   cont_cols = cont_x_cols,
                                   categ_min_freq = 0.05)

train_x, test_x = transformer.transform_train_test_xcols()

# Create Train and Test 'Y' and Identifier DataFrames
train_y = loss_cost_df[loss_cost_df.partition < 8][[y_col]]
test_y = loss_cost_df[loss_cost_df.partition >= 8][[y_col]]
train_id = loss_cost_df[loss_cost_df.partition < 8][[id_col]]

# Split Training Set into Train and Validation (for early stopping)
train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(train_x, train_y, test_size = 0.2, random_state = 11092020)



### Get 5-Fold Residuals from Training Set
######################################################################################################
# Initial Hyperparameters
param_dict = {'objective' : 'reg:tweedie',
              'eta' : 0.03,
              'min_child_weight' : 8,
              'subsample' : 0.7,
              'colsample_bytree' : 0.6,
              'max_depth' : 6,
              'early_stopping_rounds' : 15,
              'stopping_metric' : 'gini'}



# Add 5-fold partition column to train_x
train_kfold_partitions = list(range(5)) * int(np.ceil(train_x.shape[0] / 5))
train_kfold_partitions = sorted(train_kfold_partitions[:train_x.shape[0]])


# Loop over folds, appending predictions to outer list
pred_list = []

for test_fold in range(5):
    # Split Training Set into 4 train vs. 1 test folds
    train_indices = [i for i, x in enumerate(train_kfold_partitions) if x != test_fold]
    test_indices = [i for i, x in enumerate(train_kfold_partitions) if x == test_fold]
    
    subset_train_x = train_x.iloc[train_indices]
    subset_test_x = train_x.iloc[test_indices]
    subset_train_y = train_y.iloc[train_indices]
    subset_test_y = train_y.iloc[test_indices]
    
    ### Fit Model on 4 of 5 Folds
    # Create XGBoost DMatrix Objects and Validation Set Watchlist
    dat_train = xgb.DMatrix(subset_train_x, label = subset_train_y)
    dat_valid = xgb.DMatrix(valid_x, label = valid_y)
    watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]
    
    xgb_trn = xgb.train(params = param_dict,
                        dtrain = dat_train,
                        #num_boost_round = 5000,
                        num_boost_round = 50,
                        evals = watchlist,
                        early_stopping_rounds = 12,
                        verbose_eval = True)
    
    # Make Predictions on 1 Fold
    pred = xgb_trn.predict(xgb.DMatrix(subset_test_x))
    pred_list.append(list(pred))
    
    print_timestamp_message(f'Finished fold {test_fold}')
    
    
    
# Append Predictions to Training Set
train_x['prediction'] = list(itertools.chain.from_iterable(pred_list))
    

# Calculate Residual for Each Observation
train_x['residual'] = train_x['prediction'] - train_y['loss']
    
    

### Residual EDA
######################################################################################################


# Example: look at cont1 (variable) vs. residuals - average
temp = train_x[['residual', 'cat25_A']].\
groupby(['cat25_A'], as_index = False).\
agg({'residual' : 'median'})


# Example 2: continuous variable grouped into decile
temp = train_x[['residual', 'cont1']]
temp['cont1'] = pd.qcut(temp['cont1'], 10, labels=np.arange(10, 0, -1))
temp = temp.\
groupby(['cont1'], as_index = False).\
agg({'residual' : 'median'})


plt.plot(list(temp['cont1']), temp['residual'], marker = 'o')
plt.show()



def continuous_residual_eda(dframe, resid_col, var_col):
    temp = dframe[[var_col, resid_col]]
    temp[var_col] = pd.qcut(temp[var_col], 10, labels=np.arange(10, 0, -1))
    temp = temp.\
    groupby([var_col], as_index = False).\
    agg({resid_col : 'median'})
    return temp
    

attempt1 = continuous_residual_eda(dframe = train_x, resid_col = 'residual', var_col = 'cont1')




### Notes
######################################################################################################
# possible ways to improve:
# if sparse categorical levels in a variable have poor predictions, we can regroup accordinly
# for a conntinuous variable, consider interactions, capping, floors
























