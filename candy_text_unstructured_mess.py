# Packages
#####################################################################
import numpy as np, pandas as pd
import time, csv, pickle, re

# Data
#####################################################################
file_path = 'C:/Users/user/Documents/school/practicum/FA12_Data.txt'
layout_path = 'C:/Users/user/Documents/school/practicum/fa12_Layout.txt'
save_path = 'C:/Users/user/Documents/school/practicum/fa12_cleaned.csv'

# Define Data Reading & Processing Functions
#####################################################################

# Read Unstructured Text File, Return as List of Lists
def read_text_garbage(garbage_path):
    with open(garbage_path, 'r') as in_file:
        structured_garbage = in_file.read().split('\n')
    return structured_garbage

# Get Column Positions from Layout File
def get_column_pos(dat_path, start_col, len_col):
    layout = pd.read_csv(dat_path)
    col_positions = []
    for i, r in layout.iterrows():
        col_positions.append((r[start_col]-1, (r[start_col]-1) + r[len_col]))
    return col_positions
    
# Separate Columns Based on Positions in Layout File
def col_pos_sep(dat_list, col_pos_tuples):
    outer_list = []
    for dat in dat_list:
        inner_list = []
        for cpt in col_pos_tuples:
            inner_list.append(dat[cpt[0]:cpt[1]])
        outer_list.append(inner_list)
    return outer_list
            

def col_pos_sep(dat_list, col_pos_tuples):
    outer_list = []
    for dat in dat_list:
        inner_list = []
        for cpt in col_pos_tuples:
            if dat[cpt[0]:cpt[1]] in ['', ' ']:
                inner_list.append(0)
            else:
                inner_list.append(dat[cpt[0]:cpt[1]])
        outer_list.append(inner_list)
    return outer_list
            

# Run 'read_text_garbage', 'get_column_pos', and 'col_pos_sep' Functions Iteratively
def proc_aggregate(layout_path, file_path):
    layout = pd.read_csv(layout_path)
    df = read_text_garbage(file_path)
    col_pos_temp = get_column_pos(layout_path, 'col_start', 'col_len')
    df_sep = col_pos_sep(df, col_pos_temp)
    df_pandas = pd.DataFrame(df_sep, columns = layout['var_name']).fillna(0)
    return df_pandas
    
# Excecute Data Reading & Processing Functions
#####################################################################    
df = proc_aggregate(layout_path, file_path)    

# Save Cleaned Data to csv File
##################################################################### 
df.to_csv(save_path)


