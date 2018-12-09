# Import Packages
##############################################################################
import numpy as np, pandas as pd, tensorflow as tf
import gc, pyodbc, random, tqdm, re, string, itertools
from string import punctuation
from sqlalchemy import create_engine, MetaData, Table, select
from os.path import getsize, basename
from __future__ import division
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# User Input
##############################################################################
glove_txt_file = 'D:/glove/glove.840B.300d.txt'
train_file = 'D:/quora/data/train.csv'
test_file = 'D:/quora/data/test.csv'

# Define Functions
##############################################################################
def my_create_engine(mydsn = 'mssql_conn', mydatabase = 'db_general', **kwargs):
    """create sql server engine to query & write tables"""
    conn = pyodbc.connect(
    r'DSN=mssql_conn;'
    r'UID=nick;'
    r'PWD=7027green;'
    )
    cursor = conn.cursor()
    connection_string = 'mssql+pyodbc://@%s' % mydsn
    cargs = {'database': mydatabase}
    cargs.update(**kwargs)
    e = create_engine(connection_string, connect_args=cargs)
    return e, cursor, conn

def tbl_write(tbl_name, engine, pandas_df):
    """write pandas dataframe to table in sqlserver"""
    pandas_df.to_sql(tbl_name, engine, if_exists = 'append', chunksize = None, index = False)

def load_glove(glove_file_path, progress_print = 5000, encoding_type = 'utf8'):
    """load glove (Stanford NLP) file and return dictionary"""
    num_lines = sum(1 for line in open(glove_txt_file, encoding = encoding_type))
    embed_dict = dict()
    line_errors = []
    f = open(glove_file_path, encoding = encoding_type)
    for i, l in enumerate(f):
        l_split = l.split()
        try:
            embed_dict[l_split[0]] = np.asarray(l_split[1:], dtype = 'float32')
        except:
            line_errors.append(1)
        if ((i / progress_print) > 0) and (float(i / progress_print) == float(i // progress_print)):
            print(str(int(i / 1000)) + ' K of ' + str(int(num_lines / 1000)) + ' K lines completed')
        else:
            pass
    f.close()
    print('failed lines in file: ' + str(int(np.sum(line_errors))))
    return embed_dict
    
def clean_tokenize(some_string):
    """split on punct / whitespace, make lower case"""
    pattern = re.compile(r'(\s+|[{}])'.format(re.escape(punctuation)))
    clean_lower = [part.lower() for part in pattern.split(some_string) if part.strip()]
    return clean_lower

# Execute Functions
##############################################################################
# Glove Dictionary
glove_dict = load_glove(glove_file_path = glove_txt_file)
"""glove_dict.get('king')"""

# Read Train & Test Flat Files
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Extract List of Words and Primary Keys - Train
train_token_list = []
train_qid_list = []
for i, x in enumerate(train['question_text']):
    tokens = clean_tokenize(x)
    ids = [train['qid'][i]] * len(tokens)
    train_token_list.append(tokens)
    train_qid_list.append(ids)
    if float(i / 10000) == float(i // 10000):
        print(str(int(i)) + ' of ' + str(train.shape[0]) + ' lines completed')

train_token_list = list(itertools.chain.from_iterable(train_token_list))
train_qid_list = list(itertools.chain.from_iterable(train_qid_list))

# Extract List of Words and Primary Keys - Test
test_token_list = []
test_qid_list = []
for i, x in enumerate(test['question_text']):
    tokens = clean_tokenize(x)
    ids = [test['qid'][i]] * len(tokens)
    test_token_list.append(tokens)
    test_qid_list.append(ids)
    if float(i / 10000) == float(i // 10000):
        print(str(int(i)) + ' of ' + str(test.shape[0]) + ' completed')

test_token_list = list(itertools.chain.from_iterable(test_token_list))
test_qid_list = list(itertools.chain.from_iterable(test_qid_list))

# Create Dataframes and Write to Sql Server Tables
train_token_df = pd.DataFrame({'qid': train_qid_list, 'word': train_token_list})
test_token_df = pd.DataFrame({'qid': test_qid_list, 'word': test_token_list})
del train_token_list; del test_token_list; gc.collect()
#eng, cursor, conn = my_create_engine()
#tbl_write('quora_train_tokens', eng, train_token_df)
#tbl_write('quora_test_tokens', eng, test_token_df)


# Counts for Every Unique Word
train_test_token_df = pd.concat([train_token_df, test_token_df], axis = 0)
tt_token_count_df = train_test_token_df.\
groupby(['word'], as_index = False).\
agg({'qid':'nunique'}).\
sort_values('qid', ascending = False)

# List of Words Occuring 10+ Times
tt_token_count_df = tt_token_count_df[tt_token_count_df.qid >= 10]

# Glove Weights for Every Word Occuring 10+ Times
weight_list = []
for i, z in enumerate([x for x in tt_token_count_df['word']]):
    if glove_dict.get(z) is not None:
        weight_list.append([i for i in glove_dict.get(z)])
    else:
        weight_list.append([0] * 300)
    
word_weight_lookup = pd.DataFrame(weight_list)
word_weight_lookup.columns = ['wt' + str(i) for i in range(300)]
word_weight_lookup['word'] = [i for i in tt_token_count_df['word']]
#tbl_write('quora_word_glove_weights', eng, word_weight_lookup)
del train_test_token_df; gc.collect()

# Write Flat Files
train_token_df_b = pd.merge(train_token_df,
                            word_weight_lookup,
                            how = 'left',
                            on = 'word')

train_token_df_b.to_csv('D:/quora/data/train_word_weights.csv', index = False)

test_token_df_b = pd.merge(test_token_df,
                           word_weight_lookup,
                           how = 'left',
                           on = 'word')

test_token_df_b.to_csv('D:/quora/data/test_word_weights.csv', index = False)





