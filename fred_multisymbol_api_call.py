# Config
###############################################################################
# Import Packages
import numpy as np
import pandas as pd
import fred
from functools import reduce

# Define Function
###############################################################################
def fred_series_mult_pd(symbol_list, api_key):
    """
    Use fred api to retrieve time series data.
    Args:
        symbol_list: list of strings representing fred series symbols
        api_key: developer API key from https://fred.stlouisfed.org/
    Returns:
        merged pandas dataframe with date ('dt') and numeric value (<symbol>) columns
    Dependencies:
        pandas, fred, functools.reduce
    """
    # Use API key 
    fred.key(api_key)
    
    # Define inner functions
    def inner_join_pandas_list(pandas_list, join_on):
        return reduce(lambda a, b: pd.merge(a, b, on = join_on), pandas_list)
    
    # Query each series in symbol_list, append to df_list
    df_list = []
    for symbol in symbol_list:
        # Import series, convert to dataframe, drop query timestamps, rename columns, format dates
        series_df = pd.DataFrame(fred.observations(symbol)['observations']).\
                                 drop(['realtime_start','realtime_end'], axis = 1).\
                                 rename(columns = {'date' : 'dt', 'value' : symbol.lower()})
        series_df['dt'] = [x.date() for x in pd.to_datetime(series_df['dt'])]
        df_list.append(series_df)
    
    # Merge list of dataframes
    return inner_join_pandas_list(df_list, 'dt')