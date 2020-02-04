# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:28:53 2020

@author: user
"""
# Define Functions
###############################################################################

def add_seasonality(time_series, seasonal_period, beta) -> list:
    '''Use a beta coefficient <beta> to add seasonality to a vector of numbers'''
    return [x + ((i % seasonal_period) / seasonal_period * beta * x) for i, x in enumerate(time_series)]


def add_multiplicative_trend(time_series, beta) -> list:
    '''Add a multiplicative trend to a vector of numbers'''
    return [x * (beta**i) for i, x in enumerate(time_series)]


class TimeSeriesData:
    '''
    Add multiple seasonality and linear trend to a stochastic
    time series of length <n_intervals>
    
    Args:
        n_intervals: number of periods to create
        avg: average value of randomly distributed series
        sdev: standard deviation of randomly distributed series
        seasonality: list of seasonal periods to create in output series
        seasonal_beta: beta coefficients corresponding to positional index of <seasonality>
        linear_beta: a multiplicative linear trend to apply to the entire series (defaults to 1.0005)
        
    Dependencies:
        add_seasonality()
        add_multiplicative_trend
        
    Examples:
        my_ts = TimeSeriesData(n_intervals = (365 * 10), avg = 1000, sdev = 100,
                               seasonality = [90, 365], linear_beta = 1.0005)        
        t = my_ts.get_time_series()
        my_ts.plot_time_series()

    Returns:
        list of floats
    '''
    
    def __init__(self, n_intervals, avg = 1000, sdev = 100,
                 seasonality = [90, 365], seasonal_beta = [1.1, 1.3], linear_beta = 1.0005):
        self.n_intervals = n_intervals
        self.seasonality = seasonality
        self.seasonal_beta = seasonal_beta
        self.avg = avg
        self.sdev = sdev
        self.linear_beta = linear_beta
    
    def get_time_series(self) -> list:
        stoch_ts = list(np.random.normal(self.avg, self.sdev, self.n_intervals))
        for i, x in enumerate(self.seasonality):
            stoch_ts = add_seasonality(stoch_ts, x, self.seasonal_beta[i])
        return add_multiplicative_trend(stoch_ts, self.linear_beta)
    
    def plot_time_series(self, seasonal_lines = True):
        ts = self.get_time_series()
        plt.plot(ts)
        plt.show()
  
import datetime
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import statsmodels.tsa.seasonal as seas
import statsmodels.tsa.arima_model as arima
import statsmodels.tsa.statespace.sarimax as sarima
import matplotlib.pyplot as plt

# 5 years of fake data with 7, 30, and 365-day seasonality and no trend
my_ts = TimeSeriesData(n_intervals = (365 * 5),
                       seasonality = [7, 30, 365],
                       seasonal_beta = [1.3, 1.1, 1.1],
                       linear_beta = 1.0)

my_ts.plot_time_series()

tseries = my_ts.get_time_series()


# Remove a single component of seasonality
decomp = seas.seasonal_decompose(tseries, model = 'multiplicative', freq = 365)
seas_365 = decomp.seasonal
seas_adj_factor = [1/s for s in seas_365]
tseries_seas_adj = np.array(seas_adj_factor) *  np.array(tseries)
plt.plot(tseries_seas_adj)

# Write a function so we can do this later
def seasonally_adjust(num_list : list, seasonality : int,
                      model_type = 'multiplicative') -> list:
    decomp = seas.seasonal_decompose(num_list, model = 'multiplicative', freq = seasonality)
    adj_factor = [1/s for s in decomp.seasonal]
    return np.array(num_list) * np.array(adj_factor)


# Remove 365 day seasonality
adj_series = seasonally_adjust(num_list = tseries, seasonality = 365)
plt.plot(adj_series)

# Remove 30 day seasonality
adj_series = seasonally_adjust(num_list = adj_series, seasonality = 30)
plt.plot(adj_series)

# Remove 7 day seasonality
adj_series = seasonally_adjust(num_list = adj_series, seasonality = 7)
plt.plot(adj_series)


# Write a function to handle multiple seasonal patterns
def multi_seasonal_adjust(num_list : list, seasonality : int,
                      model_type = 'multiplicative') -> list:
    seas_factors = []
    for s in seasonality:
        decomp = seas.seasonal_decompose(num_list, model = 'multiplicative', freq = s)
        adj_factor = [1/s for s in decomp.seasonal]
        seas_factors.append(np.array(adj_factor))
    return np.prod(seas_factors, axis = 0) * np.array(num_list)


# Remove 7, 30, 365-day seasonality
multi_adj_series = multi_seasonal_adjust(num_list = tseries, seasonality = [365, 30, 7])
plt.plot(multi_adj_series)

plt.hist(multi_adj_series)




def make_date_range(start_date, end_date, date_format = '%Y-%m-%d', increment = 1):
    start = datetime.datetime.strptime(start_date, date_format)
    end = datetime.datetime.strptime(end_date, date_format)
    delta = datetime.timedelta(days = increment)
    date_range = []
    while start < end:
        date_range.append(start.date())
        start += delta
    return date_range
    



ts_df = pd.DataFrame({'value': tseries}, index = make_date_range('2015-01-01', '2019-12-31'))


define_sarima = sarima.SARIMAX(ts_df,
                               order = (1, 0, 0),
                               seasonal_order = (0, 1, 0, 1),
                               freq = 'D')


fit_sarima = define_sarima.fit()
fit_sarima.summary()
prediction_range = make_date_range('2020-01-01', '2024-12-31')
pred = fit_sarima.predict(start = prediction_range[0], end = prediction_range[-1])
plt.plot(pred)
        
        
        
        
        
        
        