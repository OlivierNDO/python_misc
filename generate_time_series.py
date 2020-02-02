# Configuration
###############################################################################

# Packages
import matplotlib.pyplot as plt
import numpy as np

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
        linear_beta: a multiplicative linear trend to apply to the entire series
        
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