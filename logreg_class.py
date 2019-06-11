# Config
#################################################################
from sklearn import datasets
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# Load Iris Dataset & Limit to Two Classes
#################################################################
iris = datasets.load_iris()
iris_df = pd.concat([pd.DataFrame({'y': iris.target}),
                    pd.DataFrame(iris.data)],
                    axis = 1)
iris_df = iris_df[iris_df['y'] < 2]
iris_df = iris_df.iloc[:,0:3]
iris_df.columns = ['y','x0', 'x1']

# Define Functions        
#################################################################
class sm_logit_assumptions:
    """check parametric assumptions of logistic regression fit with statsmodels api"""
    
    def __init__(self, df, y_col, x_cols,
                 fit_intercept = False):
        self.df = df
        self.y_col = y_col
        self.x_cols = x_cols
        self.fit_intercept = fit_intercept
    
    def unnest_list_of_lists(LOL):
        # unnest list of lists
        return list(itertools.chain.from_iterable(LOL))
    
    def return_model(self):
        # return fit model object
        if self.fit_intercept:
            LR = sm.Logit(self.df[self.y_col], sm.add_constant(self.df[self.x_cols]))
        else:
            LR = sm.Logit(self.df[self.y_col], sm.add_constant(self.df[self.x_cols]))
        return LR
    
    def binary_response(self):
        # check that response variable is binary
        num_response_levels = len(set(self.df[self.y_col]))
        if num_response_levels == 2:
            assumption_met = "assumption met: {yvar} variable has 2 unique values".format(yvar = self.y_col)
        else:
            assumption_met = "assumption violated: {yvar} variable has {n} unique values".format(yvar = self.y_col,
                                                   n = str(int(num_response_levels)))
        return assumption_met
    
    def correlation_matrix(self):
        # check for multicollinearity in predictor variables
        corr_df = pd.DataFrame(self.df[self.x_cols].corr())
        np.fill_diagonal(corr_df.values, 0)
        highest_corr = max([abs(x) for x in unnest_list_of_lists(corr_df.values.tolist())])
        print("max (abs.) correlation: {cu}".format(cu = str(np.round(highest_corr,3))))
        return corr_df
    
    def coefficients_and_summary(self):
        # fit & return coefficients + descriptive statistics
        if self.fit_intercept:
            lr = sm.Logit(self.df[self.y_col], sm.add_constant(self.df[self.x_cols])).fit()
        else:
            lr = sm.Logit(self.df[self.y_col], self.df[self.x_cols]).fit()
        return lr.summary()

    def residual_distribution(self):
        # plot residual distribution
        if self.fit_intercept:
            lr = sm.Logit(self.df[self.y_col], sm.add_constant(self.df[self.x_cols])).fit()
        else:
            lr = sm.Logit(self.df[self.y_col], self.df[self.x_cols]).fit()
        pred_probs = lr.predict(self.df[self.x_cols])
        residuals = [(self.df[self.y_col][i] - p) for i, p in enumerate(pred_probs)]
        plt.hist(residuals, bins = 30)
        plt.title("Residual Distribution")
        
# Execute Functions        
#################################################################
fitted_logreg = sm_logit_assumptions(df = iris_df,
                                     y_col = 'y',
                                     x_cols = ['x0','x1'])
        
fitted_logreg.binary_response()
fitted_logreg.correlation_matrix()
fitted_logreg.coefficients_and_summary()
fitted_logreg.residual_distribution()