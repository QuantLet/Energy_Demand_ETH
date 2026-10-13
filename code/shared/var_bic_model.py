"""INPUT: original Coin Metrics daily log differences. OUTPUT: in-memory BIC VAR.
Preserves the existing extension's maximum ten lags; main results use VAR(14).
"""
import numpy as np
import statsmodels.api as sm

def fit_bic(ctx):
    d=np.log(ctx.cm[['ETH / Mean Hash Rate','ETH / USD Denominated Closing Price']].replace(0,np.nan)).diff().loc[ctx.D['var_start']:ctx.D['var_end']].dropna()
    d.columns=['dlog_hashrate','dlog_price']
    return sm.tsa.VAR(d.asfreq('D')).fit(maxlags=10,ic='bic')
