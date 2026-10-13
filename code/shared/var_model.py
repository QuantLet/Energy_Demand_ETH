"""INPUT: original Coin Metrics workbook via context.cm.
OUTPUT: fixed VAR(14) model shared by Figure 11 and the Granger table.
No lag selection, FEVD, alternative VAR models or diagnostic exports.
"""
import numpy as np
import statsmodels.api as sm


def fit_var(ctx):
    daily=np.log(ctx.cm[['ETH / Mean Hash Rate','ETH / USD Denominated Closing Price']].replace(0,np.nan)).diff().loc[ctx.D['var_start']:ctx.D['var_end']].dropna()
    daily.columns=['dlog_hashrate','dlog_price']
    assert daily.index.equals(__import__('pandas').date_range(daily.index.min(),daily.index.max()))
    fitted=sm.tsa.VAR(daily.asfreq('D')).fit(14)
    assert fitted.nobs==1386
    return fitted
