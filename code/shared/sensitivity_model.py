"""INPUT: legacy workbook prices/reward schedules and original hardware rows.
OUTPUT: in-memory discrete and continuous electricity-price sensitivity frames.
Preserves the existing ExtendedAnalysis.py ETC-only residual calculation.
No output is written by this shared function; dedicated modules export results.
"""
import numpy as np
import pandas as pd


def calculate_sensitivity(ctx):
    d=ctx.legacy
    # Despite its old label "Block Reward", the original Basic.py field
    # includes the legacy fee term. Use the same reconstructed gross series.
    etc_USD_s=d.ETC_revenue_USD_day/86400
    rev=float(etc_USD_s.loc[ctx.D['stable_start']:ctx.D['stable_end']].mean())
    def point(c):
        threshold=c/.10;eff=ctx.efficiencies
        mig,res=eff[eff>=threshold],eff[eff<threshold]
        power=1e-3*ctx.L['overhead']*(ctx.HM/mig.mean()+ctx.HR/res.mean()) if len(mig) and len(res) else np.nan
        upper=rev/(c/3600000)/1e9;lower=ctx.legacy_etc_lower;mid=(upper+lower)/2
        return threshold,mig,res,power,lower,upper,mid
    discrete=[]
    for c in [.03,.05,.07,.10,.15]:
        th,mig,res,p,lo,hi,mid=point(c)
        td=d.ETH_revenue_USD_day/(24e6*c)
        discrete.append({'c_e ($/kWh)':c,'MaxEnergy at Merge (GW)':round(td.loc[ctx.DATE],2),
                         'MaxEnergy 2021 avg (GW)':round(td.loc['2021-01-01':'2021-12-31'].mean(),2),
                         'MaxEnergy at EIP1559 (GW)':round(td.loc['2021-08-05'],2),
                         'θ_ETC_post (MH/J)':round(th,3),'n_mig':len(mig),'n_res':len(res),
                         'ē_mig (MH/J)':round(mig.mean(),4),'ē_res (MH/J)':round(res.mean(),4),
                         'P_BU_hybrid (GW)':round(p,3),'post_lo (GW)':round(lo,3),
                         'post_hi (GW)':round(hi,3),'post_mid (GW)':round(mid,3),
                         'Saved mid (GW)':round(p-mid,3),'Saved (%)':round(100*(1-mid/p),1)})
    fine=[]
    for c in np.linspace(.02,.18,200):
        th,mig,res,p,lo,hi,mid=point(c)
        fine.append({'ce':c,'theta':th,'pbu':p,'post':mid,'post_lo':lo,'post_hi':hi,
                     'pct_mid':100*(1-mid/p),'pct_lo':100*(1-hi/p),
                     'pct_hi':100*(1-lo/p),'SaveQ':p-mid})
    return pd.DataFrame(discrete),pd.DataFrame(fine)
