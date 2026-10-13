"""INPUT: reconstructed original workbooks via context. OUTPUT: plot inputs in memory.
No frozen result image is copied or used as an analytical input.
"""
import numpy as np
import pandas as pd
from .io_utils import plt,mdates
from .original_plot_style import original_style,save_basic,save_extended,save_multi,add_merge_line,destination

def plot_inputs(ctx):
    original_style()
    df=ctx.legacy.copy();date=ctx.DATE
    df['ETH Profitability Threshold [MHash/J]']=df.ETH_theta
    df['ETC Profitability Threshold [MHash/J]']=df.ETC_theta
    df['MaxEnergy (GW)']=df.ETH_budget_GW
    df['BU CCAF Avg (GW)']=df.BU_equal_units_GW
    df.loc[df.index>date,'BU CCAF Avg (GW)']=0.
    df['ETH Total Revenue per Second (USD)']=df.ETH_revenue_USD_day/86400
    etc_post_df=pd.DataFrame({'ETC MaxEnergy (GW)':df.ETC_budget_GW,
                              'ETC Lower Bound (GW)':df.ETC_lower_GW})
    eff=pd.Series(ctx.efficiencies)
    mig,res=eff[eff>=ctx.L['cutoff']],eff[eff<ctx.L['cutoff']]
    return dict(df=df,etc_post_df=etc_post_df,THE_MERGE_DATE=date,
                THETA_ETC_POST=ctx.L['cutoff'],EFF_JASMINER_MH_J=ctx.L['best_efficiency'],
                DELTA_H=ctx.HM,H_ETC_pre=float(df.loc[ctx.D['pre_start']:date,'ETC / Mean Hash Rate'].mean()),
                LAMBDA_OVH=ctx.L['overhead'],eff_ccaf_avg=float(ctx.L['overhead']*ctx.H0/1000/df.loc[date,'BU CCAF Avg (GW)']),
                eff_all_series=eff,cohorts={'res_series':res,'mig_series':mig,'n_res':len(res),'n_mig':len(mig)},
                e_bar_res=res.mean(),e_bar_mig=mig.mean(),np=np,pd=pd,plt=plt,mdates=mdates,
                save_fig=save_basic,add_merge_line=add_merge_line)
