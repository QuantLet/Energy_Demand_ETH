"""INPUT: settings.py, four original Excel workbooks and five extension CSVs.
OUTPUT: in-memory legacy series/model inputs only; no extra result files.
Shared calculations required by Figures 2/6/7/9/11 and the post-Merge table.
Algorithms are preserved from the previous unified notebook.
"""
from pathlib import Path
from types import SimpleNamespace
from .settings import DEFINITIONS
import numpy as np
import pandas as pd
from .io_utils import ROOT
from .minimal_core import MinimalModel, budget as revenue_budget_GW, window

def threshold_MH_J(hashrate_TH_s,revenue_USD_day,price_USD_kWh):
    return np.asarray(hashrate_TH_s)/(1000*revenue_budget_GW(revenue_USD_day,price_USD_kWh))

def load_context():
    definitions=DEFINITIONS
    CONFIG=definitions['CONFIG']; L=CONFIG['legacy']; D=CONFIG['dates']
    DATE=pd.Timestamp(D['last_pow']);MERGE=pd.Timestamp(D['merge'])
    START,END=D['stable_start'],D['stable_end'];PRE_START=D['pre_start']
    ORIG=ROOT/'data/original'
    original=pd.read_excel(ORIG/'Ethereum+.xlsx',index_col=0).sort_index()
    original.index=pd.to_datetime(original.index)
    cm=pd.read_excel(ORIG/'Coin_Metrics_Network_Data_2023-03-14T12-53.xlsx',index_col=0).sort_index()
    cm.index=pd.to_datetime(cm.index)
    raw_catalogue=pd.concat([pd.read_excel(ORIG/'EthereumHardware.xlsx'),
                             pd.read_excel(ORIG/'EthereumHardwareGPU.xlsx')],ignore_index=True)
    legacy=original.copy()
    eth_reward=pd.Series(5.,index=legacy.index)
    eth_reward.loc[eth_reward.index>pd.Timestamp('2017-10-16')]=3.
    eth_reward.loc[eth_reward.index>pd.Timestamp('2019-02-28')]=2.
    eth_reward.loc[eth_reward.index>DATE]=0.
    etc_reward=pd.Series(5.,index=legacy.index)
    for date,reward in [('2017-12-11',4.),('2020-03-17',3.2),('2021-04-25',2.56)]:
        etc_reward.loc[etc_reward.index>pd.Timestamp(date)]=reward
    legacy['ETH_revenue_USD_day']=(eth_reward/L['eth_block_seconds']+
     legacy['ETH / Mean Tx Fee (native units)']*legacy['ETH / Tx per Second Cnt'])*legacy['ETH / USD Denominated Closing Price']*86400
    legacy['ETC_revenue_USD_day']=(etc_reward/L['etc_block_seconds']+
     legacy['ETC / Mean Tx Fee (native units)'].div(legacy['ETC / Tx per Second Cnt'].replace(0,np.nan)).fillna(0))*legacy['ETC / USD Denominated Closing Price']*86400
    legacy['ETH_budget_GW']=revenue_budget_GW(legacy.ETH_revenue_USD_day,L['eth_price'])
    legacy['ETC_budget_GW']=revenue_budget_GW(legacy.ETC_revenue_USD_day,L['etc_price'])
    legacy['ETH_theta']=threshold_MH_J(legacy['ETH / Mean Hash Rate'],legacy.ETH_revenue_USD_day,L['eth_price'])
    legacy.loc[legacy.index>DATE,'ETH_theta']=0.
    legacy['ETC_theta']=threshold_MH_J(legacy['ETC / Mean Hash Rate'],legacy.ETC_revenue_USD_day,L['etc_price'])
    legacy['ETC_lower_GW']=1e-3*legacy['ETC / Mean Hash Rate']/L['best_efficiency']
    original_H0=float(legacy.loc[DATE,'ETH / Mean Hash Rate'])
    original_HM=float(window(legacy,START,END,['ETC / Mean Hash Rate']).mean().iloc[0]-window(legacy,PRE_START,D['last_pow'],['ETC / Mean Hash Rate']).mean().iloc[0])
    original_HR=original_H0-original_HM
    original_eff=raw_catalogue['Efficiency_Mh/J'].to_numpy()
    original_mig=original_eff>=L['cutoff']
    original_e_mig=original_eff[original_mig].mean()
    original_e_res=original_eff[~original_mig].mean()
    original_power=1e-3*L['overhead']*(original_HM/original_e_mig+original_HR/original_e_res)
    raw_h=raw_catalogue['Hashing power (Mh/s)'].to_numpy()
    legacy_eff_equal_units=raw_h.sum()/np.sum(raw_h/original_eff)
    legacy['BU_equal_units_GW']=1e-3*L['overhead']*legacy['ETH / Mean Hash Rate']/legacy_eff_equal_units
    legacy_etc_upper=float(legacy.loc[START:END,'ETC_budget_GW'].mean())
    return SimpleNamespace(legacy=legacy,L=L,D=D,DATE=DATE,MERGE=MERGE,
                           original_power=original_power,legacy_etc_upper=legacy_etc_upper,
                           cm=cm,model=MinimalModel(ROOT),H0=original_H0,HM=original_HM,HR=original_HR,
                           efficiencies=original_eff,legacy_etc_lower=float(legacy.loc[START:END,"ETC_lower_GW"].mean()))
