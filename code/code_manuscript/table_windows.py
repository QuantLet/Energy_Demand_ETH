"""INPUT: model daily hashrates/revenues and reference hardware classification.
OUTPUT: Table_Upload/windows.csv/.tex only (six summary rows).
The 240/180 window specifications are calculated in memory. No economic grid,
window plot, specification-detail CSV or other auxiliary output is generated.
"""

# Resolve package imports when running this file directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_manuscript"

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from ..shared.minimal_core import window,MinimalModel
from ..shared.io_utils import save_table

def calculate_windows(model: MinimalModel):
    rows = []
    stats = {'mean': np.mean, 'median': np.median, 'trimmed_mean_10pct': lambda x: trim_mean(x, 0.1)}
    for pre_days in [3, 7, 14, 30]:
        pre_start = model.date - pd.Timedelta(days=pre_days)
        pre = window(model.daily[['ETC_hashrate_TH_s']], str(pre_start.date()), str(model.date.date())).iloc[:, 0]
        for offset in [1, 7, 30, 60]:
            start = model.date + pd.Timedelta(days=offset)
            for duration in [7, 14, 30, 60, 90]:
                end = start + pd.Timedelta(days=duration)
                post = window(model.daily[['ETC_hashrate_TH_s']], str(start.date()), str(end.date())).iloc[:, 0]
                try:
                    rev = window(model.revenue, str(start.date()), str(end.date()))
                    sync_revenue = float(rev.mean().sum())
                    sync_theta = float(np.mean(post / (1000 * (rev.ETC / (24000000.0 * 0.1)))))
                    sync_status = 'complete'
                except ValueError as exc:
                    sync_revenue = np.nan
                    sync_theta = np.nan
                    sync_status = str(exc)
                for (statistic, func) in stats.items():
                    hm = float(func(post.to_numpy()) - func(pre.to_numpy()))
                    fixed = model.deterministic(hm=hm)
                    sync_budget = sync_revenue / (24000000.0 * model.u['price'])
                    row = {'pre_lookback_days': pre_days, 'post_offset_days': offset, 'post_duration_days': duration, 'statistic': statistic, 'pre_start': str(pre_start.date()), 'post_start': str(start.date()), 'post_end': str(end.date()), 'pre_observations': len(pre), 'post_observations': len(post), 'delta_h_TH_s': hm, 'pre_GW': fixed['pre_GW'], 'fixed_post_budget_GW': fixed['post_budget_GW'], 'fixed_saving_pct': fixed['saving_at_budget_pct'], 'valid': fixed['valid'], 'sync_post_budget_GW': sync_budget, 'sync_saving_pct': 100 * (1 - sync_budget / fixed['pre_GW']), 'sync_status': sync_status, 'is_baseline': pre_days == 7 and offset == 60 and (duration == 90) and (statistic == 'mean')}
                    if np.isfinite(sync_theta):
                        linked = model.deterministic(hm=hm, revenue=sync_revenue, theta_post=sync_theta)
                        row.update({'sync_reclassified_pre_GW': linked['pre_GW'], 'sync_reclassified_saving_pct': linked['saving_at_budget_pct'], 'sync_reclassified_valid': linked['valid']})
                    else:
                        row.update({'sync_reclassified_pre_GW': np.nan, 'sync_reclassified_saving_pct': np.nan, 'sync_reclassified_valid': False})
                    rows.append(row)
    return pd.DataFrame(rows)

def generate(ctx):
    rows=calculate_windows(ctx.model)
    overview=[]
    for col in ['delta_h_TH_s','pre_GW','fixed_saving_pct','sync_post_budget_GW','sync_saving_pct','sync_reclassified_saving_pct']:
        vals=rows[col].dropna()
        overview.append({'metric':col,'valid_specifications':len(vals),'minimum':vals.min(),'median':vals.median(),'maximum':vals.max()})
    frame=pd.DataFrame(overview)
    assert frame.valid_specifications.tolist()==[240,240,240,180,180,180]
    labels=['ETC increment (TH/s)','Pre-Merge power (GW)',r'Fixed-budget saving (\%)',
            'Synchronized budget (GW)',r'Synchronized saving (\%)',r'Reclassified saving (\%)']
    save_table('windows',frame,'Quantity & Valid & Minimum & Median & Maximum',
               [[label,int(r.valid_specifications),f'{r.minimum:.3f}',f'{r.median:.3f}',f'{r.maximum:.3f}'] for label,r in zip(labels,frame.itertuples())],
               'lrrrr','Accounting reductions across alternative window specifications.',
               'Specification minima, medians and maxima, not confidence intervals. Isolated: 240 specifications; synchronized and reclassified: 180 because earliest windows lack ETHW revenue.', 'uqadd:tabwindows')


if __name__ == "__main__":
    print("Running table_windows.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed table_windows.py", flush=True)
