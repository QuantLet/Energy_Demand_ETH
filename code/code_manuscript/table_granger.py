"""INPUT: fixed VAR(14), 1,386 observations from the original workbook.
OUTPUT: Table_Upload/Table_Granger.csv/.tex only.
"""

# Resolve package imports when running this file directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_manuscript"

import numpy as np
import pandas as pd
from ..shared.io_utils import save_table


def generate(ctx, fitted):
    rows=[]
    for effect,cause in [('dlog_price','dlog_hashrate'),('dlog_hashrate','dlog_price')]:
        test=fitted.test_causality(effect,[cause],kind='f')
        rows.append({'cause':cause,'effect':effect,'F_statistic':test.test_statistic,'p_value':test.pvalue,'lags':14,'nobs':fitted.nobs})
    frame=pd.DataFrame(rows)
    np.testing.assert_allclose(frame.F_statistic,[.93623187001106,5.048111687314943],rtol=1e-10)
    labels=[r'$\Delta\ln(H)$ does not Granger-cause $\Delta\ln(p^{ETH})$',r'$\Delta\ln(p^{ETH})$ does not Granger-cause $\Delta\ln(H)$']
    save_table('Table_Granger',frame,'Null hypothesis & F-statistic & p-value',
               [[label,f'{r.F_statistic:.3f}',f'{r.p_value:.3g}'] for label,r in zip(labels,frame.itertuples())],
               'lrr','Granger causality tests from the fixed VAR(14).',
               'Daily log differences, January 1, 2018--October 31, 2021; 1,386 usable observations. Granger causality denotes conditional predictive content.', 'tab:granger')


if __name__ == "__main__":
    print("Running table_granger.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    from ..shared.var_model import fit_var
    generate(ctx, fit_var(ctx))
    print("Completed table_granger.py", flush=True)
