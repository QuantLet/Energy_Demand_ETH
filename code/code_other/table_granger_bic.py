"""INPUT: BIC-selected supplementary VAR. OUTPUT: tables/tab_granger_causality_BIC.csv/.tex."""

# Resolve package imports when running this file directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_other"

import pandas as pd
from ..shared.io_utils import save_supplement_table

def generate(ctx,fitted):
    rows=[]
    for effect,cause in [('dlog_price','dlog_hashrate'),('dlog_hashrate','dlog_price')]:
        test=fitted.test_causality(effect,[cause],kind='f')
        rows.append({'cause':cause,'effect':effect,'lags':fitted.k_ar,'F_statistic':test.test_statistic,'p_value':test.pvalue})
    save_supplement_table('tab_granger_causality_BIC',pd.DataFrame(rows))


if __name__ == "__main__":
    print("Running table_granger_bic.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    from ..shared.var_bic_model import fit_bic
    generate(ctx, fit_bic(ctx))
    print("Completed table_granger_bic.py", flush=True)
