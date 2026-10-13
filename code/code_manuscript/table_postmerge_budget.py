"""INPUT: legacy ETC workbook budget + ETHW/RVN/ERG stable-window revenues.
OUTPUT: Table_Upload/Table_PostMerge_Budget.csv/.tex only.
Historical retained-table convention; not a common-window ETC reconstruction.
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


def generate(ctx):
    budgets=ctx.model.stable_revenue.mean()/(24e6*.10)
    budgets.loc['ETC']=ctx.legacy_etc_upper
    frame=budgets.rename('budget_GW').rename_axis('chain').reset_index()
    frame['basis']=['Legacy ETC workbook revenue budget' if c=='ETC' else 'Stable: 2022-11-13 to 2023-02-11' for c in frame.chain]
    np.testing.assert_allclose(budgets.loc[['ETC','ETHW','RVN','ERG']],
                              [.13917460101243842,.019331659172229714,.03483938515105708,.019276409232223426],rtol=1e-10)
    save_table('Table_PostMerge_Budget',frame,'Network & Electricity budget (GW)',
               [[c,f'{v:.4f}'] for c,v in budgets.items()]+[['Four-chain total',f'{budgets.sum():.3f}']],
               'lr','Post-Merge revenue-supported electricity budgets at USD 0.10/kWh.',
               'Preserves the retained historical convention: original ETC workbook budget, plus stable-window ETHW/RVN/ERG budgets. The ETC result 0.139174601 rounds to 0.1392; Results\_keep.rtf truncates it to 0.1391. Whole-network budgets are not measured consumption.', 'tab:postmerge_accounting')


if __name__ == "__main__":
    print("Running table_postmerge_budget.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed table_postmerge_budget.py", flush=True)
