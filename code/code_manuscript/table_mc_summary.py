"""INPUT: shared/settings.py scenarios/seed; deduplicated hardware and four-chain inputs.
OUTPUT: Table_Upload/Table_MC_Summary.csv/.tex only.
Eight parameter-only scenarios, 20,000 attempts each. No draw files or MC figures.
"""

# Resolve package imports when running this file directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_manuscript"

import pandas as pd
from ..shared.minimal_core import summary,reference_check
from ..shared.io_utils import save_table


def generate(ctx):
    model=ctx.model
    rows=[]
    for i,spec in enumerate(model.u['scenarios']):
        draws=model.simulate(spec,i,model.u['draws'],model.u['seed'])
        rows.append(summary(draws,spec['key'],spec['name']))
    frame=pd.DataFrame(rows)
    columns=[c for c in frame.columns if c not in ['key','scenario']]
    reference_check(model,frame,'monte_carlo',['key'],columns)
    labels=['U1: Economics only','U2: Composition only',r'U3: Joint; ASIC share 10\%',
            r'U4: Joint; ASIC share 25\%',r'U5: Joint; ASIC share 50\%',
            'U6: U4 + dispersed shares','U7: U4 + correlated economics','U8: U4 + 5-GB devices']
    lines=[[label,f'{r.pre_GW_median:.3f}',f'{r.pre_GW_p025:.3f}--{r.pre_GW_p975:.3f}',
            f'{r.saving_at_budget_pct_median:.2f}',f'{r.saving_at_budget_pct_p025:.2f}--{r.saving_at_budget_pct_p975:.2f}'] for label,r in zip(labels,frame.itertuples())]
    save_table('Table_MC_Summary',frame,r'Specification & Power median (GW) & 2.5--97.5 pct. & Reduction median (\%) & 2.5--97.5 pct.',
               lines,'lrrrr','Parameter-only Monte Carlo summaries based on 20,000 draws per specification.',
               'All default draws are valid. Percentile ranges describe specified parameter distributions, not statistical confidence intervals. U1 varies tariff/overhead; U2 varies composition; U3--U5 vary both; U6--U8 modify U4.', 'tab:mc_summary')


if __name__ == "__main__":
    print("Running table_mc_summary.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed table_mc_summary.py", flush=True)
