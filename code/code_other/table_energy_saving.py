"""INPUT: continuous legacy sensitivity frame. OUTPUT: tables/EnergySaving.csv/.tex.
Align all fields on the same valid mask; the former code masked only some fields.
"""

# Resolve package imports when running this file directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_other"

from ..shared.io_utils import save_supplement_table

def generate(ctx,sensitivity):
    valid=sensitivity[1].dropna(subset=['pbu'])
    save_supplement_table('EnergySaving',valid[['ce','pbu','post','pct_mid','SaveQ']])


if __name__ == "__main__":
    print("Running table_energy_saving.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    from ..shared.sensitivity_model import calculate_sensitivity
    generate(ctx, calculate_sensitivity(ctx))
    print("Completed table_energy_saving.py", flush=True)
