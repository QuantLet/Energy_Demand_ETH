"""INPUT: BIC-selected supplementary VAR. OUTPUT: Figure/ext_var_irf_BIC.png/.pdf."""

# Resolve package imports when running this file directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_other"

from ..shared.io_utils import save_figure

def generate(ctx, fitted):
    from ..shared.original_plot_style import original_style,save_multi
    original_style()
    from statsmodels.tsa.api import VAR
    import pandas as pd
    original_names=pd.DataFrame(fitted.model.endog,columns=["ΔlnHashRate","ΔlnPrice"])
    fitted=VAR(original_names).fit(fitted.k_ar)
    irf=fitted.irf(periods=10)
    fig=irf.plot(orth=True,signif=0.1,figsize=(10,7))
    fig.suptitle('VAR Orthogonalised IRFs: Hash Rate ↔ Price (PoW era 2018–2021)',fontsize=13,fontweight='bold',y=1.01)
    save_multi(fig,'ext_var_irf_BIC.png')


if __name__ == "__main__":
    print("Running figure_var_bic_irf.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    from ..shared.var_bic_model import fit_bic
    generate(ctx, fit_bic(ctx))
    print("Completed figure_var_bic_irf.py", flush=True)
