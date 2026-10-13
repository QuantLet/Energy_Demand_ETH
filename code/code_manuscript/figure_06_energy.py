"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure_Upload/fig_energy_td_vs_bu_Fig_6.png/.pdf.
"""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_manuscript"
from ..shared.original_plot_inputs import plot_inputs
from ..shared.original_plot_style import destination,save_basic,save_extended,save_multi

def generate(ctx):
    ns=plot_inputs(ctx)
    ns['plot_start']='2017-01-01'
    ns['plot_end']='2022-09-30'
    LAMBDA_OVH=ns['LAMBDA_OVH']
    add_merge_line=ns['add_merge_line']
    df=ns['df']
    eff_ccaf_avg=ns['eff_ccaf_avg']
    plot_end=ns['plot_end']
    plot_start=ns['plot_start']
    plt=ns['plt']
    save_fig=ns['save_fig']
    (fig, ax) = plt.subplots()
    ax.plot(df.loc[plot_start:plot_end].index, df.loc[plot_start:plot_end, 'MaxEnergy (GW)'], color='tomato', lw=1.6, alpha=0.85, label='Top-down upper bound (100% rev., $0.05/kWh)')
    ax.plot(df.loc[plot_start:plot_end].index, df.loc[plot_start:plot_end, 'BU CCAF Avg (GW)'], color='steelblue', lw=1.6, alpha=0.85, label=f'Bottom-up (CCAF eff={eff_ccaf_avg:.3f} MH/J, ×λ_ovh={LAMBDA_OVH})')
    add_merge_line(ax)
    ax.set_title('ETH Energy: Top-Down vs Bottom-Up', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Power (GW)')
    save_fig(fig, ax, 'fig_energy_td_vs_bu_Fig_6.png')

if __name__ == "__main__":
    print("Running figure_06_energy.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed figure_06_energy.py", flush=True)
