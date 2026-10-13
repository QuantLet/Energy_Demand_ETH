"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure_Upload/ETCEnergy_Fig_9.png/.pdf.
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
    EFF_JASMINER_MH_J=ns['EFF_JASMINER_MH_J']
    etc_post_df=ns['etc_post_df']
    plt=ns['plt']
    save_fig=ns['save_fig']
    (fig, ax) = plt.subplots()
    etc_plot = etc_post_df.loc['2022-09-14':'2023-02-28', ['ETC MaxEnergy (GW)', 'ETC Lower Bound (GW)']].copy()
    etc_plot.columns = ['Upper bound — top-down (c_e=$0.10/kWh)', f'Lower bound — bottom-up (Jasminer {EFF_JASMINER_MH_J:.2f} MH/J)']
    etc_plot.plot(ax=ax, alpha=0.85)
    ax.fill_between(etc_plot.index, etc_plot.iloc[:, 1], etc_plot.iloc[:, 0], alpha=0.12, label='Estimated range')
    ax.set_title('ETC Post-Merge Energy Consumption Bounds', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Power (GW)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    save_fig(fig, ax, 'ETCEnergy_Fig_9.png')

if __name__ == "__main__":
    print("Running figure_09_etc_energy.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed figure_09_etc_energy.py", flush=True)
