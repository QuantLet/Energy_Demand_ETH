"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure_Upload/fig_profitability_threshold_Fig_7.png/.pdf.
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
    THETA_ETC_POST=ns['THETA_ETC_POST']
    add_merge_line=ns['add_merge_line']
    df=ns['df']
    mdates=ns['mdates']
    pd=ns['pd']
    plt=ns['plt']
    save_fig=ns['save_fig']
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    data = df.loc['2021-01-01':'2022-12-31']
    (fig, (ax1, ax2)) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0.05})
    ax1.plot(data.index, data['ETH Profitability Threshold [MHash/J]'], color='tab:blue', alpha=0.9)
    add_merge_line(ax1)
    ax1.set_ylabel('ETH efficiency')
    ax1.set_title('Miner Profitability Thresholds(MHash/J) — ETH vs ETC', fontweight='bold')
    ax2.plot(data.index, data['ETC Profitability Threshold [MHash/J]'], color='tab:orange', alpha=0.9)
    add_merge_line(ax2)
    ax2.axhline(THETA_ETC_POST, color='gray', linestyle=':', linewidth=1.0)
    ax2.set_ylabel('ETC efficiency')
    ax2.set_xlabel('Year')
    ax2.set_xlim(data.index[0], pd.Timestamp('2022-12-31'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(labelbottom=False)
    fig.tight_layout()
    save_fig(fig, ax1, 'fig_profitability_threshold_Fig_7.png')

if __name__ == "__main__":
    print("Running figure_07_threshold.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed figure_07_threshold.py", flush=True)
