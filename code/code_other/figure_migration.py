"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure/fig_hashrate_migration.png/.pdf.
"""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_other"
from ..shared.original_plot_inputs import plot_inputs
from ..shared.original_plot_style import destination,save_basic,save_extended,save_multi

def generate(ctx):
    ns=plot_inputs(ctx)
    df_merge=ns["df"].loc[ctx.DATE-__import__("pandas").Timedelta(days=30):ctx.DATE+__import__("pandas").Timedelta(days=30)]
    add_merge_line=ns['add_merge_line']
    plt=ns['plt']
    save_fig=ns['save_fig']
    (fig, ax) = plt.subplots()
    df_merge[['ETH / Mean Hash Rate', 'ETC / Mean Hash Rate']].plot(ax=ax, alpha=0.85)
    add_merge_line(ax)
    ax.set_title('Hash Rate Migration Around The Merge (±30 days)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Hash Rate (TH/s)')
    ax.legend(['ETH hash rate', 'ETC hash rate', 'The Merge'])
    save_fig(fig, ax, 'fig_hashrate_migration.png')

if __name__ == "__main__":
    print("Running figure_migration.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed figure_migration.py", flush=True)
