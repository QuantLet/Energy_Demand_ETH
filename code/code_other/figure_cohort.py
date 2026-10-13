"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure/fig_cohort_split.png/.pdf.
"""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_other"
from ..shared.original_plot_inputs import plot_inputs
from ..shared.original_plot_style import destination,save_basic,save_extended,save_multi,save_png

def generate(ctx):
    ns=plot_inputs(ctx)
    THETA_ETC_POST=ns['THETA_ETC_POST']
    cohorts=ns['cohorts']
    e_bar_mig=ns['e_bar_mig']
    e_bar_res=ns['e_bar_res']
    eff_all_series=ns['eff_all_series']
    np=ns['np']
    plt=ns['plt']
    (fig, ax) = plt.subplots(figsize=(8, 6.5))
    bins = np.linspace(0, float(eff_all_series.max()) + 0.1, 40)
    ax.hist(cohorts['res_series'].values, bins=bins, alpha=0.6, color='steelblue', label=f"Residual (e < {THETA_ETC_POST}, n={cohorts['n_res']}, ē={e_bar_res:.3f})")
    ax.hist(cohorts['mig_series'].values, bins=bins, alpha=0.6, color='darkorange', label=f"Migrating (e ≥ {THETA_ETC_POST}, n={cohorts['n_mig']}, ē={e_bar_mig:.3f})")
    ax.axvline(THETA_ETC_POST, color='black', linestyle='--', linewidth=1.5, label=f'θ_ETC = {THETA_ETC_POST} MH/J')
    ax.axvline(e_bar_mig, color='darkorange', linestyle=':', linewidth=1.5, label=f'ē_mig = {e_bar_mig:.3f} MH/J')
    ax.axvline(e_bar_res, color='steelblue', linestyle=':', linewidth=1.5, label=f'ē_res = {e_bar_res:.3f} MH/J')
    ax.set_title('CCAF Hardware Inventory — Cohort Split', fontweight='bold')
    ax.set_xlabel('Device Efficiency (MH/J)')
    ax.set_ylabel('Device Count')
    _x_max = float(eff_all_series.max())
    ax.set_xlim(0.0, _x_max * 1.05)
    _ticks = np.arange(0.0, _x_max + 0.5, 0.5)
    ax.set_xticks(_ticks)
    ax.set_xticklabels([f'{t:.1f}' for t in _ticks])
    (handles, labels) = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ncol = min(len(labels), 4)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=ncol, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.5)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save_png(fig,destination('fig_cohort_split.png'), dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(destination('fig_cohort_split.png').with_suffix('.pdf'), bbox_inches='tight', transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    print("Running figure_cohort.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    generate(ctx)
    print("Completed figure_cohort.py", flush=True)
