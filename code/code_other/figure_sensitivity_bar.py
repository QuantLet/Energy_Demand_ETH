"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure/ext_sensitivity_ce_bar.png/.pdf.
"""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_other"
from ..shared.original_plot_inputs import plot_inputs
from ..shared.original_plot_style import destination,save_basic,save_extended,save_multi,save_png

def generate(ctx,sensitivity):
    ns=plot_inputs(ctx)
    discrete,fine=sensitivity
    ns.update(sens_df=discrete,CE_GRID_KWH=[.03,.05,.07,.10,.15],
              CE_COLORS=['#d62728','#ff7f0e','#1f77b4','#2ca02c','#9467bd'],P_BU=ctx.original_power,
              ce_fine=fine.ce.to_numpy(),theta_fine=fine.theta.to_numpy(),pbu_fine=fine.pbu.to_numpy(),
              post_mid_fine=fine.post.to_numpy(),post_lo_fine=fine.post_lo.to_numpy(),post_hi_fine=fine.post_hi.to_numpy(),
              pct_mid_fine=fine.pct_mid.to_numpy(),pct_lo_fine=fine.pct_lo.to_numpy(),pct_hi_fine=fine.pct_hi.to_numpy(),
              save_fig=save_extended)
    CE_COLORS=ns['CE_COLORS']
    CE_GRID_KWH=ns['CE_GRID_KWH']
    np=ns['np']
    plt=ns['plt']
    sens_df=ns['sens_df']
    (fig, ax) = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CE_GRID_KWH))
    w = 0.26
    ax.bar(x - w, sens_df['MaxEnergy at Merge (GW)'], w, color=CE_COLORS, alpha=0.75, label='MaxEnergy at Merge (pre-Merge TD)')
    ax.bar(x, sens_df['P_BU_hybrid (GW)'], w, color=CE_COLORS, alpha=0.55, edgecolor='black', linewidth=0.8, label='P_BU hybrid (pre-Merge, c_e-adjusted θ)')
    ax.bar(x + w, sens_df['post_mid (GW)'], w, color=CE_COLORS, alpha=0.3, edgecolor='black', linewidth=0.8, label='Post-Merge mid (ETC+ETHW, ETC LB fixed)')
    err_lo = (sens_df['post_mid (GW)'] - sens_df['post_lo (GW)']).clip(lower=0)
    err_hi = (sens_df['post_hi (GW)'] - sens_df['post_mid (GW)']).clip(lower=0)
    ax.errorbar(x + w, sens_df['post_mid (GW)'], yerr=[err_lo.values, err_hi.values], fmt='none', color='black', capsize=3, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f'${c}/kWh' for c in CE_GRID_KWH])
    ax.set_xlabel('Electricity price assumption c_e')
    ax.set_ylabel('Power (GW)')
    ax.set_title('Pre-Merge and Post-Merge Power by Electricity Price Scenario\n(hybrid varies via θ_ETC → cohort split; post-Merge TD scales with c_e)', fontweight='bold')
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    (handles, labels) = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.0, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    save_png(fig,destination('ext_sensitivity_ce_bar.png'), dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(destination('ext_sensitivity_ce_bar.png').with_suffix('.pdf'), bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'    Saved: ext_sensitivity_ce_bar.png')

if __name__ == "__main__":
    print("Running figure_sensitivity_bar.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    from ..shared.sensitivity_model import calculate_sensitivity
    generate(ctx, calculate_sensitivity(ctx))
    print("Completed figure_sensitivity_bar.py", flush=True)
