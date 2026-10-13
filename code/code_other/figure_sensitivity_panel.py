"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure/fig_panel_legends_bottom.png/.pdf.
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
    ce_fine=ns['ce_fine']
    np=ns['np']
    pbu_fine=ns['pbu_fine']
    pct_mid_fine=ns['pct_mid_fine']
    pd=ns['pd']
    plt=ns['plt']
    post_mid_fine=ns['post_mid_fine']
    theta_fine=ns['theta_fine']
    (fig, axes) = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    axes[0].plot(ce_fine, theta_fine, color='steelblue', lw=2, label='θ(c_e)')
    axes[0].set_title('(a) Threshold θ', fontweight='bold')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=9, ncol=2)
    valid = ~np.isnan(pbu_fine)
    out = pd.DataFrame({'ce': ce_fine[valid], 'pbu': pbu_fine[valid], 'post': post_mid_fine, 'pct_mid': pct_mid_fine[valid], 'SaveQ': pbu_fine[valid] - post_mid_fine})
    axes[1].plot(ce_fine[valid], pbu_fine[valid], color='tomato', lw=2, label='Before-Merge ETH')
    axes[1].plot(ce_fine, post_mid_fine, color='mediumpurple', lw=2, label='Post-Merge ETC')
    axes[1].set_title('(b) Power', fontweight='bold')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), frameon=False, fontsize=9, ncol=2)
    panel3 = pd.DataFrame({'ce': ce_fine[valid], 'quantity': pbu_fine[valid] - post_mid_fine, 'pct_mid': pct_mid_fine[valid]})
    ax2 = axes[2]
    ax2.plot(panel3['ce'], panel3['pct_mid'], lw=2, color='black', label='Power savings ratio (%)')
    ax2_right = ax2.twinx()
    ax2_right.plot(panel3['ce'], panel3['quantity'], lw=2, linestyle='--', label='Power saving amount (GW)')
    ax2_right.set_ylabel('Power saving amount (GW)', fontsize=10)
    ax2_right.spines['right'].set_visible(True)
    ax2_right.tick_params(axis='y', which='both', right=True, labelright=True, direction='out', length=4, width=0.8)
    ax2.set_ylim(panel3['pct_mid'].min() - 0.02, panel3['pct_mid'].max() + 0.02)
    ax2_right.set_ylim(panel3['quantity'].min() * 0.9, panel3['quantity'].max() * 1.1)
    (lines, labels) = ax2.get_legend_handles_labels()
    (lines2, labels2) = ax2_right.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=9, ncol=2)
    axes[2].set_title('(c) Energy Savings', fontweight='bold')
    axes[3].axis('off')
    for ax in axes[:3]:
        ax.set_xlabel('Electricity price c_e ($/kWh)')
        ax.grid(False)
        ax.set_facecolor('none')
    axes[0].set_ylabel('Threshold θ')
    axes[1].set_ylabel('Power (GW)')
    axes[2].set_ylabel('Power savings ratio (%)')
    fig.tight_layout(h_pad=3.0)
    save_png(fig,destination('fig_panel_legends_bottom.png'), dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(destination('fig_panel_legends_bottom.png').with_suffix('.pdf'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    print("Running figure_sensitivity_panel.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    from ..shared.sensitivity_model import calculate_sensitivity
    generate(ctx, calculate_sensitivity(ctx))
    print("Completed figure_sensitivity_panel.py", flush=True)
