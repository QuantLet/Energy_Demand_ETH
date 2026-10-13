"""Original plot statements preserved. INPUT: original workbook calculations.
OUTPUT: Figure/ext_sensitivity_ce_series.png/.pdf.
"""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_other"
from ..shared.original_plot_inputs import plot_inputs
from ..shared.original_plot_style import destination,save_basic,save_extended,save_multi

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
    P_BU=ns['P_BU']
    df=ns['df']
    plt=ns['plt']
    save_fig=ns['save_fig']
    sens_df=ns['sens_df']
    (fig, ax) = plt.subplots()
    for (ce_kwh, color) in zip(CE_GRID_KWH, CE_COLORS):
        ce_j = ce_kwh / 3600000
        series = df.loc['2019-01-01':'2022-09-14', 'ETH Total Revenue per Second (USD)'] / ce_j / 1000000000.0
        ax.plot(series.index, series.values, color=color, lw=1.4, alpha=0.85, label=f'MaxEnergy c_e=${ce_kwh}/kWh')
    pbu_vals = sens_df['P_BU_hybrid (GW)'].dropna()
    ax.axhspan(pbu_vals.min(), pbu_vals.max(), alpha=0.12, color='black', label=f'Hybrid P_BU range [{pbu_vals.min():.2f}–{pbu_vals.max():.2f} GW]')
    ax.axhline(P_BU, color='black', linestyle='--', lw=1.4, label=f'Hybrid P_BU baseline ($0.10/kWh) = {P_BU:.2f} GW')
    pm_lo = sens_df['post_lo (GW)'].min()
    pm_hi = sens_df['post_hi (GW)'].max()
    ax.axhspan(pm_lo, pm_hi, alpha=0.15, color='mediumpurple', label=f'Post-Merge range (ETC+ETHW) [{pm_lo:.3f}–{pm_hi:.3f} GW]')
    ax.set_title('Sensitivity: Top-Down, Hybrid P_BU, and Post-Merge Power vs c_e', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Power (GW)')
    save_fig(fig, ax, 'ext_sensitivity_ce_series.png')

if __name__ == "__main__":
    print("Running figure_sensitivity_series.py", flush=True)
    # Independent execution: load required inputs, then export only this result.
    from ..shared.context import load_context
    ctx = load_context()
    from ..shared.sensitivity_model import calculate_sensitivity
    generate(ctx, calculate_sensitivity(ctx))
    print("Completed figure_sensitivity_series.py", flush=True)
