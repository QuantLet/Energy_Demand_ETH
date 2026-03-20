"""
ExtendedAnalysis.py
===================
Four extended analyses building on EthereumEnergy_clean_wrt.py outputs.
Central question: How much power was saved by the ETH Merge?

Sections
--------
1.  Hybrid robustness tests
    1a. Bootstrap on CCAF hardware inventory (P_BU sampling uncertainty)
    1b. ΔH window sensitivity (offset × length grid)
    1c. θ cutoff sensitivity (migration/resident cohort split)
    1d. VAR Granger causality (hash rate ↔ price dynamics)
    1e. EIP-1559 structural break (miner revenue composition)
2.  Post-Merge total power (ETC + ETHW time series + savings chart)
3.  Economic significance (TWh/yr, CO₂ avoided, cost savings; PoS vs PoW)
4.  Parameter sensitivity + Monte Carlo
    4a. Electricity price c_e ∈ {0.03, 0.05, 0.07, 0.10} $/kWh
    4b. Tornado chart (one-at-a-time P_BU sensitivity)
    4c. Monte Carlo (joint uncertainty over λ, ΔH, ē_mig, ē_res, post-Merge power)

Prerequisites
-------------
Run  python code/EthereumEnergy_clean.py  first.
Outputs are read from  data/processed/  and written to
  Figures/extended/  and  tables/extended/.
"""

# =============================================================================
# SECTION 0: Setup
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

np.random.seed(42)

# --- Paths ---
SCRIPT_DIR = Path("/Users/ruting/Documents/macbook/PcBack/25.The Energy Consumption of the Ethereum-Ecosystem /Energy_Demand_ETH/code")
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR      = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR   = PROJECT_DIR / "Figures" / "extended"
TABLES_DIR    = PROJECT_DIR / "tables" / "extended"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# --- Load processed data ---
print("=== ExtendedAnalysis.py ===")
print("[0] Loading processed data ...")

df     = pd.read_parquet(PROCESSED_DIR / "main_df.parquet")
var_df = pd.read_parquet(PROCESSED_DIR / "var_input.parquet")

with open(PROCESSED_DIR / "model_params.json") as fh:
    p = json.load(fh)

# Restore key scalars
THE_MERGE_DATE     = pd.Timestamp(p["THE_MERGE_DATE"])
EIP1559_DATE       = pd.Timestamp(p["EIP1559_DATE"])
LAMBDA_OVH         = p["LAMBDA_OVH"]
THETA_ETC_POST     = p["THETA_ETC_POST"]
DELTA_H            = p["DELTA_H"]
H_RES              = p["H_RES"]
H_ETH_PRE          = p["H_ETH_PRE"]
e_bar_mig          = p["e_bar_mig"]
e_bar_res          = p["e_bar_res"]
P_BU               = p["P_BU"]
P_TD_50            = p["P_TD_50"]
eff_ccaf_avg       = p["eff_ccaf_avg"]
ELEC_ETH_PER_J     = p["ELEC_ETH_PER_J"]
ELEC_ETC_PER_J     = p["ELEC_ETC_PER_J"]
EFF_JASMINER_MH_J  = p["EFF_JASMINER_MH_J"]
stable_start       = pd.Timestamp(p["stable_start"])
stable_end         = pd.Timestamp(p["stable_end"])
P_TD_BE            = p["P_TD_BE"]
etc_lb_avg         = p["etc_lb_avg"]
etc_ub_avg         = p["etc_ub_avg"]
H_ETC_pre          = p["H_ETC_pre"]
# New keys (written by Section 6c of main script):
ethw_td_avg        = p.get("ethw_td_avg", np.nan)
post_lo            = p.get("post_lo",     np.nan)
post_hi            = p.get("post_hi",     np.nan)
post_mid           = (post_lo + post_hi) / 2

# Hardware efficiencies for bootstrap / theta sensitivity (written by Section 10 of main script)
hw_eff_path = PROCESSED_DIR / "hardware_efficiencies.csv"
hw_eff_arr  = pd.read_csv(hw_eff_path)["efficiency_mh_j"].values if hw_eff_path.exists() else None

print(f"    main_df         : {df.shape}, {df.index.min().date()} – {df.index.max().date()}")
print(f"    var_input       : {var_df.shape}")
print(f"    model_params    : {len(p)} scalars")

# --- Figure style (matches main script) ---
sns.set_theme(style="ticks")
plt.rcParams.update({
    "figure.figsize":    (10, 6),
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "lines.linewidth":   1.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def save_fig(fig, ax, filename):
    """Consistent style: transparent, no grid, legend below axes, sparse ticks."""
    ax.grid(False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    try:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    except Exception:
        pass
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if ax.get_legend():
            ax.get_legend().remove()
        ax.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=min(len(labels), 4),
                  framealpha=0.0, fontsize=9, borderaxespad=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"    Saved: {filename}")


def save_fig_multi(fig, filename):
    """Save a figure with multiple axes (VAR plots), no style override."""
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.set_facecolor("none")
        ax.grid(False)
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"    Saved: {filename}")




# =============================================================================
# SECTION 4: Parameter Sensitivity + Monte Carlo
# =============================================================================
print("\n[4] Parameter sensitivity + Monte Carlo ...")

# --------------------------------------------------------------------------
# 4a. Electricity Price Sensitivity (top-down + hybrid P_BU + post-Merge)
# --------------------------------------------------------------------------
# c_e affects THREE quantities simultaneously:
#   (1) Pre-Merge top-down:  MaxEnergy = ETH Revenue / c_e          [direct]
#   (2) Pre-Merge hybrid:    c_e → θ_ETC → cohort split → P_BU     [indirect]
#   (3) Post-Merge power:    ETC TD upper = ETC Revenue / c_e       [direct]
#                            ETHW TD      = ETHW Revenue / c_e      [direct]
#                            ETC LB (Jasminer BU)  = fixed          [c_e independent]
# Savings = P_BU_hybrid(c_e) − post_mid(c_e): both sides move with c_e.
# --------------------------------------------------------------------------
print("\n[4a] Sensitivity analysis: electricity price c_e ...")
print("     Chain: c_e → θ_ETC → cohort split → ē_mig, ē_res → P_BU")
print("     Also:  c_e → post-Merge ETC TD upper + ETHW TD (ETC LB fixed)")

CE_GRID_KWH = [0.03, 0.05, 0.07, 0.10, 0.15]
CE_COLORS   = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]

# Stable post-Merge window revenue averages (compute once outside loop)
rev_etc_post_avg  = df.loc[stable_start:stable_end,
                           "ETC Block Reward per Second (USD)"].mean()
# rev_ethw_post_avg = df.loc[stable_start:stable_end,
#                            "ETHPOW Block Reward per Second (USD)"].mean()

rows_ce = []
for ce_kwh in CE_GRID_KWH:
    ce_j         = ce_kwh / 3_600_000
    max_energy_s = df["ETH Total Revenue per Second (USD)"] / ce_j / 1e9
    at_merge     = max_energy_s.loc[THE_MERGE_DATE]
    at_eip1559   = max_energy_s.loc[EIP1559_DATE]
    avg_2021     = max_energy_s.loc["2021-01-01":"2021-12-31"].mean()

    # Channel 2: pre-Merge hybrid via θ(c_e) → cohort split
    theta_ce = THETA_ETC_POST * (ce_kwh / 0.10)
    if hw_eff_arr is not None:
        mig = hw_eff_arr[hw_eff_arr >= theta_ce]
        res = hw_eff_arr[hw_eff_arr <  theta_ce]
        if len(mig) > 0 and len(res) > 0:
            e_mig_ce = mig.mean()
            e_res_ce = res.mean()
            p_bu_ce  = 1e-3 * (DELTA_H / e_mig_ce + H_RES / e_res_ce) * LAMBDA_OVH
            n_mig_ce, n_res_ce = len(mig), len(res)
        else:
            p_bu_ce = e_mig_ce = e_res_ce = np.nan
            n_mig_ce = n_res_ce = 0
    else:
        theta_ce = np.nan
        p_bu_ce  = P_BU
        e_mig_ce, e_res_ce = e_bar_mig, e_bar_res
        n_mig_ce = n_res_ce = np.nan

    # Channel 3: post-Merge power at c_e
    # ETC lower bound: Jasminer bottom-up — c_e independent (fixed)
    # ETC upper bound: top-down = ETC revenue / c_e
    # ETHW:            top-down = ETHW revenue / c_e
    etc_ub_ce   = rev_etc_post_avg  / ce_j / 1e9
    # ethw_ce     = rev_ethw_post_avg / ce_j / 1e9
    # post_lo_ce  = etc_lb_avg + ethw_ce          # ETC Jasminer LB + ETHW TD
    post_lo_ce  = etc_lb_avg 
    # post_hi_ce  = etc_ub_ce  + ethw_ce          # ETC TD UB      + ETHW TD
    post_hi_ce  = etc_ub_ce       # ETC TD UB 
    post_mid_ce = (post_lo_ce + post_hi_ce) / 2

    saved_mid_ce = (p_bu_ce - post_mid_ce) if not np.isnan(p_bu_ce) else np.nan
    pct_saved_ce = (saved_mid_ce / p_bu_ce * 100) if not np.isnan(p_bu_ce) else np.nan

    rows_ce.append({
        "c_e ($/kWh)":               ce_kwh,
        "MaxEnergy at Merge (GW)":   round(at_merge,    2),
        "MaxEnergy 2021 avg (GW)":   round(avg_2021,    2),
        "MaxEnergy at EIP1559 (GW)": round(at_eip1559,  2),
        "θ_ETC_post (MH/J)":         round(theta_ce,    3),
        "n_mig":                     n_mig_ce,
        "n_res":                     n_res_ce,
        "ē_mig (MH/J)":              round(e_mig_ce, 4) if not np.isnan(e_mig_ce) else np.nan,
        "ē_res (MH/J)":              round(e_res_ce, 4) if not np.isnan(e_res_ce) else np.nan,
        "P_BU_hybrid (GW)":          round(p_bu_ce,     3) if not np.isnan(p_bu_ce)     else np.nan,
        "post_lo (GW)":              round(post_lo_ce,  3),
        "post_hi (GW)":              round(post_hi_ce,  3),
        "post_mid (GW)":             round(post_mid_ce, 3),
        "Saved mid (GW)":            round(saved_mid_ce, 3) if not np.isnan(saved_mid_ce) else np.nan,
        "Saved (%)":                 round(pct_saved_ce, 1) if not np.isnan(pct_saved_ce) else np.nan,
    })

sens_df = pd.DataFrame(rows_ce)
print(sens_df[["c_e ($/kWh)", "MaxEnergy at Merge (GW)",
               "θ_ETC_post (MH/J)", "n_mig", "P_BU_hybrid (GW)",
               "post_lo (GW)", "post_hi (GW)",
               "Saved mid (GW)", "Saved (%)"]].to_string(index=False))
sens_df.to_csv(TABLES_DIR / "tab_sensitivity_ce.csv", index=False)
print(f"    Saved: tab_sensitivity_ce.csv")

# --- Figure 4a-1: pre-Merge top-down series + hybrid P_BU band + post-Merge ---
fig, ax = plt.subplots()
for ce_kwh, color in zip(CE_GRID_KWH, CE_COLORS):
    ce_j   = ce_kwh / 3_600_000
    series = (df.loc["2019-01-01":"2022-09-14",
                     "ETH Total Revenue per Second (USD)"] / ce_j / 1e9)
    ax.plot(series.index, series.values, color=color, lw=1.4, alpha=0.85,
            label=f"MaxEnergy c_e=${ce_kwh}/kWh")
pbu_vals = sens_df["P_BU_hybrid (GW)"].dropna()
ax.axhspan(pbu_vals.min(), pbu_vals.max(), alpha=0.12, color="black",
           label=f"Hybrid P_BU range [{pbu_vals.min():.2f}–{pbu_vals.max():.2f} GW]")
ax.axhline(P_BU, color="black", linestyle="--", lw=1.4,
           label=f"Hybrid P_BU baseline ($0.10/kWh) = {P_BU:.2f} GW")
# Post-Merge band across c_e scenarios
pm_lo = sens_df["post_lo (GW)"].min()
pm_hi = sens_df["post_hi (GW)"].max()
ax.axhspan(pm_lo, pm_hi, alpha=0.15, color="mediumpurple",
           label=f"Post-Merge range (ETC+ETHW) [{pm_lo:.3f}–{pm_hi:.3f} GW]")
ax.set_title("Sensitivity: Top-Down, Hybrid P_BU, and Post-Merge Power vs c_e",
             fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Power (GW)")
save_fig(fig, ax, "ext_sensitivity_ce_series.png")

# --- Figure 4a-2: grouped bar — MaxEnergy / P_BU_hybrid / post_mid by c_e ---
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(CE_GRID_KWH))
w = 0.26
ax.bar(x - w, sens_df["MaxEnergy at Merge (GW)"], w,
       color=CE_COLORS, alpha=0.75, label="MaxEnergy at Merge (pre-Merge TD)")
ax.bar(x,     sens_df["P_BU_hybrid (GW)"],        w,
       color=CE_COLORS, alpha=0.55, edgecolor="black", linewidth=0.8,
       label="P_BU hybrid (pre-Merge, c_e-adjusted θ)")
ax.bar(x + w, sens_df["post_mid (GW)"],            w,
       color=CE_COLORS, alpha=0.30, edgecolor="black", linewidth=0.8,
       label="Post-Merge mid (ETC+ETHW, ETC LB fixed)")
# Error bars showing post lo–hi spread
err_lo = (sens_df["post_mid (GW)"] - sens_df["post_lo (GW)"]).clip(lower=0)
err_hi = (sens_df["post_hi (GW)"]  - sens_df["post_mid (GW)"]).clip(lower=0)
ax.errorbar(x + w, sens_df["post_mid (GW)"],
            yerr=[err_lo.values, err_hi.values],
            fmt="none", color="black", capsize=3, lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"${c}/kWh" for c in CE_GRID_KWH])
ax.set_xlabel("Electricity price assumption c_e")
ax.set_ylabel("Power (GW)")
ax.set_title("Pre-Merge and Post-Merge Power by Electricity Price Scenario\n"
             "(hybrid varies via θ_ETC → cohort split; post-Merge TD scales with c_e)",
             fontweight="bold")
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center",
          bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.0, fontsize=9)
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")
fig.savefig(FIGURES_DIR / "ext_sensitivity_ce_bar.png", dpi=300,
            bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"    Saved: ext_sensitivity_ce_bar.png")

# --- Figure 4a-3: transmission chain — θ(c_e), P_BU(c_e), savings(c_e) ---
ce_fine    = np.linspace(0.02, 0.18, 200)
theta_fine = THETA_ETC_POST * (ce_fine / 0.10)


pbu_fine        = []
post_mid_fine   = []
post_lo_fine    = []
post_hi_fine    = []
for ce_f in ce_fine:
    ce_j_f = ce_f / 3_600_000
    th = THETA_ETC_POST * (ce_f / 0.10)
    mig = hw_eff_arr[hw_eff_arr >= th]
    res = hw_eff_arr[hw_eff_arr <  th]
    if len(mig) > 0 and len(res) > 0:
        pbu_fine.append(1e-3 * (DELTA_H / mig.mean() + H_RES / res.mean()) * LAMBDA_OVH)
    else:
        pbu_fine.append(np.nan)
    etc_ub_f  = rev_etc_post_avg  / ce_j_f / 1e9
    post_lo_fine.append(etc_lb_avg)                          # Jasminer BU, c_e-independent
    post_hi_fine.append(etc_ub_f)                            # ETC TD UB only
    post_mid_fine.append((etc_lb_avg + etc_ub_f) / 2)

pbu_fine      = np.array(pbu_fine)
post_mid_fine = np.array(post_mid_fine)
post_lo_fine  = np.array(post_lo_fine)
post_hi_fine  = np.array(post_hi_fine)

# % savings arrays (relative to P_BU)
with np.errstate(invalid="ignore"):
    pct_mid_fine = (pbu_fine - post_mid_fine) / pbu_fine * 100
    pct_lo_fine  = (pbu_fine - post_hi_fine)  / pbu_fine * 100  # hi post → lo saving
    pct_hi_fine  = (pbu_fine - post_lo_fine)  / pbu_fine * 100  # lo post → hi saving

   
for ce_kwh, color in zip(CE_GRID_KWH, CE_COLORS):
    row = sens_df[sens_df["c_e ($/kWh)"] == ce_kwh].iloc[0]
    ce_j_sc   = ce_kwh / 3_600_000
    etc_ub_sc = rev_etc_post_avg / ce_j_sc / 1e9
    etc_mid_sc = (etc_lb_avg + etc_ub_sc) / 2   # ETC only, no ETHW
   
   
for ce_kwh, color in zip(CE_GRID_KWH, CE_COLORS):
    row    = sens_df[sens_df["c_e ($/kWh)"] == ce_kwh].iloc[0]
    ce_j_s = ce_kwh / 3_600_000
    etc_ub_s = rev_etc_post_avg / ce_j_s / 1e9
    etc_mid_s = (etc_lb_avg + etc_ub_s) / 2
    p_bu_s    = row["P_BU_hybrid (GW)"]
    pct_s     = (p_bu_s - etc_mid_s) / p_bu_s * 100 if p_bu_s > 0 else np.nan


# --- Figure: 2x2 layout ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# ======================
# Panel 1
# ======================
axes[0].plot(ce_fine, theta_fine, color="steelblue", lw=2, label="θ(c_e)")

# for ce_kwh, color in zip(CE_GRID_KWH, CE_COLORS):
#     th = THETA_ETC_POST * (ce_kwh / 0.10)
#     axes[0].axvline(ce_kwh, color=color, lw=1, linestyle=":", alpha=0.7)
#     axes[0].scatter([ce_kwh], [th], color=color, s=40)
# axes[0].axvline(ce_kwh, color=color, lw=1, linestyle=":", alpha=0.7)

axes[0].set_title("(a) Threshold θ", fontweight="bold")

axes[0].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    frameon=False,
    fontsize=9,
    ncol=2
)

# ======================
# Panel 2
# ======================
valid = ~np.isnan(pbu_fine)

axes[1].plot(ce_fine[valid], pbu_fine[valid],
             color="tomato", lw=2, label="Hybrid P_BU")

axes[1].plot(ce_fine, post_mid_fine,
             color="mediumpurple", lw=2, label="Post-Merge ETC")

axes[1].fill_between(ce_fine, post_lo_fine, post_hi_fine,
                     alpha=0.15, color="mediumpurple",
                     label="ETC range")

axes[1].fill_between(ce_fine[valid],
                     post_mid_fine[valid], pbu_fine[valid],
                     alpha=0.12, color="tomato",
                     label="Energy savings")

axes[1].set_title("(b) Power", fontweight="bold")

axes[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.30),
    frameon=False,
    fontsize=9,
    ncol=2
)

# ======================
# Panel 3
# ======================
axes[2].plot(ce_fine[valid], pct_mid_fine[valid],
             color="tomato", lw=2, label="Savings (mid)")

axes[2].fill_between(ce_fine[valid],
                     pct_lo_fine[valid], pct_hi_fine[valid],
                     alpha=0.15, color="tomato",
                     label="Savings range")

axes[2].set_title("(c) Energy Savings", fontweight="bold")

axes[2].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    frameon=False,
    fontsize=9,
    ncol=2
)

# ======================
# Panel 4（可选）
# ======================
axes[3].axis("off")


for ax in axes[:3]:
    ax.set_xlabel("Electricity price c_e ($/kWh)")
    ax.grid(False)
    ax.set_facecolor("none")

axes[0].set_ylabel("Threshold θ")
axes[1].set_ylabel("Power (GW)")
axes[2].set_ylabel("Energy savings (%)")

# ======================
# 🔥 关键：给每一行留空间
# ======================
fig.tight_layout(h_pad=3.0)  # 👈 增加上下间距（非常重要）

# 保存
fig.savefig(
    FIGURES_DIR / "fig_panel_legends_bottom.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

fig.savefig(
    FIGURES_DIR / "fig_panel_legends_bottom.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

plt.close(fig)


# --------------------------------------------------------------------------
# 1d. VAR Granger Causality (unchanged from original Section 2)
# --------------------------------------------------------------------------
print("\n[1d] VAR dynamics: IRFs and Granger causality ...")

# Rename columns for readability in plots
var_plot = var_df.rename(columns={
    "ETH / Mean Hash Rate":                  "ΔlnHashRate",
    "ETH / USD Denominated Closing Price":   "ΔlnPrice",
})

# Fit VAR (refit from saved data so this section is self-contained)

model = sm.tsa.VAR(var_plot)

lag_selection = model.select_order(maxlags=14)
print(lag_selection.summary())


results_var = sm.tsa.VAR(var_plot).fit(maxlags=10, ic="bic")
print(f"    VAR fitted: BIC={results_var.aic:.2f}, lags={results_var.k_ar}")

# IRFs (20-day horizon, orthogonalised)
irf     = results_var.irf(periods=10)
irf_fig = irf.plot(orth=True, signif=0.1, figsize=(10, 7))
irf_fig.suptitle("VAR Orthogonalised IRFs: Hash Rate ↔ Price (PoW era 2018–2021)",
                 fontsize=13, fontweight="bold", y=1.01)
save_fig_multi(irf_fig, "ext_var_irf.png")

# Forecast Error Variance Decomposition
fevd     = results_var.fevd(10)
fevd_fig = fevd.plot(figsize=(10, 5))
fevd_fig.suptitle("FEVD: Share of Hash Rate / Price Variance Explained",
                  fontsize=13, fontweight="bold", y=1.01)
save_fig_multi(fevd_fig, "ext_var_fevd.png")

# Granger causality tests
gc_hr_price = results_var.test_causality(
    "ΔlnPrice", ["ΔlnHashRate"], kind="f")
gc_price_hr = results_var.test_causality(
    "ΔlnHashRate", ["ΔlnPrice"], kind="f")

print(f"\n    Granger causality (F-tests, lags={results_var.k_ar}):")
print(f"      H0: ΔlnHashRate does NOT Granger-cause ΔlnPrice")
print(f"        F = {gc_hr_price.test_statistic:.4f},  p = {gc_hr_price.pvalue:.6f}  "
      f"{'** REJECT' if gc_hr_price.pvalue < 0.05 else '(fail to reject)'}")
print(f"      H0: ΔlnPrice does NOT Granger-cause ΔlnHashRate")
print(f"        F = {gc_price_hr.test_statistic:.4f},  p = {gc_price_hr.pvalue:.6f}  "
      f"{'** REJECT' if gc_price_hr.pvalue < 0.05 else '(fail to reject)'}")

gc_results = pd.DataFrame([
    {"Hypothesis (H0)": "ΔlnHashRate does not cause ΔlnPrice",
     "F-stat": round(gc_hr_price.test_statistic, 4),
     "p-value": round(gc_hr_price.pvalue, 6)},
    {"Hypothesis (H0)": "ΔlnPrice does not cause ΔlnHashRate",
     "F-stat": round(gc_price_hr.test_statistic, 4),
     "p-value": round(gc_price_hr.pvalue, 6)},
])
gc_results.to_csv(TABLES_DIR / "tab_granger_causality.csv", index=False)
print(f"    Saved: tab_granger_causality.csv")



