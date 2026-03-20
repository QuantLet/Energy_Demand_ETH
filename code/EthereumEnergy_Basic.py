"""
EthereumEnergy_clean.py
=======================
Clean, standalone reproduction of the Ethereum energy consumption analysis.
Derived from EthereumEnergy.ipynb (282 cells) with bugs fixed and paper
methodology aligned with 1.Pre_Applied_Energy.tex.

Paper: "The Power Demand of the Ethereum-Ecosystem"
       Woitschig, Wang, Uddin, Härdle

Data source: Coin Metrics Network Data, CoinMarketCap, CCAF hardware survey.
Run:         python code/EthereumEnergy_clean.py   (from project root)

Sections
--------
0.  Setup (imports, paths, constants)
1.  Data Loading
2.  Revenue Calculations (block reward schedule + tx fees)
3.  Top-Down Energy Estimation (paper Eq. 1: MaxEnergy upper bound)
    [Note: Digiconomist / BestGuessTopDown are original-notebook variants,
     not part of the paper's published methodology]
4.  Hardware Efficiency Analysis (CCAF inventory)
5.  Profitability Threshold (paper Eq. 6, ETH @ 0.05/kWh, ETC @ 0.10/kWh)
6.  Merge Event: ΔH (7-day pre vs. stable post), paper Section 3.3.1
6b. Hybrid Estimation (paper Section 3.3 + 5.1):
      P_ETH_pre = 1e-3 × (H_mig/ē_mig + H_res/ē_res) × λ_ovh
7.  Bottom-Up via hardware groups (supplementary; λ_ovh = 1.33 applied)
8.  Statistical Analysis (VAR)
9.  Publication-Ready Figures (paper figures + supplementary)

NOTE: Three xlsx files are missing from data/:
  - EfficienciesETCPost.xlsx  → ETC post-Merge hardware section OMITTED
  - Efficiencies_ETH_Pre.xlsx → ETH pre-Merge hardware MC section OMITTED
  - Ethereum_More_Data.xlsx   → Extended metrics section OMITTED
"""

# =============================================================================
# SECTION 0: Setup
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize

np.random.seed(42)

# --- Paths ---
# SCRIPT_DIR  = Path(__file__).parent
SCRIPT_DIR = Path("/Users/ruting/Documents/macbook/PcBack/25.The Energy Consumption of the Ethereum-Ecosystem /Energy_Demand_ETH/code")
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR    = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "Figures"
FIGURES_DIR.mkdir(exist_ok=True)

# --- Electricity price constants ---
# ETH top-down (de Vries baseline, paper Section 3.1): $0.05/kWh
ELEC_ETH_PER_J  = 0.05 / 3_600_000    # $/J
# ETC top-down upper bound (paper Section 5.2): $0.10/kWh
ELEC_ETC_PER_J  = 0.10 / 3_600_000    # $/J

# --- Block times ---
ETH_BLOCK_TIME_S = 15    # seconds per PoW block
ETC_BLOCK_TIME_S = 13    # seconds per ETC block

# --- Key dates ---
THE_MERGE_DATE      = pd.Timestamp("2022-09-14")  # last ETH PoW block
BYZANTIUM_DATE      = pd.Timestamp("2017-10-16")  # ETH reward 5→3 ETH
CONSTANTINOPLE_DATE = pd.Timestamp("2019-02-28")  # ETH reward 3→2 ETH
EIP1559_DATE        = pd.Timestamp("2021-08-05")  # ETH fee burn

# ETC hard-fork dates (ECIP-1017 schedule)
ETC_FORK1_DATE = pd.Timestamp("2017-12-11")  # 5→4 ETC
ETC_FORK2_DATE = pd.Timestamp("2020-03-17")  # 4→3.2 ETC
ETC_FORK3_DATE = pd.Timestamp("2021-04-25")  # 3.2→2.56 ETC

# --- Hybrid estimation parameters (paper Section 3.3 / 5.1) ---
LAMBDA_OVH         = 1.33   # overhead factor (McDonald 2021): PSU + cooling + grid losses
THETA_ETC_POST     = 1.0    # paper Table 1 baseline ETC post-Merge cutoff (MH/J)
                             # corresponds to c_e ≈ $0.10/kWh at post-Merge ETC economics
DELTA_H_PRE_DAYS   = 7      # days before Merge for pre-Merge average
DELTA_H_STABLE_OFFSET = 60  # skip first 60 days (initial migration spike); paper uses settled level
DELTA_H_STABLE_DAYS   = 90  # 90-day average from day 60 to day 150 post-Merge (~75 TH/s target)

# --- Individual hardware benchmarks (supplementary, not paper's cohort method) ---
EFF_JASMINER_MH_J = 2.1667  # Jasminer X4-Q ASIC (most efficient; paper Table 1)
EFF_NVIDIA_MH_J   = 0.7286  # Nvidia RTX 3090 GPU
EFF_BEST_GPU_MH_J = 0.2     # Conservative lower bound

print("=== EthereumEnergy_clean.py ===")
print(f"Project root : {PROJECT_DIR}")
print(f"Data dir     : {DATA_DIR}")
print(f"Figures dir  : {FIGURES_DIR}")

# --- Figure style ---
sns.set_theme(style="ticks")           # clean axes, no grid
plt.rcParams.update({
    "figure.figsize":   (10, 6),
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "lines.linewidth":  1.5,
    "axes.spines.top":  False,         # remove top spine
    "axes.spines.right": False,        # remove right spine
})
SOURCE_NOTE = "Source: Coin Metrics Network Data; hardware specs from CCAF mining survey"

# =============================================================================
# SECTION 1: Data Loading
# =============================================================================
print("\n[1] Loading data ...")

df = pd.read_excel(DATA_DIR / "Ethereum+.xlsx", index_col=0)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
print(f"    Ethereum+.xlsx       : {df.shape}, {df.index.min().date()} – {df.index.max().date()}")

cm = pd.read_excel(DATA_DIR / "Coin_Metrics_Network_Data_2023-03-14T12-53.xlsx", index_col=0)
cm.index = pd.to_datetime(cm.index)
cm.sort_index(inplace=True)
print(f"    Coin Metrics (latest): {cm.shape}, {cm.index.min().date()} – {cm.index.max().date()}")

ethw_raw = pd.read_excel(DATA_DIR / "ETHW_ALL_graph_coinmarketcap.xlsx")
ethw_raw["date"] = pd.to_datetime(ethw_raw["timestamp"].str[:10])
ethw_raw = ethw_raw.set_index("date")[["open"]].rename(columns={"open": "ETHW_Price_USD"})
print(f"    ETHW price           : {ethw_raw.shape}, {ethw_raw.index.min().date()} – {ethw_raw.index.max().date()}")

# CCAF hardware inventory (ASICs + GPUs combined, paper Section 3.2.3)
hw_asic = pd.read_excel(DATA_DIR / "EthereumHardware.xlsx")
hw_gpu  = pd.read_excel(DATA_DIR / "EthereumHardwareGPU.xlsx")
hw_all  = pd.concat([hw_asic, hw_gpu], ignore_index=True)
grp1    = pd.read_excel(DATA_DIR / "group1.xlsx")
grp2    = pd.read_excel(DATA_DIR / "group2.xlsx")
grp3    = pd.read_excel(DATA_DIR / "group3.xlsx")
print(f"    CCAF inventory       : ASIC={hw_asic.shape[0]}, GPU={hw_gpu.shape[0]}, "
      f"combined={hw_all.shape[0]}")
print(f"    Group survey files   : grp1={grp1.shape[0]}, grp2={grp2.shape[0]}, grp3={grp3.shape[0]}")

df = df.join(ethw_raw, how="left")
df["ETHW_Price_USD"] = df["ETHW_Price_USD"].ffill()

# =============================================================================
# SECTION 2: Revenue Calculations
# =============================================================================
print("\n[2] Computing revenue ...")

def compute_eth_block_reward(index: pd.DatetimeIndex) -> pd.Series:
    """ETH block reward per block (native units). Paper Section 3.1 / Eq. 1.

    Schedule:
      ≤ 2017-10-15 (Byzantium)      : 5 ETH
      2017-10-16 – 2019-02-27        : 3 ETH
      2019-02-28 (Constantinople) + : 2 ETH
      > 2022-09-14 (The Merge)      : 0 ETH (PoS; no PoW rewards)
    """
    r = pd.Series(5.0, index=index)
    r[index > BYZANTIUM_DATE]      = 3.0
    r[index > CONSTANTINOPLE_DATE] = 2.0
    r[index > THE_MERGE_DATE]      = 0.0
    return r


def compute_etc_block_reward(index: pd.DatetimeIndex) -> pd.Series:
    """ETC block reward per block (native units). ECIP-1017 schedule.

    Schedule:
      ≤ 2017-12-10 : 5 ETC
      2017-12-11 + : 4 ETC
      2020-03-17 + : 3.2 ETC
      2021-04-25 + : 2.56 ETC
    """
    r = pd.Series(5.0, index=index)
    r[index > ETC_FORK1_DATE] = 4.0
    r[index > ETC_FORK2_DATE] = 3.2
    r[index > ETC_FORK3_DATE] = 2.56
    return r


df["ETH / Block Reward (native units)"] = compute_eth_block_reward(df.index).values
df["ETC / Block Reward (native units)"] = compute_etc_block_reward(df.index).values

# ETH revenue per second (USD)
df["ETH / Tx Fee per Second (native units)"] = (
    df["ETH / Mean Tx Fee (native units)"] * df["ETH / Tx per Second Cnt"]
)
df["ETH / Tx Fee per Second (USD)"] = (
    df["ETH / Tx Fee per Second (native units)"]
    * df["ETH / USD Denominated Closing Price"]
)
df["ETH Block Reward per Second (USD)"] = (
    df["ETH / Block Reward (native units)"]
    * df["ETH / USD Denominated Closing Price"]
    / ETH_BLOCK_TIME_S
)
df["ETH Total Revenue per Second (USD)"] = (
    df["ETH Block Reward per Second (USD)"] + df["ETH / Tx Fee per Second (USD)"]
)

# ETH daily revenue (million USD, for figure)
SECONDS_PER_DAY = 86_400
df["ETH Block Reward per Day (native units)"] = (
    df["ETH / Block Reward (native units)"] / ETH_BLOCK_TIME_S * SECONDS_PER_DAY
)
df["ETH Block Reward per Day (USD)"]    = (
    df["ETH Block Reward per Day (native units)"]
    * df["ETH / USD Denominated Closing Price"]
)
df["ETH / Tx Fee per Day (USD)"]        = df["ETH / Tx Fee per Second (USD)"] * SECONDS_PER_DAY
df["ETH Block Reward per Day (Mio. USD)"] = df["ETH Block Reward per Day (USD)"] / 1e6
df["ETH / Tx Fee per Day (Mio. USD)"]     = df["ETH / Tx Fee per Day (USD)"] / 1e6

# ETC revenue per second (USD)
df["ETC / Tx Fee per Second (native units)"] = (
    df["ETC / Mean Tx Fee (native units)"]
    / df["ETC / Tx per Second Cnt"].replace(0, np.nan)
)
df["ETC / Tx Fee per Block (native units)"] = (
    df["ETC / Tx Fee per Second (native units)"] * ETC_BLOCK_TIME_S
)
df["ETC Block Reward per Second (USD)"] = (
    (df["ETC / Block Reward (native units)"]
     + df["ETC / Tx Fee per Block (native units)"].fillna(0))
    / ETC_BLOCK_TIME_S
    * df["ETC / USD Denominated Closing Price"]
)

# ETHPOW (fork) reward per second, using ETHW price post-Merge
df["ETHPOW Block Reward per Second (USD)"] = (
    df["ETH / Block Reward (native units)"]
    * df["ETHW_Price_USD"].fillna(0)
    / ETH_BLOCK_TIME_S
)

print("    Revenue columns computed.")

# =============================================================================
# SECTION 3: Top-Down Energy Estimation (paper Eq. 1)
# =============================================================================
print("\n[3] Top-down energy estimation ...")

# --- MaxEnergy: paper Eq. 1 upper bound ---
# 100% of miner revenue allocated to electricity cost → maximum possible energy
# P(W) = Revenue(USD/s) / ElecPrice(USD/J)
df["MaxEnergy (W)"]  = df["ETH Total Revenue per Second (USD)"] / ELEC_ETH_PER_J
df["MaxEnergy (GW)"] = df["MaxEnergy (W)"] / 1e9

# --- ETC top-down upper bound (paper Section 5.2, c_e = $0.10/kWh) ---
df["ETC MaxEnergy (W)"]  = df["ETC Block Reward per Second (USD)"] / ELEC_ETC_PER_J
df["ETC MaxEnergy (GW)"] = df["ETC MaxEnergy (W)"] / 1e9

# --- Note: the two methods below are from de Vries (2021) / original notebook,
#     not part of this paper's published methodology. Retained as context. ---

# Digiconomist (de Vries 2021): 60% of revenue to electricity
df["Digiconomist (W)"]  = df["MaxEnergy (W)"] * 0.60
df["Digiconomist (GW)"] = df["Digiconomist (W)"] / 1e9

# BestGuessTopDown (original notebook): 55% revenue, PUE=2, utilization=65%
df["BestGuessTopDown (W)"]  = (
    df["ETH Total Revenue per Second (USD)"] * 0.55
    / (ELEC_ETH_PER_J * 2 * 0.65)
)
df["BestGuessTopDown (GW)"] = df["BestGuessTopDown (W)"] / 1e9

print("    On Merge date (2022-09-14):")
for col, label in [
    ("MaxEnergy (GW)",       "MaxEnergy [paper Eq.1]"),
    ("Digiconomist (GW)",    "Digiconomist [notebook variant]"),
    ("BestGuessTopDown (GW)", "BestGuessTopDown [notebook variant]"),
]:
    val = df.loc[THE_MERGE_DATE, col] if THE_MERGE_DATE in df.index else np.nan
    print(f"      {label}: {val:.3f} GW")

# =============================================================================
# SECTION 4: Hardware Efficiency — CCAF Inventory (paper Section 3.2.3)
# =============================================================================
print("\n[4] Hardware efficiency (CCAF inventory) ...")

def device_efficiencies(hw_df: pd.DataFrame) -> pd.Series:
    """Return efficiency series e_i = Efficiency_Mh/J, dropping invalid rows."""
    e = hw_df["Efficiency_Mh/J"].replace(0, np.nan).dropna()
    return e[e > 0]


eff_all_series = device_efficiencies(hw_all)
print(f"    CCAF inventory: {len(eff_all_series)} devices with valid efficiency")
print(f"    Range: {eff_all_series.min():.4f} – {eff_all_series.max():.4f} Mh/J")
print(f"    Mean (equal weights): {eff_all_series.mean():.4f} Mh/J")
print(f"    Median: {eff_all_series.median():.4f} Mh/J")

# Devices above and below the paper's baseline ETC post-Merge cutoff
n_mig = (eff_all_series >= THETA_ETC_POST).sum()
n_res = (eff_all_series <  THETA_ETC_POST).sum()
print(f"    Devices ≥ {THETA_ETC_POST} MH/J (migrating cohort): {n_mig}")
print(f"    Devices <  {THETA_ETC_POST} MH/J (residual cohort): {n_res}")

# =============================================================================
# SECTION 5: Profitability Threshold (paper Eq. 6)
# =============================================================================
print("\n[5] Profitability threshold (paper Eq. 6) ...")

# Paper Eq. 6:  Θ_t [MH/J] = c_e / (3.6×10^12 × p_t × r_t)
# where r_t = R_t / H_t  (coin reward per hash)
# Equivalently: J/hash = Revenue_per_s / (H_TH_s × 1e12 × c_e_per_J)
#               MHash/J = 1 / (J/hash × 1e6)

def compute_profitability_threshold(
    revenue_per_s: pd.Series,
    hash_rate_th_s: pd.Series,
    elec_per_j: float,
) -> pd.DataFrame:
    """Paper Eq. 6 profitability threshold.

    Returns DataFrame with columns J/Hash, Hash/J, MHash/J.
    Rows where hash rate is zero → NaN (no mining; not profitable to enter).
    """
    j_per_hash  = revenue_per_s / (
        hash_rate_th_s.replace(0, np.nan) * 1e12 * elec_per_j
    )
    return pd.DataFrame({
        "J/Hash":  j_per_hash,
        "Hash/J":  1.0 / j_per_hash,
        "MHash/J": 1.0 / (j_per_hash * 1e6),
    })


# ETH threshold (c_e = $0.05/kWh, paper baseline)
eth_thresh = compute_profitability_threshold(
    df["ETH Total Revenue per Second (USD)"],
    df["ETH / Mean Hash Rate"],
    ELEC_ETH_PER_J,
)
eth_thresh.loc[df.index > THE_MERGE_DATE] = 0.0   # PoW mining ceases

df["ETH Profitability Threshold [J/Hash]"]  = eth_thresh["J/Hash"].values
df["ETH Profitability Threshold [MHash/J]"] = eth_thresh["MHash/J"].values

# ETC threshold (c_e = $0.10/kWh, paper Section 5.2)
etc_thresh = compute_profitability_threshold(
    df["ETC Block Reward per Second (USD)"],
    df["ETC / Mean Hash Rate"],
    ELEC_ETC_PER_J,
)
df["ETC Profitability Threshold [J/Hash]"]  = etc_thresh["J/Hash"].values
df["ETC Profitability Threshold [MHash/J]"] = etc_thresh["MHash/J"].values

# ETHPOW threshold (post-Merge fork)
ethpow_thresh = compute_profitability_threshold(
    df["ETHPOW Block Reward per Second (USD)"],
    df["ETH / Mean Hash Rate"],
    ELEC_ETH_PER_J,
)
df["ETHPOW Profitability Threshold [MHash/J]"] = ethpow_thresh["MHash/J"].values

val_eth = df.loc[THE_MERGE_DATE, "ETH Profitability Threshold [MHash/J]"]
val_etc = df.loc[THE_MERGE_DATE, "ETC Profitability Threshold [MHash/J]"]
print(f"    On Merge date — ETH threshold (0.05/kWh): {val_eth:.4f} MHash/J")
print(f"    On Merge date — ETC threshold (0.10/kWh): {val_etc:.4f} MHash/J")
print(f"    Paper Table 1 baseline cutoff:             {THETA_ETC_POST:.4f} MHash/J")

# Empirical post-Merge ETC threshold (7-day average after stable window)
etc_post_window = df.loc[
    THE_MERGE_DATE + pd.Timedelta(days=DELTA_H_STABLE_OFFSET):
    THE_MERGE_DATE + pd.Timedelta(days=DELTA_H_STABLE_OFFSET + DELTA_H_STABLE_DAYS),
    "ETC Profitability Threshold [MHash/J]",
]
theta_etc_empirical = etc_post_window.mean()
print(f"    Empirical Θ_ETC_post (stable window, 0.10/kWh): {theta_etc_empirical:.4f} MHash/J")

# =============================================================================
# SECTION 6: Merge Event — ΔH (paper Section 3.3.1)
# =============================================================================
print("\n[6] Merge event — ΔH calculation (paper Section 3.3.1) ...")

# Pre-Merge baseline: 7-day average ending on Merge date
pre_window = df.loc[
    THE_MERGE_DATE - pd.Timedelta(days=DELTA_H_PRE_DAYS):
    THE_MERGE_DATE,
    "ETC / Mean Hash Rate",
]
H_ETC_pre = pre_window.mean()

# Stable post-Merge window: skip first 7 days (spike), then 30-day average
# Paper says ETC "stabilizes around 120 TH/s" → use stable level for ΔH
stable_start = THE_MERGE_DATE + pd.Timedelta(days=DELTA_H_STABLE_OFFSET)
stable_end   = stable_start + pd.Timedelta(days=DELTA_H_STABLE_DAYS)
post_window  = df.loc[stable_start:stable_end, "ETC / Mean Hash Rate"]
H_ETC_post_stable = post_window.mean()

DELTA_H = H_ETC_post_stable - H_ETC_pre   # TH/s
H_ETH_PRE = df.loc[THE_MERGE_DATE, "ETH / Mean Hash Rate"]   # TH/s
H_RES     = H_ETH_PRE - DELTA_H            # residual (non-migrating) hash rate

print(f"    H_ETC pre-Merge   (7-day avg)    : {H_ETC_pre:.1f} TH/s")
print(f"    H_ETC post-Merge  (stable 30-day): {H_ETC_post_stable:.1f} TH/s")
print(f"    ΔH  (migration increment)        : {DELTA_H:.1f} TH/s  (paper reports ~75 TH/s)")
print(f"    H_ETH_pre (last PoW day)          : {H_ETH_PRE:.1f} TH/s")
print(f"    H_res = H_ETH_pre − ΔH           : {H_RES:.1f} TH/s")

MERGE_WINDOW_DAYS = 30
df_merge = df.loc[
    THE_MERGE_DATE - pd.Timedelta(days=MERGE_WINDOW_DAYS):
    THE_MERGE_DATE + pd.Timedelta(days=MERGE_WINDOW_DAYS)
].copy()

# =============================================================================
# SECTION 6b: Hybrid Estimation (paper Sections 3.3 + 5.1)
# =============================================================================
print("\n[6b] Hybrid estimation (paper Sections 3.3 + 5.1) ...")

def compute_cohort_efficiencies(
    hw_df: pd.DataFrame,
    theta_cutoff: float,
) -> dict:
    """Compute cohort-average efficiencies using equal weights (paper Section 3.2.3).

    Splits CCAF hardware inventory at theta_cutoff (MH/J):
      - Migrating cohort : e_i >= theta_cutoff  (profitable on post-Merge ETC)
      - Residual cohort  : e_i <  theta_cutoff  (not profitable on post-Merge ETC)

    Returns dict with keys: e_bar_mig, e_bar_res, n_mig, n_res.
    """
    e = device_efficiencies(hw_df)
    mig = e[e >= theta_cutoff]
    res = e[e <  theta_cutoff]
    return {
        "e_bar_mig": mig.mean() if len(mig) > 0 else np.nan,
        "e_bar_res": res.mean() if len(res) > 0 else np.nan,
        "n_mig":     len(mig),
        "n_res":     len(res),
        "mig_series": mig,
        "res_series": res,
    }


def hybrid_power_gw(
    h_mig: float,
    h_res: float,
    e_bar_mig: float,
    e_bar_res: float,
    lambda_ovh: float,
) -> float:
    """Paper hybrid formula: P [GW] = 1e-3 × (H_mig/ē_mig + H_res/ē_res) × λ_ovh.

    H in TH/s, e in MH/J → TH/s / MH/J = 10^6 J/s = MW → ×10^-3 = GW.
    """
    return 1e-3 * (h_mig / e_bar_mig + h_res / e_bar_res) * lambda_ovh


# ---- Paper Table scenarios ----
# Scenario A: Top-down break-even (Section 5.1.1)
#   ē_mig = Θ_ETC_post (min efficiency to mine ETC profitably = 1 MH/J)
#   ē_res = Θ_ETH_pre  (min efficiency to mine ETH pre-Merge ≈ 0.1 MH/J)
#   λ_ovh NOT applied to top-down (overheads already implicit in revenue-based calc)
e_mig_td_be = THETA_ETC_POST                                   # 1.0 MH/J
e_res_td_be = val_eth if val_eth > 0 else 0.1                  # ~0.046 MH/J from data
P_TD_BE = 1e-3 * (DELTA_H / e_mig_td_be + H_RES / e_res_td_be)  # no λ_ovh for TD

# Scenario B: Top-down 50% rule (Section 5.1.1, de Vries f≈0.5)
#   ē_mig = 2 × Θ_ETC_post, ē_res = 2 × Θ_ETH_pre  (miners use only ~half revenue for electricity)
e_mig_td_50 = 2 * THETA_ETC_POST
e_res_td_50 = 2 * e_res_td_be
P_TD_50 = 1e-3 * (DELTA_H / e_mig_td_50 + H_RES / e_res_td_50)  # no λ_ovh for TD

# Scenario C: Bottom-up CCAF (Section 5.1.2)
#   ē_mig, ē_res = conditional means from hardware inventory at THETA_ETC_POST
cohorts = compute_cohort_efficiencies(hw_all, THETA_ETC_POST)
e_bar_mig = cohorts["e_bar_mig"]
e_bar_res = cohorts["e_bar_res"]
P_BU = hybrid_power_gw(DELTA_H, H_RES, e_bar_mig, e_bar_res, LAMBDA_OVH)

# Sensitivity: use empirical theta (computed from data)
cohorts_emp = compute_cohort_efficiencies(hw_all, theta_etc_empirical)
e_bar_mig_emp = cohorts_emp["e_bar_mig"]
e_bar_res_emp = cohorts_emp["e_bar_res"]
P_BU_emp = hybrid_power_gw(DELTA_H, H_RES, e_bar_mig_emp, e_bar_res_emp, LAMBDA_OVH)

print(f"\n    CCAF cohort split at θ = {THETA_ETC_POST} MH/J (paper baseline):")
print(f"      ē_mig = {e_bar_mig:.4f} MH/J  ({cohorts['n_mig']} devices)")
print(f"      ē_res = {e_bar_res:.4f} MH/J  ({cohorts['n_res']} devices)")
print(f"      Paper Table 1 reports: ē_mig ≈ 1.62 MH/J, ē_res ≈ 0.35 MH/J")
print()
print("    ┌────────────────────────────────────────────────────────────────────┐")
print("    │  Pre-Merge ETH Power Demand — Hybrid Estimates                    │")
print(f"    │  ΔH = {DELTA_H:.1f} TH/s,  H_res = {H_RES:.1f} TH/s                          │")
print("    ├──────────────────────────────────────┬───────────────────────────┤")
print("    │  Scenario                            │  P_ETH_pre (GW)           │")
print("    ├──────────────────────────────────────┼───────────────────────────┤")
print(f"    │  TD break-even (ē_mig=1.0, ē_res=Θ_ETH) │  {P_TD_BE:>6.2f} GW              │")
print(f"    │  TD 50% rule   (ē_mig=2.0, ē_res=2Θ_ETH)│  {P_TD_50:>6.2f} GW              │")
print(f"    │  BU CCAF       (ē_mig={e_bar_mig:.2f}, ē_res={e_bar_res:.2f}) │  {P_BU:>6.2f} GW (paper: 2.96) │")
print(f"    │  BU CCAF (empirical θ)               │  {P_BU_emp:>6.2f} GW              │")
print("    └──────────────────────────────────────┴───────────────────────────┘")
print()
print(f"    Paper reports range: 2.96 – 3.83 GW  (Table 1 + Eq. after 6)")

# ---- ETC post-Merge power bounds (paper Section 5.2) ----
# Lower bound: bottom-up with most efficient device (Jasminer 2.167 MH/J), no overhead
# Upper bound: top-down at c_e = $0.10/kWh

etc_post_df = df.loc[THE_MERGE_DATE:].copy()
etc_post_df["ETC Lower Bound (GW)"] = (
    etc_post_df["ETC / Mean Hash Rate"] * 1e6 / EFF_JASMINER_MH_J / 1e9
)
# ETC upper bound = ETC MaxEnergy already computed with 0.10/kWh in Section 3

etc_lb_avg = etc_post_df.loc[stable_start:stable_end, "ETC Lower Bound (GW)"].mean()
etc_ub_avg = etc_post_df.loc[stable_start:stable_end, "ETC MaxEnergy (GW)"].mean()
print(f"\n    ETC post-Merge (stable window average):")
print(f"      Lower bound (Jasminer 2.167 MH/J): {etc_lb_avg:.3f} GW  (paper: ~0.065 GW)")
print(f"      Upper bound (top-down 0.10/kWh)  : {etc_ub_avg:.3f} GW  (paper: ~0.125 GW)")


# =============================================================================
# SECTION 7: Bottom-Up via Hardware Groups (supplementary)
# =============================================================================
print("\n[7] Bottom-up via hardware groups (supplementary, λ_ovh applied) ...")

GROUP_HASH_RATES_TH_S = {"group1": 200, "group2": 540, "group3": 85}
GROUP_DATA = {"group1": grp1, "group2": grp2, "group3": grp3}


def estimate_hardware_counts(
    hw_df: pd.DataFrame,
    target_hash_th_s: float,
    seed: int = 42,
) -> tuple:
    """Estimate unit counts per hardware type to match observed total hash rate.

    Minimises distance from uniform prior subject to hash-rate constraint.
    Returns (n_opt, total_energy_W_with_overhead, weighted_eff_MhJ).
    """
    np.random.seed(seed)
    h   = hw_df["Hashing power (Mh/s)"].values
    eta = hw_df["Efficiency_Mh/J"].replace(0, np.nan).values
    valid = ~np.isnan(eta) & (h > 0)
    h_v, eta_v = h[valid], eta[valid]
    if len(h_v) == 0:
        return np.array([]), 0.0, 0.0

    target_mh_s = target_hash_th_s * 1e6
    prior = np.ones(len(h_v))

    def objective(n):
        return (np.sum((n - prior) ** 2)
                + 1e-4 * (np.dot(n, h_v) - target_mh_s) ** 2)

    res = minimize(
        objective, x0=prior, method="SLSQP",
        bounds=[(0, None)] * len(h_v),
        constraints=[{"type": "eq", "fun": lambda n: np.dot(n, h_v) - target_mh_s}],
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    n_opt    = np.maximum(res.x, 0)
    energy_w = np.sum(n_opt * h_v / eta_v) * LAMBDA_OVH   # apply overhead factor
    total_h  = np.dot(n_opt, h_v)
    wt_eff   = total_h / (energy_w / LAMBDA_OVH) if energy_w > 0 else 0.0
    return n_opt, energy_w, wt_eff


total_energy_gw = 0.0
for grp_name, grp_df in GROUP_DATA.items():
    target = GROUP_HASH_RATES_TH_S[grp_name]
    _, energy_w, wt_eff = estimate_hardware_counts(grp_df, target)
    total_energy_gw += energy_w / 1e9
    print(f"    {grp_name}: target={target} TH/s, energy (×λ_ovh)={energy_w/1e9:.3f} GW, "
          f"eff={wt_eff:.4f} Mh/J")

print(f"    Total (all groups, λ_ovh={LAMBDA_OVH}): {total_energy_gw:.3f} GW")

# =============================================================================
# SECTION 8: Statistical Analysis (VAR)
# =============================================================================
print("\n[8] Statistical analysis (VAR model) ...")

try:
    import statsmodels.api as sm
    var_df = cm[[
        "ETH / Mean Hash Rate",
        "ETH / USD Denominated Closing Price",
    ]].copy()
    for col in var_df.columns:
        var_df[col] = np.log(var_df[col].replace(0, np.nan)).diff()
    var_df = var_df.loc["2018-01-01":"2021-10-31"].dropna()
    results_var = sm.tsa.VAR(var_df).fit(14, ic="aic")
    print(f"    VAR fitted: AIC={results_var.aic:.2f}, lags={results_var.k_ar}")
except Exception as e:
    print(f"    VAR fitting failed: {e}")
    results_var = None


# =============================================================================
# SECTION 9: Figures
# =============================================================================
print("\n[9] Generating figures ...")


# def save_fig(fig, ax, filename):
#     """Apply consistent style and save figure.

#     Style rules:
#       - Transparent background (figure + axes)
#       - No grid, no source note
#       - Keep x/y axes (spines + ticks); sparse tick marks
#       - Legend centred below the axes (horizontal)
#     """
#     ax.grid(False)
#     ax.set_facecolor("none")
#     fig.patch.set_alpha(0.0)

#     # Sparse, readable date ticks on x-axis
#     try:
#         locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
#         ax.xaxis.set_major_locator(locator)
#         ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
#     except Exception:
#         pass
#     ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

#     # Legend centred below the axes
#     handles, labels = ax.get_legend_handles_labels()
#     if handles:
#         if ax.get_legend():
#             ax.get_legend().remove()
#         ncol = min(len(labels), 4)
#         ax.legend(
#             handles, labels,
#             loc="upper center",
#             bbox_to_anchor=(0.5, -0.18),
#             ncol=ncol,
#             framealpha=0.0,
#             fontsize=9,
#             borderaxespad=0,
#         )

#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.22)
#     fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight",
#                 transparent=True)
#     plt.close(fig)
#     print(f"    Saved: {filename}")


def save_fig(fig, ax, filename):
    """Apply consistent publication-style formatting and save figure."""

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator

    # ---- Clean style ----
    ax.grid(False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    # ---- Remove minor ticks (解决你之前“两层刻度线”问题) ----
    ax.minorticks_off()
    
    # ax.xaxis.set_minor_locator(mdates.NullLocator())

    # ---- X-axis: clean date ticks ----
    try:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    except Exception:
        pass

    # ---- Y-axis: clean numeric ticks ----
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # ---- Legend below plot (robust version) ----
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if ax.get_legend():
            ax.get_legend().remove()

        ncol = min(len(labels), 4)

        legend = fig.legend(   # ❗改成 fig.legend（更稳定，不会被裁）
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            frameon=False,
            fontsize=9
        )

    # ---- Layout (关键顺序) ----
    fig.tight_layout()

    # 给 legend 留空间（比你原来的更稳）
    fig.subplots_adjust(bottom=0.20)

    # ---- Save (PNG + PDF 双版本) ----
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    # 可选：自动保存 PDF（强烈推荐论文）
    fig.savefig(
        FIGURES_DIR / filename.replace(".png", ".pdf"),
        bbox_inches="tight",
        transparent=True
    )

    # ---- Close ----
    plt.close(fig)

    print(f"Saved: {filename}")
    
# def add_merge_line(ax, label: bool = True) -> None:
#     ax.axvline(THE_MERGE_DATE, color="green", linestyle="--", linewidth=1.2,
#                label="The Merge (Sep 2022)" if label else "_nolegend_")

def add_merge_line(ax, label: bool = True) -> None:
    ax.axvline(THE_MERGE_DATE, color="green", linestyle="--", linewidth=1.2)

# ------------------------------------------------------------------
# Figure 1: ETH and ETC Price (USD)  → PriceETHETC.png
# ------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

data = df.loc["2016-01-01":"2023-02-28"]

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(8,6),
    gridspec_kw={"hspace":0.05}
)

# ---- ETH ----
ax1.plot(
    data.index,
    data["ETH / USD Denominated Closing Price"],
    color="tab:blue",
    alpha=0.9
)

add_merge_line(ax1)   # 画竖线但不加legend

ax1.set_ylabel("ETH Price (USD)")
ax1.set_title("ETH and ETC Price (USD)", fontweight="bold")

# ---- ETC ----
ax2.plot(
    data.index,
    data["ETC / USD Denominated Closing Price"],
    color="tab:orange",
    alpha=0.9
)

add_merge_line(ax2)

ax2.set_ylabel("ETC Price (USD)")
ax2.set_xlabel("Year")

# ---- x轴范围 ----
ax2.set_xlim(data.index[0], pd.Timestamp("2023-02-28"))

# ---- 年度刻度 ----
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# 上图不显示x刻度
ax1.tick_params(labelbottom=False)


fig.tight_layout()

save_fig(fig, ax1, "PriceETHETC.png")



# ------------------------------------------------------------------
# Figure 2: ETH and ETC Hash Rate   → HashrateETHETC.png
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

data = df.loc["2016-01-01":"2023-02-28"]

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(8,6),
    gridspec_kw={"hspace":0.05}
)

# ---- ETH Hash Rate ----
ax1.plot(
    data.index,
    data["ETH / Mean Hash Rate"],
    color="tab:blue",
    alpha=0.9
)

add_merge_line(ax1)

ax1.set_ylabel("ETH Hash Rate (TH/s)")
ax1.set_title("ETH and ETC Hash Rate (TH/s)", fontweight="bold")

# ---- ETC Hash Rate ----
ax2.plot(
    data.index,
    data["ETC / Mean Hash Rate"],
    color="tab:orange",
    alpha=0.9
)

add_merge_line(ax2)

ax2.set_ylabel("ETC Hash Rate (TH/s)")
ax2.set_xlabel("Year")

# ---- x轴范围 ----
ax2.set_xlim(data.index[0], pd.Timestamp("2023-02-28"))

# ---- 年度刻度 ----
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# 上图不显示x刻度
ax1.tick_params(labelbottom=False)

fig.tight_layout()

save_fig(fig, ax1, "HashrateETHETC.png")

# # ------------------------------------------------------------------
# # Figure 3: Hash rates around Merge (zoom)  → HashrateFour.png
# # ------------------------------------------------------------------
# fig, ax = plt.subplots()
# df.loc["2022-07-01":"2023-02-28",
#        ["ETH / Mean Hash Rate", "ETC / Mean Hash Rate"]].plot(ax=ax, alpha=0.85)
# add_merge_line(ax)
# ax.set_title("Hash Rates Around The Merge: ETH and ETC", fontweight="bold")
# ax.set_xlabel("Date")
# ax.set_ylabel("Hash Rate (TH/s)")
# ax.legend(["ETH (Ethash)", "ETC (Ethash)", "The Merge"])
# ax.text(0.01, 0.97,
#         "Note: Vertcoin / Bitcoin Gold (non-Ethash placebo series)\n"
#         "not available in dataset — see paper Figure 3.",
#         transform=ax.transAxes, fontsize=8, va="top", color="gray")
# save_fig(fig, ax, "HashrateFour.png")

# ------------------------------------------------------------------
# Figure 7: Hash Rate Migration ±30 days  → fig_hashrate_migration.png
# ------------------------------------------------------------------
fig, ax = plt.subplots()
df_merge[["ETH / Mean Hash Rate", "ETC / Mean Hash Rate"]].plot(ax=ax, alpha=0.85)
add_merge_line(ax)
# ax.annotate(
#     f"ΔH ≈ {DELTA_H:.0f} TH/s",
#     xy=(THE_MERGE_DATE + pd.Timedelta(days=7), H_ETC_pre + DELTA_H * 0.8),
#     xytext=(THE_MERGE_DATE + pd.Timedelta(days=14), H_ETC_pre + DELTA_H * 1.1),
#     fontsize=9,
#     arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
# )
ax.set_title("Hash Rate Migration Around The Merge (±30 days)", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Hash Rate (TH/s)")
ax.legend(["ETH hash rate", "ETC hash rate", "The Merge"])
save_fig(fig, ax, "fig_hashrate_migration.png")


# ------------------------------------------------------------------
# Figure 4: Profitability Thresholds ETH vs ETC  → fig_profitability_threshold.png
# ETH: c_e = $0.05/kWh ; ETC: c_e = $0.10/kWh (paper Eq. 6)
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

data = df.loc["2021-01-01":"2022-12-31"]

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(8,6),
    gridspec_kw={"hspace":0.05}
)

# ---- ETH Profitability Threshold ----
ax1.plot(
    data.index,
    data["ETH Profitability Threshold [MHash/J]"],
    color="tab:blue",
    alpha=0.9
)

add_merge_line(ax1)

# ax1.axhline(
#     0.1,
#     color="gray",
#     linestyle=":",
#     linewidth=1.0
# )

ax1.set_ylabel("ETH efficiency")
ax1.set_title("Miner Profitability Thresholds(MHash/J) — ETH vs ETC", fontweight="bold")

# ---- ETC Profitability Threshold ----
ax2.plot(
    data.index,
    data["ETC Profitability Threshold [MHash/J]"],
    color="tab:orange",
    alpha=0.9
)

add_merge_line(ax2)

# ETC cutoff line
ax2.axhline(
    THETA_ETC_POST,
    color="gray",
    linestyle=":",
    linewidth=1.0
)


ax2.set_ylabel("ETC efficiency")
ax2.set_xlabel("Year")

# ---- x轴范围 ----
ax2.set_xlim(data.index[0], pd.Timestamp("2022-12-31"))

# ---- 年度刻度 ----
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# 上图不显示x刻度
ax1.tick_params(labelbottom=False)

fig.tight_layout()

save_fig(fig, ax1, "fig_profitability_threshold.png")

# ------------------------------------------------------------------
# Figure 5: ETC Post-Merge Energy Bounds  → ETCEnergy.png
# ------------------------------------------------------------------

# import matplotlib.dates as mdates

fig, ax = plt.subplots()

etc_plot = etc_post_df.loc["2022-09-14":"2023-02-28",
                           ["ETC MaxEnergy (GW)", "ETC Lower Bound (GW)"]].copy()

etc_plot.columns = [
    "Upper bound — top-down (c_e=$0.10/kWh)",
    f"Lower bound — bottom-up (Jasminer {EFF_JASMINER_MH_J:.2f} MH/J)",
]

etc_plot.plot(ax=ax, alpha=0.85)

ax.fill_between(
    etc_plot.index,
    etc_plot.iloc[:, 1],
    etc_plot.iloc[:, 0],
    alpha=0.12,
    label="Estimated range"
)

# ---- Date formatting ----
ax.set_title("ETC Post-Merge Energy Consumption Bounds", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Power (GW)")

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),  # 往下移动
    ncol=3,                       # 横向排开（你有3个元素）
    frameon=False
)
# plt.show()

# plt.close()

save_fig(fig, ax, "ETCEnergy.png")


# ------------------------------------------------------------------
# Figure 9: ETHPOW Profitability Threshold (post-Merge fork)  → fig_ethpow_threshold.png
# ------------------------------------------------------------------
# fig, ax = plt.subplots()
# ethpow_post = df.loc["2022-09-15":"2023-02-28",
#                      "ETHPOW Profitability Threshold [MHash/J]"].copy()
# ethpow_post = ethpow_post.replace(0, np.nan)
# ax.plot(ethpow_post.index, ethpow_post.values,
#         color="purple", lw=1.5, alpha=0.85,
#         label="ETHPOW threshold (PoW fork)")
# ax.axhline(THETA_ETC_POST, color="gray", linestyle=":", lw=1.0,
#            label=f"θ_ETC_post = {THETA_ETC_POST} MH/J")
# ax.set_title("ETHPOW Profitability Threshold (Post-Merge Fork)", fontweight="bold")
# ax.set_xlabel("Date")
# ax.set_ylabel("Min. required efficiency (MHash/J)")
# save_fig(fig, ax, "fig_ethpow_threshold.png")

# ------------------------------------------------------------------
# Figure 10: Energy Comparison — Top-Down vs Bottom-Up vs Hybrid
#            → fig_energy_comparison.png
# Bottom-up series: CCAF energy-weighted average efficiency × hash rate × λ_ovh
# ------------------------------------------------------------------
def energy_weighted_avg_eff(hw_df):
    """Energy-weighted average efficiency: Σ H_i / Σ (H_i / η_i)."""
    h   = hw_df["Hashing power (Mh/s)"].values
    eta = hw_df["Efficiency_Mh/J"].replace(0, np.nan).values
    ok  = ~np.isnan(eta) & (h > 0)
    return np.sum(h[ok]) / np.sum(h[ok] / eta[ok])


eff_ccaf_avg = energy_weighted_avg_eff(hw_all)   # MH/J, energy-weighted

# Bottom-up series: H_ETH (TH/s) × 1e6 (MH/TH) / eff (MH/J) / 1e9 → GW × λ_ovh
df["BU CCAF Avg (GW)"] = (
    df["ETH / Mean Hash Rate"] * 1e6 / eff_ccaf_avg / 1e9 * LAMBDA_OVH
)
df.loc[df.index > THE_MERGE_DATE, "BU CCAF Avg (GW)"] = 0.0

plot_start = "2017-01-01"
plot_end   = "2022-09-30"
plot_idx   = df.loc[plot_start:plot_end].index

ref_date  = pd.Timestamp("2021-06-01")
td_at_ref = df.loc[ref_date, "MaxEnergy (GW)"]
bu_at_ref = df.loc[ref_date, "BU CCAF Avg (GW)"]

# ------------------------------------------------------------------
# Figure 11: Bottom-Up Energy (CCAF) over time  → fig_energy_bottomup.png
# ------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(df.loc[plot_start:plot_end].index,
        df.loc[plot_start:plot_end, "BU CCAF Avg (GW)"],
        color="steelblue", lw=1.6, alpha=0.85,
        label=f"Bottom-up (CCAF eff={eff_ccaf_avg:.3f} MH/J, ×λ_ovh={LAMBDA_OVH})")
add_merge_line(ax)
ax.set_title("ETH Bottom-Up Energy Estimate (CCAF Hardware Inventory)",
             fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Power (GW)")
save_fig(fig, ax, "fig_energy_bottomup.png")

# ------------------------------------------------------------------
# Figure 12: Top-Down vs Bottom-Up (direct comparison)
#            → fig_energy_td_vs_bu.png
# ------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(df.loc[plot_start:plot_end].index,
        df.loc[plot_start:plot_end, "MaxEnergy (GW)"],
        color="tomato", lw=1.6, alpha=0.85,
        label="Top-down upper bound (100% rev., $0.05/kWh)")
ax.plot(df.loc[plot_start:plot_end].index,
        df.loc[plot_start:plot_end, "BU CCAF Avg (GW)"],
        color="steelblue", lw=1.6, alpha=0.85,
        label=f"Bottom-up (CCAF eff={eff_ccaf_avg:.3f} MH/J, ×λ_ovh={LAMBDA_OVH})")
add_merge_line(ax)
ax.set_title("ETH Energy: Top-Down vs Bottom-Up", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Power (GW)")
save_fig(fig, ax, "fig_energy_td_vs_bu.png")



# ------------------------------------------------------------------
# Supplementary: CCAF cohort efficiency histogram
# ------------------------------------------------------------------
# fig, ax = plt.subplots()
# bins = np.linspace(0, eff_all_series.max() + 0.1, 40)
# ax.hist(cohorts["res_series"], bins=bins, alpha=0.6,
#         label=f"Residual (e<{THETA_ETC_POST}, n={cohorts['n_res']}, ē={e_bar_res:.3f})")
# ax.hist(cohorts["mig_series"], bins=bins, alpha=0.6,
#         label=f"Migrating (e≥{THETA_ETC_POST}, n={cohorts['n_mig']}, ē={e_bar_mig:.3f})")
# ax.axvline(THETA_ETC_POST, color="black",    linestyle="--",
#            label=f"θ_ETC = {THETA_ETC_POST} MH/J")
# ax.axvline(e_bar_mig,      color="darkorange", linestyle=":",
#            label=f"ē_mig = {e_bar_mig:.3f} MH/J")
# ax.axvline(e_bar_res,      color="steelblue", linestyle=":",
#            label=f"ē_res = {e_bar_res:.3f} MH/J")
# ax.set_title("CCAF Hardware Inventory — Cohort Split", fontweight="bold")
# ax.set_xlabel("Device Efficiency (MH/J)")
# ax.set_ylabel("Device Count")
# save_fig(fig, ax, "fig_cohort_split.png")

fig, ax = plt.subplots(figsize=(8, 6.5)) 

bins = np.linspace(0, float(eff_all_series.max()) + 0.1, 40)

# Residual cohort
ax.hist(
    cohorts["res_series"].values,
    bins=bins,
    alpha=0.6,
    color="steelblue",
    label=f"Residual (e < {THETA_ETC_POST}, n={cohorts['n_res']}, ē={e_bar_res:.3f})"
)

# Migrating cohort
ax.hist(
    cohorts["mig_series"].values,
    bins=bins,
    alpha=0.6,
    color="darkorange",
    label=f"Migrating (e ≥ {THETA_ETC_POST}, n={cohorts['n_mig']}, ē={e_bar_mig:.3f})"
)

# Threshold line
ax.axvline(
    THETA_ETC_POST,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"θ_ETC = {THETA_ETC_POST} MH/J"
)

# Mean efficiency lines
ax.axvline(
    e_bar_mig,
    color="darkorange",
    linestyle=":",
    linewidth=1.5,
    label=f"ē_mig = {e_bar_mig:.3f} MH/J"
)

ax.axvline(
    e_bar_res,
    color="steelblue",
    linestyle=":",
    linewidth=1.5,
    label=f"ē_res = {e_bar_res:.3f} MH/J"
)

# Titles and labels
ax.set_title("CCAF Hardware Inventory — Cohort Split", fontweight="bold")
ax.set_xlabel("Device Efficiency (MH/J)")
ax.set_ylabel("Device Count")

# Axis formatting
_x_max = float(eff_all_series.max())
ax.set_xlim(0.0, _x_max * 1.05)

_ticks = np.arange(0.0, _x_max + 0.5, 0.5)
ax.set_xticks(_ticks)
ax.set_xticklabels([f"{t:.1f}" for t in _ticks])

# -------------------------------
# Legend (correct handling)
# -------------------------------

# 1. Collect handles & labels
handles, labels = ax.get_legend_handles_labels()

# 2. Remove axis legend (avoid duplication)
if ax.get_legend() is not None:
    ax.get_legend().remove()

# 3. Create figure-level legend at bottom
ncol = min(len(labels), 4)

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=ncol,
    frameon=False,
    fontsize=9
)

# -------------------------------
# Layout (IMPORTANT ORDER)
# -------------------------------
fig.subplots_adjust(bottom=0.50)  # 或 0.32 / 0.35

fig.tight_layout(rect=[0, 0.1, 1, 1])  # 👈 预留底部22%

  # ---- Save (PNG + PDF 双版本) ----
fig.savefig(
    FIGURES_DIR / "fig_cohort_split.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)


fig.savefig(
    FIGURES_DIR / "fig_cohort_split.png".replace(".png", ".pdf"),
    bbox_inches="tight",
    transparent=True
)

# ---- Close ----
plt.close(fig)

# =============================================================================
# SECTION 10: Export processed data for ExtendedAnalysis.py
# =============================================================================
print("\n[10] Exporting processed data ...")
import json

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Main DataFrame — all computed columns
df.to_parquet(PROCESSED_DIR / "main_df.parquet")
print(f"    main_df.parquet          : {df.shape}")

# VAR input — log-differenced hash rate + price (2018–2021)
var_export = cm[[
    "ETH / Mean Hash Rate",
    "ETH / USD Denominated Closing Price",
]].copy()
for col in var_export.columns:
    var_export[col] = np.log(var_export[col].replace(0, np.nan)).diff()
var_export = var_export.loc["2018-01-01":"2021-10-31"].dropna()
var_export.to_parquet(PROCESSED_DIR / "var_input.parquet")
print(f"    var_input.parquet        : {var_export.shape}")

# Scalar parameters
params = {
    # Merge event
    "DELTA_H":          float(DELTA_H),
    "H_ETH_PRE":        float(H_ETH_PRE),
    "H_RES":            float(H_RES),
    "H_ETC_pre":        float(H_ETC_pre),
    # Hybrid estimates
    "P_BU":             float(P_BU),
    "P_TD_50":          float(P_TD_50),
    "P_TD_BE":          float(P_TD_BE),
    # Hardware efficiency
    "e_bar_mig":        float(e_bar_mig),
    "e_bar_res":        float(e_bar_res),
    "eff_ccaf_avg":     float(eff_ccaf_avg),
    "n_mig":            int(cohorts["n_mig"]),
    "n_res":            int(cohorts["n_res"]),
    # ETC post-Merge bounds
    "etc_lb_avg":       float(etc_lb_avg),
    "etc_ub_avg":       float(etc_ub_avg),
    # Constants
    "ELEC_ETH_PER_J":   float(ELEC_ETH_PER_J),
    "ELEC_ETC_PER_J":   float(ELEC_ETC_PER_J),
    "LAMBDA_OVH":       float(LAMBDA_OVH),
    "THETA_ETC_POST":   float(THETA_ETC_POST),
    "EFF_JASMINER_MH_J": float(EFF_JASMINER_MH_J),
    "ETH_BLOCK_TIME_S": int(ETH_BLOCK_TIME_S),
    "ETC_BLOCK_TIME_S": int(ETC_BLOCK_TIME_S),
    # Key dates
    "THE_MERGE_DATE":       str(THE_MERGE_DATE.date()),
    "EIP1559_DATE":         str(EIP1559_DATE.date()),
    "BYZANTIUM_DATE":       str(BYZANTIUM_DATE.date()),
    "CONSTANTINOPLE_DATE":  str(CONSTANTINOPLE_DATE.date()),
    "stable_start":         str(stable_start.date()),
    "stable_end":           str(stable_end.date()),
}
with open(PROCESSED_DIR / "model_params.json", "w") as fh:
    json.dump(params, fh, indent=2)
print(f"    model_params.json        : {len(params)} parameters")

hw_eff_series = device_efficiencies(hw_all)
hw_eff_series.to_frame(name="efficiency_mh_j").to_csv(
    PROCESSED_DIR / "hardware_efficiencies.csv", index=False
)
print(f"    hardware_efficiencies.csv: {len(hw_eff_series)} devices")

# =============================================================================
# Summary
# =============================================================================
figures_created = list(FIGURES_DIR.glob("*.png"))
print(f"\n=== Done. {len(figures_created)} figures saved to {FIGURES_DIR} ===")
for f in sorted(figures_created):
    print(f"    {f.name}")

print("\n=== Key Results Summary ===")
print(f"  ΔH (migration increment)       : {DELTA_H:.1f} TH/s  (paper: ~75 TH/s)")
print(f"  ē_mig (CCAF, θ=1.0 MH/J)      : {e_bar_mig:.4f} MH/J  (paper: 1.62 MH/J)")
print(f"  ē_res (CCAF, θ=1.0 MH/J)      : {e_bar_res:.4f} MH/J  (paper: 0.35 MH/J)")
print(f"  P_ETH_pre BU                    : {P_BU:.2f} GW     (paper: 2.96 GW)")
print(f"  P_ETH_pre TD 50% rule           : {P_TD_50:.2f} GW     (paper: 3.83 GW)")
print(f"  CCAF energy-weighted avg eff    : {eff_ccaf_avg:.4f} MH/J")
print(f"  ETC post-Merge lower bound      : {etc_lb_avg:.3f} GW   (paper: ~0.065 GW)")
print(f"  ETC post-Merge upper bound      : {etc_ub_avg:.3f} GW   (paper: ~0.125 GW)")
