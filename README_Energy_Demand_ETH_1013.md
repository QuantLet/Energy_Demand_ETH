# Energy_Demand_ETH_1013: Manual Execution of Individual Files

This directory has no unified execution entry point. Each result script loads its required inputs directly and generates its own outputs, without requiring other result scripts to run first or invoking `run_all.py`. Shared modules perform only necessary computations and do not export figures or tables.

## Initial Setup

```sh
cd /Users/ruting/Documents/macbook/PcBack/25_CryptoC_Energy/Code_Upload/Energy_Demand_ETH_1013
python -m pip install -r requirements.txt
```

If `python` is unavailable in the terminal, try `python3` or use the installed interpreter `/Users/ruting/opt/anaconda3/bin/python`.

After setup, run any script independently. You can also open a script in an editor and select **Run Python File**. All paths are resolved relative to file locations, so scripts can be executed from any working directory using absolute paths.

## Main Manuscript Figures: Run Individually

```sh
python code/code_manuscript/figure_02_hashrate.py
```
Output: `Figure_Upload/HashrateETHETC_Fig_2` (PNG, PDF).

```sh
python code/code_manuscript/figure_06_energy.py
```
Output: `Figure_Upload/fig_energy_td_vs_bu_Fig_6` (PNG, PDF).

```sh
python code/code_manuscript/figure_07_threshold.py
```
Output: `Figure_Upload/fig_profitability_threshold_Fig_7` (PNG, PDF).

```sh
python code/code_manuscript/figure_09_etc_energy.py
```
Output: `Figure_Upload/ETCEnergy_Fig_9` (PNG, PDF).

```sh
python code/code_manuscript/figure_11_var_irf.py
```
Output: `Figure_Upload/ext_var_irf_Fig_11` (PNG, PDF).

## Main Manuscript Tables: Run Individually

```sh
python code/code_manuscript/table_mc_summary.py
```
Output: `Table_Upload/Table_MC_Summary` (CSV, TEX).

```sh
python code/code_manuscript/table_postmerge_budget.py
```
Output: `Table_Upload/Table_PostMerge_Budget` (CSV, TEX).

```sh
python code/code_manuscript/table_granger.py
```
Output: `Table_Upload/Table_Granger` (CSV, TEX).

```sh
python code/code_manuscript/table_windows.py
```
Output: `Table_Upload/windows` (CSV, TEX).

```sh
python code/code_manuscript/table_window_methods.py
```
Output: `Table_Upload/Table_Window_Methods` (TEX).

## Additional Results

| Execution command | Output |
|---|---|
| `python code/code_other/figure_price.py` | `Figure/PriceETHETC` (PNG, PDF) |
| `python code/code_other/figure_migration.py` | `Figure/fig_hashrate_migration` (PNG, PDF) |
| `python code/code_other/figure_bottomup.py` | `Figure/fig_energy_bottomup` (PNG, PDF) |
| `python code/code_other/figure_cohort.py` | `Figure/fig_cohort_split` (PNG, PDF) |
| `python code/code_other/figure_sensitivity_series.py` | `Figure/ext_sensitivity_ce_series` (PNG, PDF) |
| `python code/code_other/figure_sensitivity_bar.py` | `Figure/ext_sensitivity_ce_bar` (PNG, PDF) |
| `python code/code_other/figure_sensitivity_panel.py` | `Figure/fig_panel_legends_bottom` (PNG, PDF) |
| `python code/code_other/table_sensitivity.py` | `tables/tab_sensitivity_ce` (CSV, TEX) |
| `python code/code_other/table_energy_saving.py` | `tables/EnergySaving` (CSV, TEX) |
| `python code/code_other/figure_var_bic_irf.py` | `Figure/ext_var_irf_BIC` (PNG, PDF) |
| `python code/code_other/figure_var_bic_fevd.py` | `Figure/ext_var_fevd` (PNG, PDF) |
| `python code/code_other/table_granger_bic.py` | `tables/tab_granger_causality_BIC` (CSV, TEX) |

## Data, Computation, and Existing Original Figures

Inputs consist of four Excel files in `data/original`, five CSV files in `data/extension/inputs`, and `code/shared/settings.py`.

Each result script documents its **INPUT** and **OUTPUT** at the top. CSV/LaTeX and PNG/PDF are alternative formats of the same results.

Frozen Monte Carlo (MC) values in `code/shared/settings.py` are used only for verification, not as substitutes for computation. Input data SHA-256 hashes are checked upon loading.

Previously validated main-manuscript MC, budget, Granger, and window tables are preserved. The fixed VAR(14) table and the original figures based on the BIC-selected model are computed separately.

The budget table uses a historical combined-chain estimate of approximately **0.213 GW**; the MC analysis uses an internal stable-window budget of approximately **0.210 GW**.

The electricity-price sensitivity calculation covers **ETC only**, although restored historical figure labels may still mention ETC+ETHW.

The methodology table template is `code/shared/window_methods.tex`; the preserved original table is `Table_Upload/Results_keep.rtf`.

The 14 existing original PNG figures remain unchanged by default, with their verification values recorded in `code/shared/settings.py`. Figure functions still recompute results and generate PDFs. To regenerate an individual PNG manually, for example:

```sh
ENERGY_RERENDER_PNG=1 python code/code_manuscript/figure_02_hashrate.py
```

Regeneration follows the original plotting logic but may produce minor differences in rendering boundaries.

Execution-check summaries and logs are stored in `execution_check/`; no JSON files are required.

LaTeX tables require `booktabs`. The methodology table additionally requires `float`, `tabularx`, and `array`.

## Code Directory Structure

- **`code/code_manuscript/`**: 10 result scripts for five main-manuscript figures and five tables.
- **`code/code_other/`**: 12 result scripts for nine additional figures and three tables.
- **`code/shared/`**: Shared input loaders, numerical models, plotting styles, output-saving functions, and the methodology table template. These files are imported automatically and do not need to be executed manually.
