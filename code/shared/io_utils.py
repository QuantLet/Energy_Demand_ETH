"""INPUT: calculated DataFrames / figures. OUTPUT: routed main and supplementary results from independent result functions.
Tables have CSV and LaTeX versions; figures have PNG and vector PDF versions.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / 'Table_Upload'
OTHER_TABLES = ROOT / 'tables'
FIGURE = ROOT / 'Figure_Upload'
OTHER_FIGURES = ROOT / 'Figure'
MAIN_FIGURES = {'HashrateETHETC_Fig_2','fig_energy_td_vs_bu_Fig_6','fig_profitability_threshold_Fig_7','ETCEnergy_Fig_9','ext_var_irf_Fig_11'}
COLORS = ['#1764a0','#dd8531','#269584','#9657a1']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'figure.figsize':(8,4.3)})


def save_figure(fig, name):
    folder = FIGURE if name in MAIN_FIGURES else OTHER_FIGURES
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def save_table(name, frame, header, lines, spec, caption, note='', label=None):
    TABLE.mkdir(exist_ok=True)
    frame.to_csv(TABLE/(name+'.csv'),index=False)
    text='\\begin{table}[htbp]\n\\centering\n\\small\n\\caption{'+caption+'}\n'
    if label: text+='\\label{'+label+'}\n'
    text+='\\begin{tabular}{'+spec+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,line))+r' \\' for line in lines)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    if note: text+='\\par\\smallskip\n\\begin{minipage}{\\linewidth}\n\\footnotesize\n\\textit{Note:} '+note+'\n\\end{minipage}\n'
    text+='\\end{table}\n'
    (TABLE/(name+'.tex')).write_text(text)


def save_supplement_table(name,frame):
    """INPUT: computed frame; OUTPUT: tables/<name>.csv and .tex."""
    OTHER_TABLES.mkdir(exist_ok=True)
    frame.to_csv(OTHER_TABLES/(name+'.csv'),index=False)
    frame.to_latex(OTHER_TABLES/(name+'.tex'),index=False,escape=True,float_format=lambda x:f'{x:.6g}')
