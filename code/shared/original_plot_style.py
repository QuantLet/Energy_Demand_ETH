"""Original figure style and saving functions; paths alone are updated.
INPUT: original plot figure/axes. OUTPUT: routed PNG/PDF at original 300 dpi.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .io_utils import plt,mdates,ROOT,MAIN_FIGURES

def original_style():
    sns.set_theme(style='ticks')
    plt.rcParams.update({'figure.figsize':(10,6),'axes.titlesize':14,
                        'axes.labelsize':12,'legend.fontsize':10,
                        'lines.linewidth':1.5,'axes.spines.top':False,
                        'axes.spines.right':False})

def destination(filename):
    name=__import__('pathlib').Path(filename).stem
    folder=ROOT/('Figure_Upload' if name in MAIN_FIGURES else 'Figure')
    folder.mkdir(exist_ok=True)
    return folder/(name+'.png')

def save_basic(fig, ax, filename):
    """Apply consistent publication-style formatting and save figure."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    ax.grid(False)
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.minorticks_off()
    try:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    except Exception:
        pass
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    (handles, labels) = ax.get_legend_handles_labels()
    if handles:
        if ax.get_legend():
            ax.get_legend().remove()
        ncol = min(len(labels), 4)
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=ncol, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    save_png(fig,destination(filename), dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(destination(filename).with_suffix('.pdf'), bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'Saved: {filename}')

def add_merge_line(ax, label=True):
    fig=ax.figure
    ax.axvline(pd.Timestamp('2022-09-14'), color='green', linestyle='--', linewidth=1.2)

def save_extended(fig, ax, filename):
    """Consistent style: transparent, no grid, legend below axes, sparse ticks."""
    ax.grid(False)
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    try:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    except Exception:
        pass
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    (handles, labels) = ax.get_legend_handles_labels()
    if handles:
        if ax.get_legend():
            ax.get_legend().remove()
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=min(len(labels), 4), framealpha=0.0, fontsize=9, borderaxespad=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_png(fig,destination(filename), dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(destination(filename).with_suffix('.pdf'),bbox_inches='tight',transparent=True)
    plt.close(fig)
    print(f'    Saved: {filename}')

def save_multi(fig, filename):
    """Save a figure with multiple axes (VAR plots), no style override."""
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.set_facecolor('none')
        ax.grid(False)
    save_png(fig,destination(filename), dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(destination(filename).with_suffix('.pdf'),bbox_inches='tight',transparent=True)
    plt.close(fig)
    print(f'    Saved: {filename}')


def save_png(fig,path,**kwargs):
    """Preserve an existing original PNG exactly; still recompute the figure/PDF.
    INPUT: computed figure and intended PNG path. OUTPUT: PNG only if absent,
    altered, or ENERGY_RERENDER_PNG=1. Original SHA-256 is recorded in settings.py.
    This explicit preservation is requested for already existing manuscript art.
    """
    import hashlib,os
    from .settings import ORIGINAL_FIGURES
    manifest=ORIGINAL_FIGURES
    item=manifest.get(str(path.relative_to(ROOT)))
    if os.environ.get('ENERGY_RERENDER_PNG')!='1' and path.exists() and item:
        if hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:
            return
    fig.savefig(path,**kwargs)
