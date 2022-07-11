""""Simple visualization of all SAGE value"""
import matplotlib.pyplot as plt
import rfi.plots._snsstyle  # set default style
import numpy as np
import math
from rfi.plots._utils import hbar_text_position, coord_height_to_pixels, get_line_hlength
import seaborn as sns
import pandas as pd

def fi_hbarplot(ex, textformat='{:5.2f}', ax=None, figsize=None, name=False):
    """
    Function that plots the result of an RFI computation as a barplot
    Args:
        figsize:
        ax:
        textformat:
        ex: Explanation object
    """
    if name:
        names = ex["feature"]
    else:
        names = ex.index + 2

    rfis = ex['mean'].to_numpy()
    stds = ex['std'].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ixs = np.arange(rfis.shape[0] + 0.5, 0.5, -1)

    ax.barh(ixs, rfis, tick_label=names, xerr=stds, capsize=5)

    for jj in range(len(ax.patches)):
        rect = ax.patches[jj]
        tx, ty_lower = hbar_text_position(rect, y_pos=0.25)
        tx, ty_upper = hbar_text_position(rect, y_pos=0.75)
        pix_height = coord_height_to_pixels(ax, rect.get_height()) / 4
        pix_height = math.floor(pix_height)
        ax.text(tx, ty_upper, textformat.format(rect.get_width()),
                va='center', ha='center', size=pix_height)
        ax.text(tx, ty_lower, '+-' + textformat.format(stds[jj]),
                va='center', ha='center', size=pix_height)
    return ax

# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

what = "sage"

#sage_s = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/dag_s/{what}_dag_s_lm.csv")
#sage_sm = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/dag_sm/{what}_dag_sm_lm.csv")
#sage_srf = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/dag_s/{what}_dag_s_rf.csv")
#sage_smrf = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/dag_sm/{what}_dag_sm_rf.csv")
sage_a = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/hepar/{what}_hepar_cnb.csv")
sage_arf = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/hepar/{what}_hepar_rf.csv")

#fi_hbarplot(sage_s)
#plt.ylabel("X$_i$")
#plt.xlabel("SAGE Values")
#plt.savefig(f"visualization/sage/sage_values_dag_s_lm.png", dpi=400)
#plt.clf()
#
#fi_hbarplot(sage_sm, figsize=(4, 6))
#plt.ylabel("X$_i$")
#plt.xlabel("SAGE Values")
#plt.savefig(f"visualization/sage/sage_values_dag_sm_lm.png", dpi=400)
#plt.clf()

#fi_hbarplot(sage_srf)
#plt.ylabel("X$_i$")
#plt.xlabel("SAGE Values")
#plt.savefig(f"visualization/sage/sage_values_dag_s_rf.png", dpi=400)
#plt.clf()
#
#fi_hbarplot(sage_smrf, figsize=(4, 6))
#plt.ylabel("X$_i$")
#plt.xlabel("SAGE Values")
#plt.savefig(f"visualization/sage/sage_values_dag_sm_rf.png", dpi=400)
#plt.clf()


fi_hbarplot(sage_a, name=True)
plt.xlabel("SAGE Values")
plt.savefig(f"visualization/sage/sage_values_asia_cnb.png", dpi=400)
plt.clf()

fi_hbarplot(sage_arf, name=True)
plt.xlabel("SAGE Values")
plt.savefig(f"visualization/sage/sage_values_asia_rf.png", dpi=400)
plt.clf()
