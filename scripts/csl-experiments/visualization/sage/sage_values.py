"""Plot SAGE values and their differences to SAGE_CG and SAGE_CG_CD"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


# Functions from utils file of rfi.plots
def coord_height_to_pixels(ax, height):
    p1 = ax.transData.transform((0, height))
    p2 = ax.transData.transform((0, 0))

    pix_height = p1[1] - p2[1]
    return pix_height


def hbar_text_position(rect, x_pos=0.5, y_pos=0.5):
    rx, ry = rect.get_xy()
    width = rect.get_width()
    height = rect.get_height()

    tx = rx + (width * x_pos)
    ty = ry + (height * y_pos)
    return (tx, ty)


def fi_hbarplot(ex, textformat='{:5.2f}', ax=None, figsize=None):
    """
    Function that plots the result of an RFI computation as a barplot
    Args:
        figsize:
        ax:
        textformat:
        ex: Explanation object
    """

    names = diff_sage_cg.index
    rfis = ex['mean'].to_numpy()
    stds = ex['std'].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ixs = np.arange(rfis.shape[0] + 0.5, 0.5, -1)

    ax.barh(ixs, rfis, tick_label=names, xerr=stds, capsize=5, color=['mistyrose',
                                                                      'salmon', 'tomato',
                                                                      'darksalmon', 'coral'])

    # color = ['lightcoral',
    #          'moccasin', 'darkseagreen',
    #          'paleturquoise', 'lightsteelblue']
    return ax


# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# data, model
datasets = ["dag_s", "dag_sm", "asia"]
model = "lm"
model_discrete = "rf"

# initiate plot, fill it in for loop
fig, ax = plt.subplots(5, 3, figsize=(7, 4))
fig.tight_layout(pad=2.1)
ax[0, 0].set_title('DAG$_s$')
ax[0, 1].set_title('DAG$_{sm}$')
ax[0, 0].set_title('Asia')

# (for defense)
#ax[0, 0].set_title('SAGE')
#ax[0, 1].set_title('SAGE - SAGE$_{cg}^{o,*}$')
#ax[0, 2].set_title('SAGE - SAGE$_{cg}^{*}$')

# load data
for i in range(2):
    data = datasets[i]
    if i < 2:
        sage = pd.read_csv(f"experiments_cg/results/continuous/true_amat/{data}/sage_{data}_{model}.csv")
    else:
        sage = pd.read_csv(f"experiments_cg/results/discrete/true_amat/{data}/sage_{data}_{model_discrete}.csv")
    sage_mean = sage["mean"]
    sage_mean = abs(sage_mean)
    sage_ordered = sage_mean.sort_values(ascending=False)
    sage_five = sage_ordered.iloc[0:5]
    indices = sage_five.index
    labels = []

    if i < 2:
        for a in range(5):
            labels.append(int(indices[a]+2))
    else:
        for a in range(5):
            labels.append(sage["feature"][indices[a]])

    if i < 2:
        sage_r = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/sage_r_{data}_{model}.csv")
        cg_r = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/cg_r_{data}_{model}.csv")
        cg_cd_r = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/cg_cd_r_{data}_{model}.csv")
        cg_r_est = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/{data}/cg_r_{data}_{model}.csv")
        cg_cd_r_est = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/{data}/cg_cd_r_{data}_{model}.csv")
    else:
        sage_r = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/{data}/sage_r_{data}_{model_discrete}.csv")
        cg_r = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/{data}/cg_r_{data}_{model_discrete}.csv")
        cg_cd_r = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/{data}/cg_cd_r_{data}_{model_discrete}.csv")
        cg_r_est = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/est_amat/{data}/cg_r_{data}_{model_discrete}.csv")
        cg_cd_r_est = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/est_amat/{data}/cg_cd_r_{data}_{model_discrete}.csv")

    sage = sage[sage["feature"].isin(labels)]
    labels_str = []
    for a in range(5):
        labels_str.append(str(indices[a]+2))

    # differences between sage and sage_cg per run
    sage_cg = sage_r - cg_r
    sage_cg = sage_cg.drop(['sample'], axis=1)
    # make df of differences w corresponding stds
    diff_sage_cg = pd.DataFrame(sage_cg.mean(), columns=['mean'])
    diff_sage_cg['std'] = sage_cg.std()
    diff_sage_cg.index.set_names(['feature'], inplace=True)
    diff_sage_cg['feature'] = diff_sage_cg.index
    if i < 2:
        diff_sage_cg = diff_sage_cg[diff_sage_cg["feature"].isin(labels_str)]
    else:
        diff_sage_cg = diff_sage_cg[diff_sage_cg["feature"].isin(labels)]
    # differences between sage and sage_cg_cd per run
    sage_cg_cd = sage_r - cg_cd_r
    sage_cg_cd = sage_cg_cd.drop(['sample'], axis=1)
    # make df of differences w corresponding stds
    diff_sage_cg_cd = pd.DataFrame(sage_cg_cd.mean(), columns=['mean'])
    diff_sage_cg_cd['std'] = sage_cg_cd.std()
    diff_sage_cg_cd.index.set_names(['feature'], inplace=True)
    diff_sage_cg_cd['feature'] = diff_sage_cg_cd.index
    if i < 2:
        diff_sage_cg_cd = diff_sage_cg_cd[diff_sage_cg_cd["feature"].isin(labels_str)]
    else:
        diff_sage_cg_cd = diff_sage_cg_cd[diff_sage_cg_cd["feature"].isin(labels)]
    # differences between sage and sage_cg per run
    sage_cg_est = sage_r - cg_r_est
    sage_cg_est = sage_cg_est.drop(['sample'], axis=1)
    # make df of differences w corresponding stds
    diff_sage_cg_est = pd.DataFrame(sage_cg_est.mean(), columns=['mean'])
    diff_sage_cg_est['std'] = sage_cg_est.std()
    diff_sage_cg_est.index.set_names(['feature'], inplace=True)
    diff_sage_cg_est['feature'] = diff_sage_cg_est.index
    if i < 2:
        diff_sage_cg_est = diff_sage_cg_est[diff_sage_cg_est["feature"].isin(labels_str)]
    else:
        diff_sage_cg_est = diff_sage_cg_est[diff_sage_cg_est["feature"].isin(labels)]
    # differences between sage and sage_cg_cd per run
    sage_cg_cd_est = sage_r - cg_cd_r_est
    sage_cg_cd_est = sage_cg_cd_est.drop(['sample'], axis=1)
    # make df of differences w corresponding stds
    diff_sage_cg_cd_est = pd.DataFrame(sage_cg_cd_est.mean(), columns=['mean'])
    diff_sage_cg_cd_est['std'] = sage_cg_cd_est.std()
    diff_sage_cg_cd_est.index.set_names(['feature'], inplace=True)
    diff_sage_cg_cd_est['feature'] = diff_sage_cg_cd_est.index
    if i < 2:
        diff_sage_cg_cd_est = diff_sage_cg_cd_est[diff_sage_cg_cd_est["feature"].isin(labels_str)]
    else:
        diff_sage_cg_cd_est = diff_sage_cg_cd_est[diff_sage_cg_cd_est["feature"].isin(labels)]

    # plots
    fi_hbarplot(sage, ax=ax[0, i])
    fi_hbarplot(diff_sage_cg, ax=ax[1, i])
    fi_hbarplot(diff_sage_cg_cd, ax=ax[2, i])
    fi_hbarplot(diff_sage_cg_est, ax=ax[3, i])
    fi_hbarplot(diff_sage_cg_cd_est, ax=ax[4, i])

#ax[0, 0].set_ylabel('DAG$_s$')
#ax[1, 0].set_ylabel('DAG$_{sm}$')
ax[0].set_xlabel('SAGE')
ax[1].set_xlabel('SAGE-SAGE$_{cg}^{o}$')
ax[2].set_xlabel('SAGE-SAGE$_{cg}$')
ax[3].set_xlabel('SAGE-SAGE$_{cg}^{o,*}$')
ax[4].set_xlabel('SAGE-SAGE$_{cg}^{*}$')

plt.savefig(f"visualization/sage_values_lm_def.png", dpi=600, transparent=True)
