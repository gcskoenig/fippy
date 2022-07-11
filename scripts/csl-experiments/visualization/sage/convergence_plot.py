"""Plot convergence behaviour of SAGE Experiments"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from scipy.stats import norm    # TODO add to requirements? or is scipy(.stats) already included?


# load data
sage = pd.read_csv("scripts/csl-experiments/visualization/sage/sage_o_dag_s_lm.csv")


def convergence_plot(data, top=None, bottom=None, choose=None, latex_font=False, alpha=0.05, ax=None, figsize=None,
                     legend=True, loc='upper right'):
    """
    Function that plots the result of an RFI computation as a convergence plot based on the values per ordering
    Args:
        data: explanation object from package
        top: number of top values to be plotted (cannot be combined with bottom)
        bottom: number of bottom values to be plotted (cannot be combined with top)
        choose: [x,y], x int, y int, range of values to plot, x is (x+1)th largest, y is y-th largest value
        latex_font: Bool - toggle LaTeX font
        alpha: alpha for confidence bands
        ax:
        figsize:
        legend: addd legend to plot/axes
        loc: position of legend; default: 'upper right' # TODO (cl): custom location of legend
    """

    # TODO (cl) check correct syntax for version > 3.8
    # TODO (cl) for package uncomment next line, delete l33
    # data = data.scores.groupby(level='orderings').mean()
    data = data
    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig, ax = plt.subplots(figsize=figsize)

    # drop column 'ordering' if it is present
    if 'ordering' in data.columns:
        data = data.drop(['ordering'], axis=1)

    # latex font TODO (cl) more generic? use all plt options?
    if latex_font:
        # for latex font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    # get the sage values (mean across all orderings)
    sage_mean = data.mean()

    if (top is not None and bottom is not None) or (top is not None and choose is not None) \
            or (bottom is not None and choose is not None):
        raise ValueError("Arguments top, bottom or choose cannot be used together.")

    if top is not None:
        # absolute values to rank sage values and then retrieve the top values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=False)   # i.e descending -> top first
        sage_top = sage_ordered.iloc[0:top]
        # indices of top values
        indices = sage_top.index

    elif bottom is not None:
        # absolute values to rank sage values and then retrieve the bottom values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=True)
        sage_bottom = sage_ordered.iloc[0:bottom]
        # indices of bottom values
        indices = sage_bottom.index

    elif choose is not None:
        # absolute values to rank sage values and then retrieve the bottom values
        sage_mean = abs(sage_mean)
        sage_ordered = sage_mean.sort_values(ascending=False)
        sage_choose = sage_ordered.iloc[choose[0]:choose[1]]
        # indices of bottom values
        indices = sage_choose.index

    else:
        indices = sage_mean.index

    # trim to relevant data
    data = data[indices]

    # get the standard deviations after every ordering starting with the third
    std_sage = pd.DataFrame(columns=data.columns)
    for i in range(2, len(data)):
        # TODO rewrite and shorten
        diffs = data[0:i+1] - data[0:i+1].mean()
        # squared differences
        diffs2 = diffs*diffs
        # sum of squared diffs
        diffs2_sum = diffs2.sum()
        # sum of diffs
        diffs_sum = diffs.sum()
        # diffs_sum2 = (diffs_sum * diffs_sum)
        # diffs_sum2_n = (diffs_sum2/ii)
        variance = (diffs2_sum - ((diffs_sum * diffs_sum)/i)) / (i - 1)
        std = variance**0.5
        std_sage.loc[i-2] = std

    # running means up to current ordering
    running_mean = pd.DataFrame(columns=data.columns)
    for j in range(2, len(data)):
        running_mean.loc[j] = data[0:j + 1].mean()
    running_mean = running_mean.reset_index(drop=True)

    # make confidence bands
    lower = pd.DataFrame(columns=running_mean.columns)
    upper = pd.DataFrame(columns=running_mean.columns)
    for k in range(len(running_mean)):
        # NOTE: k+3 because first 3 rows were dropped before
        # TODO: use t-distribution (in general: Gaussian in the limit, because every summand is a RV -> are they indep?)
        lower.loc[k] = running_mean.loc[k] + norm.ppf(alpha/2) * (std_sage.loc[k] / np.sqrt(k + 3))
        upper.loc[k] = running_mean.loc[k] - norm.ppf(alpha/2) * (std_sage.loc[k] / np.sqrt(k + 3))

    x_axis = []
    for ll in range(len(running_mean)):
        x_axis.append(ll)

    # fig = plt.figure(figsize=(5,5))
    ax.plot(x_axis, running_mean, linewidth=0.7)

    for n in lower.columns:
        ax.fill_between(x_axis, lower[n], upper[n], alpha=.1)

    labels = data.columns
    if legend:
        ax.legend(loc=loc, labels=labels)
    return ax


# fig = convergence_plot(sage, top=5)
# plt.show()


# initiate plot, fill it in for loop
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
fig.tight_layout(pad=1.8)

ax[0, 0].set_title('DAG$_s$')
ax[0, 1].set_title('DAG$_{sm}$')

ax[0, 0] = convergence_plot(sage, choose=[3, 6], ax=ax[0, 0], legend=False)
ax[0, 1] = convergence_plot(sage, choose=[3, 6], ax=ax[0, 1], legend=False)
ax[1, 0] = convergence_plot(sage, choose=[3, 6], ax=ax[1, 0], legend=False)
ax[1, 1] = convergence_plot(sage, choose=[3, 6], ax=ax[1, 1])


ax[0, 0].set_ylabel('DAG$_s$')
ax[1, 0].set_ylabel('DAG$_{sm}$')

ax[1, 0].set_xlabel('SAGE')
ax[1, 1].set_xlabel('SAGE$_{cg}^{o}$')

fig.text(0.5, 0, 'No. of Permutations', ha='center')

plt.show()
