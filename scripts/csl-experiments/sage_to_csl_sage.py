"""Use results over orderings from sage estimation, check for d-separation and set value to zero"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

orderings = pd.read_csv("scripts/csl-experiments/new_results/results/continuous/dag_s/order_sage_dag_s_0.22222_lm.csv")
values = pd.read_csv("scripts/csl-experiments/new_results/results/continuous/dag_s/sage_o_dag_s_0.22222_lm.csv")
if 'ordering' in values.columns:
    values = values.drop(['ordering'], axis=1)
adj_mat = pd.read_csv("scripts/csl-experiments/data/true_amat/dag_s_0.22222.csv")

# dag_s amat to zeros and ones
col_names_str = []
for k in range(len(adj_mat.columns)):
    col_names_str.append(str(k+1))
adj_mat.columns = col_names_str
mapping_rf = {False: 0, True: 1}
col_names = adj_mat.columns
for j in col_names:
    adj_mat[j] = adj_mat.replace({j: mapping_rf})[j]
# modify adjacency matrix for use in networkx package
adj_mat = adj_mat.set_axis(col_names, axis=0)

target = "10"

# create graph
g = nx.DiGraph(adj_mat)
predictors = adj_mat.columns.drop(target)

# initiate vector for the estimated value for the summand when a d-separation is found
diffs = []

for i in range(len(orderings)):
    # TODO if it is the first ordering to go through, set value to zero if dsep found, else set value to value
    # from ordering before, also safe the difference b/w current value and value from ordering before
    # make 'ordering string' a list of numerics
    ordering = orderings["ordering"][i]     # this is a string
    ordering = filter(str.isdigit, ordering)
    ordering = " ".join(ordering)
    ordering = ordering.split()     # this is a list of strings
    # if list of integers required, uncomment following two lines
    # ordering = map(int, ordering)
    # ordering = list(ordering)
    for j in range(len(ordering)):
        column_index = int(ordering[j]) - 1
        if j == 0:
            J = set(ordering[0:j])
            C = set(ordering[j:])
            d_sep = nx.d_separated(g, J, {target},  C)
            if d_sep:
                print("yes")
                if i == 0:
                    diffs.append(values.iloc[i, column_index])
                    values.iloc[i, column_index] = 0
                else:
                    diffs.append(values.iloc[i, column_index]-values.iloc[i-1, column_index])
                    values.iloc[i, column_index] = values.iloc[i-1, column_index]   # TODO (cl): set to zero
            else:
                print("no")

        else:
            J = set(ordering[0:j])
            C = set(ordering[j:])
            d_sep = nx.d_separated(g, J, {target}, C)

            if d_sep:
                print("yes")
                if i == 0:
                    diffs.append(values.iloc[i, column_index])
                    values.iloc[i, column_index] = 0
                else:
                    diffs.append(values.iloc[i, column_index]-values.iloc[i-1, column_index])
                    values.iloc[i, column_index] = values.iloc[i-1, column_index]   # TODO (cl): set to zero
            else:
                print("no")

# TODO plot the differences between you know what

values.to_csv("scripts/csl-experiments/new_results/results/continuous/dag_s/csl_sage_o_dag_s_0.22222_lm.csv")


def detect_conv(scores, ii, threshold, extra_orderings=0):  # TODO (cl) make class?
    """Detect convergence for SAGE values up to the current ordering (avg over all runs)
        when 'largest sd is sufficiently low proportion of range of estimated values' (Covert
        et al., 2020; p.6)

    Args:
        scores: Dataframe of scores as in explainer l. 725
        ii: current ordering in SAGE estimation
        threshold: Threshold for convergence detection
        extra_orderings: orderings after convergence has been detected, default: 0
        """
    if ii == 0 or ii == 1:
        # the first two orderings are not sufficient to detect convergence
        return False
    else:
        # input scores are nr_runs runs per ordering, mean makes it one value per ordering
        # scores = scores.loc[(slice(0, ii), slice(None), slice(None))].groupby('ordering').mean()
        if 'ordering' in scores.columns:
            scores = scores.drop(['ordering'], axis=1)

        # TODO (cl) Use Welford's algorithm when making class and continuously update
        # diffs = scores - scores.mean()
        # diffs2 = diffs * diffs
        # diffs2_sum = diffs2.sum()
        # diffs_sum = diffs.sum()
        # variance = (diffs2_sum - ((diffs_sum * diffs_sum) / ii)) / (ii - 1)
        # ratio = ((variance ** 0.5) / np.sqrt(ii)) / (scores.max() - scores.min())
        # max_ratio = ratio.max()

        variance = np.var(scores)*(ii/(ii-1))
        ratio = ((variance ** 0.5) / np.sqrt(ii)) / (scores.max() - scores.min())
        max_ratio = ratio.max()

        if max_ratio < threshold:
            if extra_orderings == 0:
                # stop when convergence detected
                return True
            else:
                # extra runs to verify flat curve after convergence has been detected
                extra_orderings -= 1
                return int(extra_orderings)
        else:
            # convergence not yet detected
            return False


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
        # TODO: Why, again, can I use Gaussian distribution here?
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


tf = []
for k in range(1000):
    tf.append(detect_conv(values, k, 0.01))
detected_at = 1000 - sum(tf)

fig_m, ax = plt.subplots(1, 1, figsize=(6, 6))

ax = convergence_plot(values, top=5, ax=ax, legend=False)
ax.axvline(x=detected_at, color='k', linestyle='--')

fig_m.suptitle('DAG S 0.22', fontsize=14)
plt.savefig("meeting_may_25/csl_sage_test_top5.png")

# plot the differences
