"""Plot convergence behaviour of SAGE Experiments"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np


# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# data, model
datasets = ["dag_s", "dag_sm", "asia"]
model = "rf"
model_discrete = "cnb"

# initiate plot, fill it in for loop
fig, ax = plt.subplots(5, 3, figsize=(8, 3))
fig.tight_layout(pad=1.8)

'''
for 'transpose' grid:
ax[0, 0].set_title('SAGE')
ax[0, 1].set_title('SAGE$_{cg}^{o,*}$')
ax[0, 2].set_title('SAGE$_{cg}^{*}$')'''

ax[0, 0].set_title('DAG$_s$')
ax[0, 1].set_title('DAG$_{sm}$')
ax[0, 2].set_title('Asia')

# load data
for i in range(2):

    if i < 2:
        data = datasets[i]
        sage = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/sage_{data}_{model}.csv")
    else:
        data = datasets[i]
        sage = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/{data}/sage_{data}_{model_discrete}.csv")

    sage_mean = sage["mean"]
    sage_mean = abs(sage_mean)
    sage_ordered = sage_mean.sort_values(ascending=False)
    sage_five = sage_ordered.iloc[0:5]
    indices = sage_five.index
    labels = []

    for a in range(5):
        labels.append(str(indices[a]+2))

    if data == "dag_s":
        sage_o = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/sage_o_{data}_{model}.csv")[0:500]
        cg_o = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/cg_o_{data}_{model}.csv")[0:500]
        cg_cd_o = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/cg_cd_o_{data}_{model}.csv")[0:500]
        cg_o_est = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/{data}/cg_o_{data}_{model}.csv")[0:500]
        cg_cd_o_est = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/{data}/cg_cd_o_{data}_{model}.csv")[0:500]
    if data == "dag_sm":
        sage_o = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/sage_o_{data}_{model}.csv")
        cg_o = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/cg_o_{data}_{model}.csv")
        cg_cd_o = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/true_amat/{data}/cg_cd_o_{data}_{model}.csv")
        cg_o_est = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/{data}/cg_o_{data}_{model}.csv")
        cg_cd_o_est = pd.read_csv(f"rfi/examples/experiments_cg/results/continuous/est_amat/{data}/cg_cd_o_{data}_{model}.csv")
    if data == "asia":
        sage_o = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/{data}/sage_o_{data}_{model_discrete}.csv")
        cg_o = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/{data}/cg_o_{data}_{model_discrete}.csv")
        cg_cd_o = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/true_amat/{data}/cg_cd_o_{data}_{model_discrete}.csv")
        cg_o_est = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/est_amat/{data}/cg_o_{data}_{model_discrete}.csv")
        cg_cd_o_est = pd.read_csv(f"rfi/examples/experiments_cg/results/discrete/est_amat/{data}/cg_cd_o_{data}_{model_discrete}.csv")

    sage_o = sage_o.drop(['ordering'], axis=1)
    cg_o = cg_o.drop(['ordering'], axis=1)
    cg_cd_o = cg_cd_o.drop(['ordering'], axis=1)
    cg_o_est = cg_o_est.drop(['ordering'], axis=1)
    cg_cd_o_est = cg_cd_o_est.drop(['ordering'], axis=1)

    if i < 2:
        sage_o = sage_o[labels]
        cg_o = cg_o[labels]
        cg_cd_o = cg_cd_o[labels]
        cg_o_est = cg_o_est[labels]
        cg_cd_o_est = cg_cd_o_est[labels]

    else:
        sage_o = sage_o.iloc[:, indices]
        cg_o = cg_o.iloc[:, indices]
        cg_cd_o = cg_cd_o.iloc[:, indices]
        cg_o_est = cg_o_est.iloc[:, indices]
        cg_cd_o_est = cg_cd_o_est.iloc[:, indices]

    # same for sage cg  cg_cd cg_est cg_cd_est
    std_sage = pd.DataFrame(columns=sage_o.columns)

    for j in range(2, len(sage_o)):
        diffs = sage_o[0:j+1] - sage_o[0:j+1].mean()
        # squared differences
        diffs2 = diffs*diffs
        # sum of squared diffs
        diffs2_sum = diffs2.sum()
        # sum of diffs
        diffs_sum = diffs.sum()
        # diffs_sum2 = (diffs_sum * diffs_sum)
        # diffs_sum2_n = (diffs_sum2/ii)
        variance = (diffs2_sum - ((diffs_sum * diffs_sum)/j)) / (j - 1)
        std = variance**0.5
        std_sage.loc[j-2] = std

    # get the means up to current ordering

    sage_running_mean = pd.DataFrame(columns=sage_o.columns)
    for k in range(2, len(sage_o)):
        sage_running_mean.loc[k] = sage_o[0:k+1].mean()

    sage_running_mean = sage_running_mean.reset_index(drop=True)

    # make confidence bands
    sage_lower = pd.DataFrame(columns=sage_running_mean.columns)
    sage_upper = pd.DataFrame(columns=sage_running_mean.columns)
    for ll in range(len(sage_running_mean)):
        sage_lower.loc[ll] = sage_running_mean.loc[ll] - 1.96*(std_sage.loc[ll]/np.sqrt(ll+3))
        sage_upper.loc[ll] = sage_running_mean.loc[ll] + 1.96*(std_sage.loc[ll]/np.sqrt(ll+3))

    x_sage = []
    for m in range(len(sage_running_mean)):
        x_sage.append(m)

    std_cg = pd.DataFrame(columns=cg_o.columns)
    for j in range(2, len(cg_o)):
        diffs = cg_o[0:j+1] - cg_o[0:j+1].mean()
        # squared differences
        diffs2 = diffs*diffs
        # sum of squared diffs
        diffs2_sum = diffs2.sum()
        # sum of diffs
        diffs_sum = diffs.sum()
        # diffs_sum2 = (diffs_sum * diffs_sum)
        # diffs_sum2_n = (diffs_sum2/ii)
        variance = (diffs2_sum - ((diffs_sum * diffs_sum)/j)) / (j - 1)
        std = variance**0.5
        std_cg.loc[j-2] = std

    # get the means up to current ordering

    cg_running_mean = pd.DataFrame(columns=cg_o.columns)
    for k in range(2, len(cg_o)):
        cg_running_mean.loc[k] = cg_o[0:k+1].mean()

    cg_running_mean = cg_running_mean.reset_index(drop=True)

    # make confidence bands
    cg_lower = pd.DataFrame(columns=cg_running_mean.columns)
    cg_upper = pd.DataFrame(columns=cg_running_mean.columns)
    for ll in range(len(cg_running_mean)):
        cg_lower.loc[ll] = cg_running_mean.loc[ll] - 1.96*(std_cg.loc[ll]/np.sqrt(ll+3))
        cg_upper.loc[ll] = cg_running_mean.loc[ll] + 1.96*(std_cg.loc[ll]/np.sqrt(ll+3))

    x_cg = []
    for m in range(len(cg_running_mean)):
        x_cg.append(m)

    std_cg_cd = pd.DataFrame(columns=cg_cd_o.columns)
    for j in range(2, len(cg_cd_o)):
        diffs = cg_cd_o[0:j+1] - cg_cd_o[0:j+1].mean()
        # squared differences
        diffs2 = diffs*diffs
        # sum of squared diffs
        diffs2_sum = diffs2.sum()
        # sum of diffs
        diffs_sum = diffs.sum()
        # diffs_sum2 = (diffs_sum * diffs_sum)
        # diffs_sum2_n = (diffs_sum2/ii)
        variance = (diffs2_sum - ((diffs_sum * diffs_sum)/j)) / (j - 1)
        std = variance**0.5
        std_cg_cd.loc[j-2] = std

    # get the means up to current ordering

    cg_cd_running_mean = pd.DataFrame(columns=cg_cd_o.columns)
    for k in range(2, len(cg_cd_o)):
        cg_cd_running_mean.loc[k] = cg_cd_o[0:k+1].mean()

    cg_cd_running_mean = cg_cd_running_mean.reset_index(drop=True)

    # make confidence bands
    cg_cd_lower = pd.DataFrame(columns=cg_cd_running_mean.columns)
    cg_cd_upper = pd.DataFrame(columns=cg_cd_running_mean.columns)
    for ll in range(len(cg_cd_running_mean)):
        cg_cd_lower.loc[ll] = cg_cd_running_mean.loc[ll] - 1.96*(std_cg_cd.loc[ll]/np.sqrt(ll+3))
        cg_cd_upper.loc[ll] = cg_cd_running_mean.loc[ll] + 1.96*(std_cg_cd.loc[ll]/np.sqrt(ll+3))

    x_cg_cd = []
    for m in range(len(cg_cd_running_mean)):
        x_cg_cd.append(m)

    std_cg_est = pd.DataFrame(columns=cg_o_est.columns)
    for j in range(2, len(cg_o)):
        diffs = cg_o_est[0:j+1] - cg_o_est[0:j+1].mean()
        # squared differences
        diffs2 = diffs*diffs
        # sum of squared diffs
        diffs2_sum = diffs2.sum()
        # sum of diffs
        diffs_sum = diffs.sum()
        # diffs_sum2 = (diffs_sum * diffs_sum)
        # diffs_sum2_n = (diffs_sum2/ii)
        variance = (diffs2_sum - ((diffs_sum * diffs_sum)/j)) / (j - 1)
        std = variance**0.5
        std_cg_est.loc[j-2] = std

    # get the means up to current ordering

    cg_running_mean_est = pd.DataFrame(columns=cg_o_est.columns)
    for k in range(2, len(cg_o_est)):
        cg_running_mean_est.loc[k] = cg_o_est[0:k+1].mean()

    cg_running_mean_est = cg_running_mean_est.reset_index(drop=True)

    # make confidence bands
    cg_est_lower = pd.DataFrame(columns=cg_running_mean_est.columns)
    cg_est_upper = pd.DataFrame(columns=cg_running_mean_est.columns)
    for ll in range(len(cg_running_mean_est)):
        cg_est_lower.loc[ll] = cg_running_mean_est.loc[ll] - 1.96*(std_cg_est.loc[ll]/np.sqrt(ll+3))
        cg_est_upper.loc[ll] = cg_running_mean_est.loc[ll] + 1.96*(std_cg_est.loc[ll]/np.sqrt(ll+3))

    x_cg_est = []
    for m in range(len(cg_running_mean_est)):
        x_cg_est.append(m)

    std_cg_cd_est = pd.DataFrame(columns=cg_cd_o_est.columns)
    for j in range(2, len(cg_cd_o_est)):
        diffs = cg_cd_o_est[0:j+1] - cg_cd_o_est[0:j+1].mean()
        # squared differences
        diffs2 = diffs*diffs
        # sum of squared diffs
        diffs2_sum = diffs2.sum()
        # sum of diffs
        diffs_sum = diffs.sum()
        # diffs_sum2 = (diffs_sum * diffs_sum)
        # diffs_sum2_n = (diffs_sum2/ii)
        variance = (diffs2_sum - ((diffs_sum * diffs_sum)/j)) / (j - 1)
        std = variance**0.5
        std_cg_cd_est.loc[j-2] = std

    # get the means up to current ordering

    cg_cd_running_mean_est = pd.DataFrame(columns=cg_cd_o_est.columns)
    for k in range(2, len(cg_cd_o_est)):
        cg_cd_running_mean_est.loc[k] = cg_cd_o_est[0:k+1].mean()

    cg_cd_running_mean_est = cg_cd_running_mean_est.reset_index(drop=True)

    # make confidence bands
    cg_cd_est_lower = pd.DataFrame(columns=cg_cd_running_mean_est.columns)
    cg_cd_est_upper = pd.DataFrame(columns=cg_cd_running_mean_est.columns)
    for ll in range(len(cg_cd_running_mean_est)):
        cg_cd_est_lower.loc[ll] = cg_cd_running_mean_est.loc[ll] - 1.96*(std_cg_cd_est.loc[ll]/np.sqrt(ll+3))
        cg_cd_est_upper.loc[ll] = cg_cd_running_mean_est.loc[ll] + 1.96*(std_cg_cd_est.loc[ll]/np.sqrt(ll+3))

    x_cg_cd_est = []
    for m in range(len(cg_cd_running_mean_est)):
        x_cg_cd_est.append(m)

    if i == 2:
        labels = cg_cd_o_est.columns

    # TODO (27.04.22, 19:00 hier geehts weiter in der Funktion)
    ax[0, i].plot(x_sage, sage_running_mean, linewidth=0.7)
    for n in sage_lower.columns:
        ax[0, i].fill_between(x_sage, sage_lower[n], sage_upper[n], alpha=.1)

    ax[0, i].legend(loc='upper right', labels=labels)

    ax[1, i].plot(x_cg, cg_running_mean, linewidth=0.7)
    for n in cg_lower.columns:
        ax[1, i].fill_between(x_cg, cg_lower[n], cg_upper[n], alpha=.1)

    ax[2, i].plot(x_cg_cd, cg_cd_running_mean, linewidth=0.7)
    for n in cg_cd_lower.columns:
        ax[2, i].fill_between(x_cg_cd, cg_cd_lower[n], cg_cd_upper[n], alpha=.1)

    ax[3, i].plot(x_cg_est, cg_running_mean_est, linewidth=0.7)
    for n in cg_est_lower.columns:
        ax[3, i].fill_between(x_cg_est, cg_est_lower[n], cg_est_upper[n], alpha=.1)

    ax[4, i].plot(x_cg_cd_est, cg_cd_running_mean_est, linewidth=0.7)
    for n in cg_cd_est_lower.columns:
        ax[4, i].fill_between(x_cg_cd_est, cg_cd_est_lower[n], cg_cd_est_upper[n], alpha=.1)

    l1 = ax[i, 0].plot(x_sage, sage_running_mean, linewidth=0.7)
    for n in sage_lower.columns:
        ax[i, 0].fill_between(x_sage, sage_lower[n], sage_upper[n], alpha=.1)

ax[0, 0].set_ylabel('DAG$_s$')
ax[1, 0].set_ylabel('DAG$_{sm}$')

ax[0].set_xlabel('SAGE')
ax[1].set_xlabel('SAGE$_{cg}^{o}$')
ax[2].set_xlabel('SAGE$_{cg}$')
ax[3].set_xlabel('SAGE$_{cg}^{o,*}$')
ax[4].set_xlabel('SAGE$_{cg}^{*}$')

fig.subplots_adjust(left=0.3)
fig.legend([l1],  # The line objects
           labels=labels,  # The labels for each line
           loc="lower center",  # Position of legend
           bbox_to_anchor=(0.5, 0.0),  # Title for the legend
           fancybox=True, shadow=True, ncol=1, fontsize=8
           )

fig.text(0.5, 0, 'No. of Permutations', ha='center')
plt.savefig(f"visualization/sage/convergence_rf_t.png", dpi=600, bbox_inches='tight', transparent=True)


"""for 'transpose' grid:
 # ax[i, 0].legend(loc="upper right", labels=labels, handlelength=0.5, handleheight=0.5)
    ax[i, 0].legend(loc="upper right", labels=labels, prop={'size': 5})

    #ax[1].plot(x_cg, cg_running_mean, linewidth=0.7)
    #for n in cg_lower.columns:
    #    ax[1].fill_between(x_cg, cg_lower[n], cg_upper[n], alpha=.1)
#
    #ax[2].plot(x_cg_cd, cg_cd_running_mean, linewidth=0.7)
    #for n in cg_cd_lower.columns:
    #    ax[2].fill_between(x_cg_cd, cg_cd_lower[n], cg_cd_upper[n], alpha=.1)

    ax[i,1].plot(x_cg_est, cg_running_mean_est, linewidth=0.7)
    for n in cg_est_lower.columns:
        ax[i,1].fill_between(x_cg_est, cg_est_lower[n], cg_est_upper[n], alpha=.1)

    ax[i,2].plot(x_cg_cd_est, cg_cd_running_mean_est, linewidth=0.7)
    for n in cg_cd_est_lower.columns:
        ax[i,2].fill_between(x_cg_cd_est, cg_cd_est_lower[n], cg_cd_est_upper[n], alpha=.1)
        
ax[0, 0].set_ylabel('SAGE')
ax[1, 0].set_ylabel('SAGE$_{cg}^{o}$')
ax[2, 0].set_ylabel('SAGE$_{cg}$')
ax[3, 0].set_ylabel('SAGE$_{cg}^{o,*}$')
ax[4, 0].set_ylabel('SAGE$_{cg}^{*}$')

#fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

"""