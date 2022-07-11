"""Plot Confidence Intervals for the differences of the deltas between SAGE
and SAGE_CG if the is conditional independence and corresponding boxplot

One confidence interval per experiment, confidence over nr_orderings"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.style.use('ggplot')

# data, model
datasets = ["dag_s", "dag_s", "dag_sm", "dag_sm", "asia", "asia"]
models = ["lm", "rf", "lm", "rf", "mnb", "rf"]
est = ["true", "est"]
fig, ax = plt.subplots(1, 1, figsize=(1, 1))

for k in range(2):
    inference = est[k]
    for i in range(4):
        # load data
        data = datasets[i]
        model = models[i]
        if i < 4:
            sage_o = pd.read_csv(f"experiments_cg/results/continuous/true_amat/{data}/sage_o_{data}_{model}.csv")
            cg_o = pd.read_csv(f"experiments_cg/results/continuous/{inference}_amat/{data}/cg_o_{data}_{model}.csv")
            if data == "dag_s":
                sage_o = sage_o[0:500]
                cg_o = cg_o[0:500]
        else:
            sage_o = pd.read_csv(f"experiments_cg/results/discrete/true_amat/{data}/sage_o_{data}_{model}.csv")
            cg_o = pd.read_csv(f"experiments_cg/results/discrete/{inference}_amat/{data}/cg_o_{data}_{model}.csv")

        # Do this for every experiment and plot CIs (0.95)
        diffs = []
        for ii in range(len(cg_o)):
            for j in range(1, len(cg_o.iloc[0])):
                if cg_o.iloc[ii, j] == 0:
                    diffs.append(sage_o.iloc[ii, j])

        diffs = np.array(diffs)
        ci = 1.96 * np.std(diffs)/np.sqrt(len(diffs))
        mean = diffs.mean()
        lower = mean - ci
        upper = mean + ci

        # plot CIs
        plt.plot((lower, upper), (i, i), '-|', color='grey')
        plt.scatter(mean, i, color='black', marker="|")


    #plt.yticks([1, 2, 3, 4, 5, 6], ["DAG$_s$ (lm)", "DAG$_s$ (rf)", "DAG$_{sm}$ (lm)", "DAG$_{sm}$ (rf)", "Asia (nb)",
    #                                    "Asia (rf)"])
    plt.yticks([1, 2, 3, 4], ["DAG$_s$ (LM)", "DAG$_s$ (RF)", "DAG$_{sm}$ (LM)", "DAG$_{sm}$ (RF)"], fontsize=4)
    #plt.yticks([1, 2], ["Asia (NB)", "Asia (RF)"])
    #plt.subplots_adjust(left=0.1)


    plt.vlines(0, 0, 4, colors='red', linestyles='--', linewidth=0.4)
    plt.title(r"$\overline{\Delta_{sage}}$")
    plt.savefig(f"visualization/sage/deltas_{inference}_def.png", dpi=400, bbox_inches='tight', transparent=True)
    plt.clf()

    plt.boxplot(diffs)
    plt.xticks([1], ["$\Delta$"])
    plt.savefig(f"visualization/sage/deltas_bp_{inference}_def.png", dpi=400, bbox_inches='tight', transparent=True)
    plt.clf()
