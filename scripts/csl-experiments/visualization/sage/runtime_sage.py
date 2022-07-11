import pandas as pd
import matplotlib.pyplot as plt


# for latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# data, model
datasets = ["dag_s", "dag_sm", "asia"]
model = "lm"
model_discrete = "cnb"

plot_titles = [r"DAG$_{s}$", r"DAG$_{sm}$", r"Asia"]

fig, ax = plt.subplots(1, 3, figsize=(6, 3))
# fig, ax = plt.subplots()
plt.tight_layout(pad=1)

for i in range(3):
    data = datasets[i]
    # load data
    if i < 2:
        runtime = pd.read_csv(f"experiments_cg/results/continuous/true_amat/{data}/metadata_{data}_{model}.csv")
        runtime_cd = pd.read_csv(f"experiments_cg/results/continuous/true_amat/{data}/metadata2_{data}_{model}.csv")
    else:
        runtime = pd.read_csv(f"experiments_cg/results/discrete/true_amat/{data}/metadata_{data}_{model_discrete}.csv")
        runtime_cd = pd.read_csv(f"experiments_cg/results/discrete/true_amat/{data}/metadata2_{data}_{model_discrete}.csv")

    runtime = runtime[["runtime sage", "runtime cg", "runtime cg est"]].to_numpy()
    runtime_cd = runtime_cd[["runtime cg cd", "runtime cg cd est"]].to_numpy()

    if data == "dag_s":
        rt = []
        rt.append(runtime[0][0]/2)
        rt.append(runtime[0][1]/2)
        rt.append(runtime[0][2]/2)
        rt.append(runtime_cd[0][0]/2)
        rt.append(runtime_cd[0][1]/2)
    else:
        rt = []
        rt.append(runtime[0][0])
        rt.append(runtime[0][1])
        rt.append(runtime[0][2])
        rt.append(runtime_cd[0][0])
        rt.append(runtime_cd[0][1])

    # runtime g learning
    if data == "dag_s":
        g_rt = [0, 0, 0.46, 0, 0.46]
    # TODO: adapt the data here
    elif data == "dag_sm":
        g_rt = [0, 0, 0.754, 0, 0.754]
    elif data == "asia":
        g_rt = [0, 0, 0.067, 0, 0.067]

    labels = [r"SAGE", r"SAGE$_{cg}^{o}$", r"SAGE$_{cg}^{o,*}$", r"SAGE$_{cg}$",
              r"SAGE$_{cg}^{*}$"]

    ax[i].bar(labels, rt, width=0.4, label='SAGE', color="tab:blue")
    ax[i].bar(labels, g_rt, width=0.4, bottom=rt,
              label='TABU', color="orange")

    ax[i].set_title(plot_titles[i])
    if i == 1:
        ax[i].legend(loc='upper right')
    if i == 0:
        ax[i].set_ylabel('Runtime in s')
    for tick in ax[i].get_xticklabels():
        tick.set_rotation(60)

fig.subplots_adjust(bottom=0.28)

plt.savefig(f"visualization/runtime/runtime_sage_optimal.png", dpi=400, bbox_inches='tight', transparent=True)
