"Visualization of runtime data"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import inspect


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from functions import create_folder

create_folder("visualization/runtime/")

# now for cont
hc_cont = pd.read_csv("bnlearn/results/hc/runtime_data_cont.csv")
tabu_cont = pd.read_csv("bnlearn/results/tabu/runtime_data_cont.csv")
hc_cont_sm = pd.read_csv("bnlearn/results/hc/runtime_data_cont_sm.csv")
tabu_cont_sm = pd.read_csv("bnlearn/results/tabu/runtime_data_cont_sm.csv")

#  x data (sample size)
x_ticks = [r"1k", r"10k", r"100k", r"1000k", r"2000k"]
x1 = [0.9, 1.9, 2.9, 3.9, 4.9]
x2 = [1.1, 2.1, 3.1, 4.1, 5.1]

# y data (runtime per graph)
y1_s = hc_cont[hc_cont['Graph'] == "dag_s"]["Runtime in s"]
y1_sm = hc_cont[hc_cont['Graph'] == "dag_sm"]["Runtime in s"]
y1_m = hc_cont[hc_cont['Graph'] == "dag_m"]["Runtime in s"]
y1_l = hc_cont[hc_cont['Graph'] == "dag_l"]["Runtime in s"]
y2_s = tabu_cont[tabu_cont['Graph'] == "dag_s"]["Runtime in s"]
y2_sm = tabu_cont[tabu_cont['Graph'] == "dag_sm"]["Runtime in s"]
y2_m = tabu_cont[tabu_cont['Graph'] == "dag_m"]["Runtime in s"]
y2_l = tabu_cont[tabu_cont['Graph'] == "dag_l"]["Runtime in s"]

fig, axes = plt.subplots(1, 4, figsize=(9.5, 3.3))
fig.tight_layout()

# DAG_s
axes[0].set_title(r'DAG$_{s}$')
# axes[0].set_xlabel('Sample Size')
axes[0].set_ylabel('Runtime in s')
b1 = axes[0].bar(x1, y1_s, width=0.2, color='b', align='center')
b2 = axes[0].bar(x2, y2_s, width=0.2, color='g', align='center')
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_xticklabels(x_ticks, fontsize=10)

# DAG_sm

axes[1].set_title(r'DAG$_{s}$')
# axes[0].set_xlabel('Sample Size')
# axes[1].set_ylabel('Runtime in s')
axes[1].bar(x1, y1_sm, width=0.2, color='b', align='center')
axes[1].bar(x2, y2_sm, width=0.2, color='g', align='center')
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].set_xticklabels(x_ticks, fontsize=10)

# DAG_m
axes[2].set_title(r'DAG$_{m}$')
# axes[2].set_xlabel('Sample Size')
# axes[2].set_ylabel('Runtime in s')
axes[2].bar(x1, y1_m, width=0.2, color='b', align='center')
axes[2].bar(x2, y2_m, width=0.2, color='g', align='center')
axes[2].set_xticks([1, 2, 3, 4, 5])
axes[2].set_xticklabels(x_ticks, fontsize=10)

# DAG_l
axes[3].set_title(r'DAG$_{l}$')
# axes[3].set_xlabel('Sample Size')
# axes[3].set_ylabel('Runtime in s')
axes[3].bar(x1, y1_l, width=0.2, color='b', align='center')
axes[3].bar(x2, y2_l, width=0.2, color='g', align='center')
axes[3].set_xticks([1, 2, 3, 4, 5])
axes[3].set_xticklabels(x_ticks, fontsize=10)

legend_labels = [r"HC", r"TABU"]
fig.legend([b1, b2],     # The line objects
           labels=legend_labels,   # The labels for each line
           loc="lower center",   # Position of legend
           bbox_to_anchor=(0.5, 0.0),
           title="Algorithm",  # Title for the legend
           fancybox=True, shadow=True, ncol=2, fontsize=8
           )
plt.subplots_adjust(bottom=0.32)
fig.text(0.5, 0.18, 'Sample Size', ha='center')
plt.savefig("visualization/runtime/cont_runtime.png", dpi=400, transparent=True)
plt.clf()

# now for discrete
hc_d = pd.read_csv("bnlearn/results/hc/runtime_data_discrete.csv")
tabu_d = pd.read_csv("bnlearn/results/tabu/runtime_data_discrete.csv")
mmhc_d = pd.read_csv("bnlearn/results/mmhc/runtime_data_discrete.csv")
h2pc_d = pd.read_csv("bnlearn/results/h2pc/runtime_data_discrete.csv")

#  x data (sample size)
x_ticks = [r"1k", r"10k", r"100k", r"1000k", r"2000k"]
x1 = [0.7, 1.7, 2.7, 3.7, 4.7]
x2 = [0.9, 1.9, 2.9, 3.9, 4.9]
x3 = [1.1, 2.1, 3.1, 4.1, 5.1]
x4 = [1.3, 2.3, 3.3, 4.3, 5.3]

# y data (runtime per graph)
y1_asia = hc_d[hc_d['Graph'] == "asia"]["Runtime in s"]
y1_sachs = hc_d[hc_d['Graph'] == "sachs"]["Runtime in s"]
y1_alarm = hc_d[hc_d['Graph'] == "alarm"]["Runtime in s"]
y1_hepar = hc_d[hc_d['Graph'] == "hepar"]["Runtime in s"]

y2_asia = tabu_d[tabu_d['Graph'] == "asia"]["Runtime in s"]
y2_sachs = tabu_d[tabu_d['Graph'] == "sachs"]["Runtime in s"]
y2_alarm = tabu_d[tabu_d['Graph'] == "alarm"]["Runtime in s"]
y2_hepar = tabu_d[tabu_d['Graph'] == "hepar"]["Runtime in s"]

y3_asia = mmhc_d[mmhc_d['Graph'] == "asia"]["Runtime in s"]
y3_sachs = mmhc_d[mmhc_d['Graph'] == "sachs"]["Runtime in s"]
y3_alarm = mmhc_d[mmhc_d['Graph'] == "alarm"]["Runtime in s"]
y3_hepar = mmhc_d[mmhc_d['Graph'] == "hepar"]["Runtime in s"]

y4_asia = h2pc_d[h2pc_d['Graph'] == "asia"]["Runtime in s"]
y4_sachs = h2pc_d[h2pc_d['Graph'] == "sachs"]["Runtime in s"]
y4_alarm = h2pc_d[h2pc_d['Graph'] == "alarm"]["Runtime in s"]
y4_hepar = h2pc_d[h2pc_d['Graph'] == "hepar"]["Runtime in s"]

fig, axes = plt.subplots(2, 2, figsize=(7, 4))
fig.tight_layout(pad=1.7, h_pad=3)

axes[0, 0].set_title(r'Asia')
#axes[0, 0].set_xlabel('Sample Size')
#axes[0, 0].set_ylabel('Runtime in s')
b1 = axes[0, 0].bar(x1, y1_asia, width=0.2, color='b', align='center')
b2 = axes[0, 0].bar(x2, y2_asia, width=0.2, color='g', align='center')
b3 = axes[0, 0].bar(x3, y3_asia, width=0.2, color='c', align='center')
b4 = axes[0, 0].bar(x4, y4_asia, width=0.2, color='m', align='center')
axes[0, 0].set_xticks([1, 2, 3, 4, 5])
axes[0, 0].set_xticklabels(x_ticks, fontsize=10)

axes[0, 1].set_title(r'Sachs')
#axes[0, 1].set_xlabel('Sample Size')
#axes[0, 1].set_ylabel('Runtime in s')
axes[0, 1].bar(x1, y1_sachs, width=0.2, color='b', align='center')
axes[0, 1].bar(x2, y2_sachs, width=0.2, color='g', align='center')
axes[0, 1].bar(x3, y3_sachs, width=0.2, color='c', align='center')
axes[0, 1].bar(x4, y4_sachs, width=0.2, color='m', align='center')
axes[0, 1].set_xticks([1, 2, 3, 4, 5])
axes[0, 1].set_xticklabels(x_ticks, fontsize=10)

axes[1, 0].set_title(r'Alarm')
#axes[1, 0].set_xlabel('Sample Size')
#axes[1, 0].set_ylabel('Runtime in s')
axes[1, 0].bar(x1, y1_alarm, width=0.2, color='b', align='center')
axes[1, 0].bar(x2, y2_alarm, width=0.2, color='g', align='center')
axes[1, 0].bar(x3, y3_alarm, width=0.2, color='c', align='center')
axes[1, 0].bar(x4, y4_alarm, width=0.2, color='m', align='center')
axes[1, 0].set_xticks([1, 2, 3, 4, 5])
axes[1, 0].set_xticklabels(x_ticks, fontsize=10)

axes[1, 1].set_title(r'Hepar II')
#axes[1, 1].set_xlabel('Sample Size')
#axes[1, 1].set_ylabel('Runtime in s')
axes[1, 1].bar(x1, y1_hepar, width=0.2, color='b', align='center')
axes[1, 1].bar(x2, y2_hepar, width=0.2, color='g', align='center')
axes[1, 1].bar(x3, y3_hepar, width=0.2, color='c', align='center')
axes[1, 1].bar(x4, y4_hepar, width=0.2, color='m', align='center')
axes[1, 1].set_xticks([1, 2, 3, 4, 5])
axes[1, 1].set_xticklabels(x_ticks, fontsize=10)


legend_labels = [r"HC", r"TABU", r"MMHC", r"H2PC"]
fig.legend([b1, b2, b3, b4],     # The line objects
           labels=legend_labels,   # The labels for each line
           loc="lower center",   # Position of legend
           bbox_to_anchor=(0.5, 0.0),
           title="Algorithm",  # Title for the legend
           fancybox=True, shadow=True, ncol=4, fontsize=8
           )
plt.subplots_adjust(bottom=0.25)
fig.text(0.5, 0.15, 'Sample Size', ha='center')
fig.text(0.01, 0.5, 'Runtime in s', ha='center', rotation='vertical')

plt.savefig("visualization/runtime/discrete_runtime.png", dpi=400, transparent=True)
plt.clf()
