"""Visualization of graph comparison (continuous graphs), f1 score"""
import pandas as pd
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

create_folder("visualization/dseps/")

df = pd.read_csv("results_py/graph_evaluation.csv")

x = [r"1k", r"10k", r"100k", r"1000k", r"2000k"]


y1 = df[df['method'] == "hc"]  # hc
y2 = df[df['method'] == "tabu"]   # tabu

y1_s = y1[y1['d'] == 10]["F1"]
y2_s = y2[y2['d'] == 10]["F1"]

y1_sm = y1[y1['d'] == 20]["F1"]
y2_sm = y2[y2['d'] == 20]["F1"]

y1_m = y1[y1['d'] == 50]["F1"]
y2_m = y2[y2['d'] == 50]["F1"]

y1_l = y1[y1['d'] == 100]["F1"]
y2_l = y2[y2['d'] == 100]["F1"]

mark = [r"1k", r"10k", r"100k", r"1000k", r"2000k"]
fig, axes = plt.subplots(2, 2, figsize=(7, 4))
fig.tight_layout(pad=2.1, h_pad=4)

axes[0, 0].set_title(r'DAG$_s$')
# axes[0, 0].set_xlabel('Sample Size')
# axes[0, 0].set_ylabel('F1 Score')

axes[0, 0].scatter(x, y1_s, color='b', s=0.4)
axes[0, 0].scatter(x, y2_s, color='g', s=0.4)
l1 = axes[0, 0].plot(x, y1_s, color='b', linestyle='-', markevery=mark, linewidth=0.7)
l2 = axes[0, 0].plot(x, y2_s, color='g', linestyle=':', markevery=mark, linewidth=0.7)

axes[0, 1].set_title(r'DAG$_{sm}$')
# axes[0, 1].set_xlabel('Sample Size')
# axes[0, 1].set_ylabel('F1 Score')

axes[0, 1].scatter(x, y1_sm, color='b', s=0.4)
axes[0, 1].scatter(x, y2_sm, color='g', s=0.4)

axes[0, 1].plot(x, y1_sm, color='b', linestyle='-', markevery=mark, linewidth=0.7)
axes[0, 1].plot(x, y2_sm, color='g', linestyle=':', markevery=mark, linewidth=0.7)


axes[1, 0].set_title(r'DAG$_{m}$')
# axes[1, 0].set_xlabel('Sample Size')
# axes[1, 0].set_ylabel('F1 Score')

axes[1, 0].scatter(x, y1_m, color='b', s=0.4)
axes[1, 0].scatter(x, y2_m, color='g', s=0.4)
axes[1, 0].plot(x, y1_m, color='b', linestyle='-', markevery=mark, linewidth=0.7)
axes[1, 0].plot(x, y2_m, color='g', linestyle=':', markevery=mark, linewidth=0.7)


axes[1, 1].set_title(r'DAG$_{l}$')
# axes[1, 1].set_xlabel('Sample Size')
# axes[1, 1].set_ylabel('F1 Score')

axes[1, 1].scatter(x, y1_l, color='b', s=0.4)
axes[1, 1].scatter(x, y2_l, color='g', s=0.4)

axes[1, 1].plot(x, y1_l, color='b', linestyle='-', markevery=mark, linewidth=0.7)
axes[1, 1].plot(x, y2_l, color='g', linestyle=':', markevery=mark, linewidth=0.7)


line_labels = ["HC", "TABU"]

fig.legend([l1, l2],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="lower center",   # Position of legend
           bbox_to_anchor=(0.5, 0.0),
           title="Algorithm",  # Title for the legend
           fancybox=True, shadow=True, ncol=4, fontsize=8
           )
plt.subplots_adjust(bottom=0.25)

fig.text(0.01, 0.6, 'F1 Score', va='center', rotation='vertical')
fig.text(0.5, 0.15, 'Sample Size', ha='center')

plt.savefig(f"visualization/dseps/cont_dseps.png", dpi=400, transparent=True)
plt.clf()


