"""Visualization of graph comparison (discrete graphs), f1 score"""
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
y3 = df[df['method'] == "mmhc"]   # mmhc
y4 = df[df['method'] == "h2pc"]   # h2pc

y1_asia = y1[y1['d'] == 8]["F1"]
y2_asia = y2[y2['d'] == 8]["F1"]
y3_asia = y3[y3['d'] == 8]["F1"]
y4_asia = y4[y4['d'] == 8]["F1"]


y1_sachs = y1[y1['d'] == 11]["F1"]
y2_sachs = y2[y2['d'] == 11]["F1"]
y3_sachs = y3[y3['d'] == 11]["F1"]
y4_sachs = y4[y4['d'] == 11]["F1"]

y1_hepar = y1[y1['d'] == 70]["F1"]
y2_hepar = y2[y2['d'] == 70]["F1"]
y3_hepar = y3[y3['d'] == 70]["F1"]
y4_hepar = y4[y4['d'] == 70]["F1"]

y1_alarm = y1[y1['d'] == 37]["F1"]
y2_alarm = y2[y2['d'] == 37]["F1"]
y3_alarm = y3[y3['d'] == 37]["F1"]
y4_alarm = y4[y4['d'] == 37]["F1"]

mark = [r"1k", r"10k", r"100k", r"1000k", r"2000k"]
fig, axes = plt.subplots(2, 2, figsize=(7, 4))
# fig.suptitle(r"F1 Score of Estimated D-separations - Discrete Graphs", fontsize=12)
fig.tight_layout(pad=2.1, h_pad=4)

axes[0, 0].set_title(r'Asia')
# axes[0, 0].set_xlabel('Sample Size')
# axes[0, 0].set_ylabel('F1 Score')

axes[0, 0].scatter(x, y1_asia, color='b', s=0.4)
axes[0, 0].scatter(x, y2_asia, color='g', s=0.4)
axes[0, 0].scatter(x, y3_asia, color='c', s=0.4)
axes[0, 0].scatter(x, y4_asia, color='m', s=0.4)
l1 = axes[0, 0].plot(x, y1_asia, color='b', linestyle='-', markevery=mark, linewidth=0.7)
l2 = axes[0, 0].plot(x, y2_asia, color='g', linestyle=':', markevery=mark, linewidth=0.7)
l3 = axes[0, 0].plot(x, y3_asia, color='c', linestyle='--', markevery=mark, linewidth=0.7)
l4 = axes[0, 0].plot(x, y4_asia, color='m', linestyle='-.', markevery=mark, linewidth=0.7)

axes[0, 1].set_title(r'Sachs')
# axes[0, 1].set_xlabel('Sample Size')
# axes[0, 1].set_ylabel('F1 Score')

axes[0, 1].scatter(x, y1_sachs, color='b', s=0.4)
axes[0, 1].scatter(x, y2_sachs, color='g', s=0.4)
axes[0, 1].scatter(x, y3_sachs, color='c', s=0.4)
axes[0, 1].scatter(x, y4_sachs, color='m', s=0.4)
axes[0, 1].plot(x, y1_sachs, color='b', linestyle='-', markevery=mark, linewidth=0.7)
axes[0, 1].plot(x, y2_sachs, color='g', linestyle=':', markevery=mark, linewidth=0.7)
axes[0, 1].plot(x, y3_sachs, color='c', linestyle='--', markevery=mark, linewidth=0.7)
axes[0, 1].plot(x, y4_sachs, color='m', linestyle='-.', markevery=mark, linewidth=0.7)

axes[1, 0].set_title(r'Alarm')
# axes[1, 0].set_xlabel('Sample Size')
# axes[1, 0].set_ylabel('F1 Score')

axes[1, 0].scatter(x, y1_alarm, color='b', s=0.4)
axes[1, 0].scatter(x, y2_alarm, color='g', s=0.4)
axes[1, 0].scatter(x, y3_alarm, color='c', s=0.4)
axes[1, 0].scatter(x, y4_alarm, color='m', s=0.4)
axes[1, 0].plot(x, y1_alarm, color='b', linestyle='-', markevery=mark, linewidth=0.7)
axes[1, 0].plot(x, y2_alarm, color='g', linestyle=':', markevery=mark, linewidth=0.7)
axes[1, 0].plot(x, y3_alarm, color='c', linestyle='--', markevery=mark, linewidth=0.7)
axes[1, 0].plot(x, y4_alarm, color='m', linestyle='-.', markevery=mark, linewidth=0.7)

axes[1, 1].set_title(r'Hepar II')
# axes[1, 1].set_xlabel('Sample Size')
# axes[1, 1].set_ylabel('F1 Score')

axes[1, 1].scatter(x, y1_hepar, color='b', s=0.4)
axes[1, 1].scatter(x, y2_hepar, color='g', s=0.4)
axes[1, 1].scatter(x, y3_hepar, color='c', s=0.4)
axes[1, 1].scatter(x, y4_hepar, color='m', s=0.4)
axes[1, 1].plot(x, y1_hepar, color='b', linestyle='-', markevery=mark, linewidth=0.7)
axes[1, 1].plot(x, y2_hepar, color='g', linestyle=':', markevery=mark, linewidth=0.7)
axes[1, 1].plot(x, y3_hepar, color='c', linestyle='--', markevery=mark, linewidth=0.7)
axes[1, 1].plot(x, y4_hepar, color='m', linestyle='-.', markevery=mark, linewidth=0.7)

line_labels = ["HC", "TABU", "MMHC", "H2PC"]


fig.legend([l1, l2, l3, l4],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="lower center",   # Position of legend
           bbox_to_anchor=(0.5, 0.0),
           title="Algorithm",  # Title for the legend
           fancybox=True, shadow=True, ncol=4, fontsize=8
           )
plt.subplots_adjust(bottom=0.25)

fig.text(0.01, 0.6, 'F1 Score', va='center', rotation='vertical')
fig.text(0.5, 0.15, 'Sample Size', ha='center')

plt.savefig("visualization/dseps/discrete_dseps.png", dpi=400, transparent=True)
plt.clf()
