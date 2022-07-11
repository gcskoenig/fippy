"""Confusion matrices of d-separation inference by causal structure learning algorithms"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import sys
import os
import inspect


# set plt font to standard Latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from functions import create_folder

create_folder("visualization/confusion/")

# import graph evaluation file
df = pd.read_csv("results_py/graph_evaluation.csv")
df = df[df['method'] == "tabu"]

# sample sizes
n_discrete = 1e+06
n_cont = 10000

# get rows for the graphs learned with n=10,000
dag_s_df = df[df['graph'] == f"dag_s_{n_cont}_obs"]
dag_m_df = df[df['graph'] == f"dag_m_{n_cont}_obs"]
dag_l_df = df[df['graph'] == f"dag_l_{n_cont}_obs"]
dag_sm_df = df[df['graph'] == f"dag_sm_{n_cont}_obs"]

asia = df[df['graph'] == f"asia_{n_discrete}_obs"]
sachs = df[df['graph'] == f"sachs_{n_discrete}_obs"]
alarm = df[df['graph'] == f"alarm_{n_discrete}_obs"]
hepar = df[df['graph'] == f"hepar_{n_discrete}_obs"]

# create vectors with true and predicted labels for every graph
# true labels
asia_true = []
for i in range(int(asia['TP'].iloc[0]) + int(asia['FN'].iloc[0])):
    asia_true.append(1)
for j in range(int(asia['TN'].iloc[0]) + int(asia['FP'].iloc[0])):
    asia_true.append(0)

# predictions
asia_pred = []
for i in range(int(asia['TP'].iloc[0])):
    asia_pred.append(1)
for j in range(int(asia['FN'].iloc[0])):
    asia_pred.append(0)
for k in range(int(asia['TN'].iloc[0])):
    asia_pred.append(0)
for m in range(int(asia['FP'].iloc[0])):
    asia_pred.append(1)

# true labels
sachs_true = []
for i in range(int(sachs['TP'].iloc[0]) + int(sachs['FN'].iloc[0])):
    sachs_true.append(1)
for j in range(int(sachs['TN'].iloc[0]) + int(sachs['FP'].iloc[0])):
    sachs_true.append(0)

# predictions
sachs_pred = []
for i in range(int(sachs['TP'].iloc[0])):
    sachs_pred.append(1)
for j in range(int(sachs['FN'].iloc[0])):
    sachs_pred.append(0)
for k in range(int(sachs['TN'].iloc[0])):
    sachs_pred.append(0)
for m in range(int(sachs['FP'].iloc[0])):
    sachs_pred.append(1)

# true labels
alarm_true = []
for i in range(int(alarm['TP'].iloc[0]) + int(alarm['FN'].iloc[0])):
    alarm_true.append(1)
for j in range(int(alarm['TN'].iloc[0]) + int(alarm['FP'].iloc[0])):
    alarm_true.append(0)

# predictions
alarm_pred = []
for i in range(int(alarm['TP'].iloc[0])):
    alarm_pred.append(1)
for j in range(int(alarm['FN'].iloc[0])):
    alarm_pred.append(0)
for k in range(int(alarm['TN'].iloc[0])):
    alarm_pred.append(0)
for m in range(int(alarm['FP'].iloc[0])):
    alarm_pred.append(1)

# true labels
hepar_true = []
for i in range(int(hepar['TP'].iloc[0]) + int(hepar['FN'].iloc[0])):
    hepar_true.append(1)
for j in range(int(hepar['TN'].iloc[0]) + int(hepar['FP'].iloc[0])):
    hepar_true.append(0)

# predictions
hepar_pred = []
for i in range(int(hepar['TP'].iloc[0])):
    hepar_pred.append(1)
for j in range(int(hepar['FN'].iloc[0])):
    hepar_pred.append(0)
for k in range(int(hepar['TN'].iloc[0])):
    hepar_pred.append(0)
for m in range(int(hepar['FP'].iloc[0])):
    hepar_pred.append(1)


cm_asia = confusion_matrix(asia_true, asia_pred, labels=[0, 1], normalize='all')
cm_sachs = confusion_matrix(sachs_true, sachs_pred, labels=[0, 1], normalize='all')
cm_alarm = confusion_matrix(alarm_true, alarm_pred, labels=[0, 1], normalize='all')
cm_hepar = confusion_matrix(hepar_true, hepar_pred, labels=[0, 1], normalize='all')

fig, ax = plt.subplots(1, 4, figsize=(8, 2.2))
fig.tight_layout()

disp_asia = ConfusionMatrixDisplay(confusion_matrix=cm_asia,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_sachs = ConfusionMatrixDisplay(confusion_matrix=cm_sachs,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_alarm = ConfusionMatrixDisplay(confusion_matrix=cm_alarm,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_hepar = ConfusionMatrixDisplay(confusion_matrix=cm_hepar,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_asia.plot(ax=ax[0], cmap="Blues")
disp_asia.ax_.set_title(r"Asia")
disp_asia.im_.colorbar.remove()
disp_asia.ax_.set_xlabel('')
disp_asia.ax_.set_ylabel('')

disp_sachs.plot(ax=ax[1], cmap="Blues")
disp_sachs.ax_.set_title(r"Sachs")
disp_sachs.im_.colorbar.remove()
disp_sachs.ax_.set_xlabel('')
disp_sachs.ax_.set_ylabel('')

disp_alarm.plot(ax=ax[2], cmap="Blues")
disp_alarm.ax_.set_title(r"Alarm")
disp_alarm.im_.colorbar.remove()
disp_alarm.ax_.set_xlabel('')
disp_alarm.ax_.set_ylabel('')

disp_hepar.plot(ax=ax[3], cmap="Blues")
disp_hepar.ax_.set_title(r"Hepar II")
disp_hepar.im_.colorbar.remove()
disp_hepar.ax_.set_xlabel('')
disp_hepar.ax_.set_ylabel('')

fig.text(0.0, 0.5, 'True label', va='center', rotation='vertical')
fig.text(0.5, 0.01, 'Predicted label', ha='center')

# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.7])
# fig.colorbar(disp_asia.im_, cax=cbar_ax)

plt.savefig(f"visualization/confusion/confusion_discrete_{n_discrete}_obs.png", dpi=400, transparent=True)
plt.clf()

# true labels
dag_s_true = []
for i in range(int(dag_s_df['TP'].iloc[0]) + int(dag_s_df['FN'].iloc[0])):
    dag_s_true.append(1)
for j in range(int(dag_s_df['TN'].iloc[0]) + int(dag_s_df['FP'].iloc[0])):
    dag_s_true.append(0)

# predictions
dag_s_pred = []
for i in range(int(dag_s_df['TP'].iloc[0])):
    dag_s_pred.append(1)
for j in range(int(dag_s_df['FN'].iloc[0])):
    dag_s_pred.append(0)
for k in range(int(dag_s_df['TN'].iloc[0])):
    dag_s_pred.append(0)
for m in range(int(dag_s_df['FP'].iloc[0])):
    dag_s_pred.append(1)


# true labels
dag_sm_true = []
for i in range(int(dag_sm_df['TP'].iloc[0]) + int(dag_sm_df['FN'].iloc[0])):
    dag_sm_true.append(1)
for j in range(int(dag_sm_df['TN'].iloc[0]) + int(dag_sm_df['FP'].iloc[0])):
    dag_sm_true.append(0)

# predictions
dag_sm_pred = []
for i in range(int(dag_sm_df['TP'].iloc[0])):
    dag_sm_pred.append(1)
for j in range(int(dag_sm_df['FN'].iloc[0])):
    dag_sm_pred.append(0)
for k in range(int(dag_sm_df['TN'].iloc[0])):
    dag_sm_pred.append(0)
for m in range(int(dag_sm_df['FP'].iloc[0])):
    dag_sm_pred.append(1)

# true labels
dag_m_true = []
for i in range(int(dag_m_df['TP'].iloc[0]) + int(dag_m_df['FN'].iloc[0])):
    dag_m_true.append(1)
for j in range(int(dag_m_df['TN'].iloc[0]) + int(dag_m_df['FP'].iloc[0])):
    dag_m_true.append(0)

# predictions
dag_m_pred = []
for i in range(int(dag_m_df['TP'].iloc[0])):
    dag_m_pred.append(1)
for j in range(int(dag_m_df['FN'].iloc[0])):
    dag_m_pred.append(0)
for k in range(int(dag_m_df['TN'].iloc[0])):
    dag_m_pred.append(0)
for m in range(int(dag_m_df['FP'].iloc[0])):
    dag_m_pred.append(1)

# true labels
dag_l_true = []
for i in range(int(dag_l_df['TP'].iloc[0]) + int(dag_l_df['FN'].iloc[0])):
    dag_l_true.append(1)
for j in range(int(dag_l_df['TN'].iloc[0]) + int(dag_l_df['FP'].iloc[0])):
    dag_l_true.append(0)

# predictions
dag_l_pred = []
for i in range(int(dag_l_df['TP'].iloc[0])):
    dag_l_pred.append(1)
for j in range(int(dag_l_df['FN'].iloc[0])):
    dag_l_pred.append(0)
for k in range(int(dag_l_df['TN'].iloc[0])):
    dag_l_pred.append(0)
for m in range(int(dag_l_df['FP'].iloc[0])):
    dag_l_pred.append(1)

cm_s = confusion_matrix(dag_s_true, dag_s_pred, labels=[0, 1], normalize='all')
cm_sm = confusion_matrix(dag_sm_true, dag_sm_pred, labels=[0, 1], normalize='all')
cm_m = confusion_matrix(dag_m_true, dag_m_pred, labels=[0, 1], normalize='all')
cm_l = confusion_matrix(dag_l_true, dag_l_pred, labels=[0, 1], normalize='all')

fig1, ax1 = plt.subplots(1, 4, figsize=(8, 2.2))
fig1.tight_layout()

disp_s = ConfusionMatrixDisplay(confusion_matrix=cm_s,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_sm = ConfusionMatrixDisplay(confusion_matrix=cm_sm,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_m = ConfusionMatrixDisplay(confusion_matrix=cm_m,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_l = ConfusionMatrixDisplay(confusion_matrix=cm_l,
            display_labels=[r"$\not\perp_{\mathcal{G}}$",r"$\perp_{\mathcal{G}}$"])

disp_s.plot(ax=ax1[0], cmap="Blues")
disp_s.ax_.set_title(r"DAG$_s$")
disp_s.im_.colorbar.remove()
disp_s.ax_.set_xlabel('')
disp_s.ax_.set_ylabel('')

disp_sm.plot(ax=ax1[1], cmap="Blues")
disp_sm.ax_.set_title(r"DAG$_{sm}$")
disp_sm.im_.colorbar.remove()
disp_sm.ax_.set_xlabel('')
disp_sm.ax_.set_ylabel('')

disp_m.plot(ax=ax1[2], cmap="Blues")
disp_m.ax_.set_title(r"DAG$_m$")
disp_m.im_.colorbar.remove()
disp_m.ax_.set_xlabel('')
disp_m.ax_.set_ylabel('')

disp_l.plot(ax=ax1[3], cmap="Blues")
disp_l.ax_.set_title(r"DAG$_l$")
disp_l.im_.colorbar.remove()
disp_l.ax_.set_xlabel('')
disp_l.ax_.set_ylabel('')

fig1.text(0.0, 0.5, 'True label', va='center', rotation='vertical')
fig1.text(0.5, 0.01, 'Predicted label', ha='center')

# fig1.subplots_adjust(right=0.8)
# cbar_ax1 = fig1.add_axes([0.83, 0.18, 0.02, 0.71])
# fig1.colorbar(disp_l.im_, cax=cbar_ax1)

plt.savefig(f"visualization/confusion/confusion_cont_{n_cont}_obs.png", dpi=400, transparent=True)
