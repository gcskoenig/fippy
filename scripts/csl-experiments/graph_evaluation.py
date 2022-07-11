"""File to evaluate estimated graphs wrt inferred d-separations by comparison to
    ground truth graph using GraphComparison class from comparison.py used in
    Causal Structure Learning for SAGE Estimation
    """
import networkx as nx
import pandas as pd
from utils import exact, approx, convert_amat

# file for comparison   # TODO (cl) More elegant way than try-except-statement?
try:
    graph_evaluation = pd.read_csv("results_py/graph_evaluation.csv")
except:
    col_names = ["Graph", "Target", "Method", "MC",
                 "True total", "False total", "D-sep share", "TP", "TN", "FP", "FN", "TP rate",
                 "TN rate", "FP rate", "FN rate", "Precision", "Recall", "F1"]
    graph_evaluation = pd.DataFrame(columns=col_names)

# to access the target nodes in graph loop  # TODO: adept targets after they have been drawn in experiment file
targets = {"alarm": "PULMEMBOLUS", "asia": "dysp", "hepar": "RHepatitis", "sachs": "Erk",
           "dag_s": "1",  "dag_sm": "1",  "dag_m": "1",  "dag_l": "1"}

# adapt for final experiments
discrete_graphs = ["asia", "sachs", "alarm", "hepar"]
# cont_graphs = ["dag_s", "dag_sm", "dag_m", "dag_l"]
cont_graphs = []
sample_sizes = ["10", "100", "1000"]
# discrete_algs = ["hc", "tabu", "mmhc", "h2pc"]
discrete_algs = ["tabu"]
cont_algs = ["hc", "tabu"]


# compare graphs with less than 1M d-separations by evaluating exact inference
for graph in discrete_graphs:
    # ground truth graph
    true_amat = pd.read_csv(f"scripts/csl-experiments/data/true_amat/{graph}.csv")
    true_amat = convert_amat(true_amat)
    g_true = nx.DiGraph(true_amat)
    for method in discrete_algs:
        for n in sample_sizes:
            est_amat = pd.read_csv(f"scripts/csl-experiments/bnlearn/results/{method}/{graph}_{n}_obs.csv")
            est_amat = convert_amat(est_amat)
            g_est = nx.DiGraph(est_amat)
            if graph in ["asia", "sachs"]:
                tp, tn, fp, fn, d_separated_total, d_connected_total = exact(g_true, g_est, targets[f"{graph}"])
                mc = "n/a"
            elif graph in ["alarm", "hepar"]:
                mc = 1000000
                tp, tn, fp, fn, d_separated_total, d_connected_total = approx(g_true, g_est, targets[f"{graph}"],
                                                                              mc=mc, rand_state=42)
            else:
                print("graph is not defined properly")
                break

            dsep_share = d_separated_total / (d_separated_total + d_connected_total)
            if d_separated_total == 0:
                TP_rate = 0
                FN_rate = 0
            else:
                TP_rate = tp / d_separated_total
                FN_rate = fn / d_separated_total
            if d_connected_total == 0:
                TN_rate = 0
                FP_rate = 0
            else:
                TN_rate = tn / d_connected_total
                FP_rate = fp / d_connected_total
            # F1 score
            precision = tp / (tp + fp)
            recall = TP_rate
            F1 = (2 * precision * recall) / (precision + recall)
            content = [f"{graph}_{n}_obs", targets[f"{graph}"], method, mc, d_separated_total, d_connected_total,
                       dsep_share, tp, tn, fp, fn, TP_rate, TN_rate, FP_rate, FN_rate, precision, recall, F1]
            graph_evaluation.loc[len(graph_evaluation)] = content

for graph in cont_graphs:
    # ground truth graph
    true_amat = pd.read_csv(f"scripts/csl-experiments/data/true_amat/{graph}.csv")
    true_amat = convert_amat(true_amat, col_names=True)
    g_true = nx.DiGraph(true_amat)
    for method in ["hc", "tabu"]:
        for n in sample_sizes:
            est_amat = pd.read_csv(f"scripts/csl-experiments/bnlearn/results/{method}/{graph}_{n}_obs.csv")
            est_amat = convert_amat(est_amat, col_names=True)
            g_est = nx.DiGraph(est_amat)
            if graph in ["dag_s"]:
                tp, tn, fp, fn, d_separated_total, d_connected_total = exact(g_true, g_est, targets[f"{graph}"])
                mc = "n/a"
            elif graph in ["dag_sm", "dag_m", "dag_l"]:
                mc = 1000
                tp, tn, fp, fn, d_separated_total, d_connected_total = approx(g_true, g_est, targets[f"{graph}"],
                                                                              mc=mc, rand_state=42)
            else:
                print("graph is not defined properly")
                break
            dsep_share = d_separated_total / (d_separated_total + d_connected_total)
            if d_separated_total == 0:
                TP_rate = 0
                FN_rate = 0
            else:
                TP_rate = tp / d_separated_total
                FN_rate = fn / d_separated_total
            if d_connected_total == 0:
                TN_rate = 0
                FP_rate = 0
            else:
                TN_rate = tn / d_connected_total
                FP_rate = fp / d_connected_total
            # F1 score
            try:
                precision = tp / (tp + fp)
            except:
                precision = 0
            recall = TP_rate
            try:
                F1 = (2 * precision * recall) / (precision + recall)
            except:
                F1 = 0
            content = [f"{graph}_{n}_obs", targets[f"{graph}"], method, mc, d_separated_total, d_connected_total,
                       dsep_share, tp, tn, fp, fn, TP_rate, TN_rate, FP_rate, FN_rate, precision, recall, F1]
            graph_evaluation.loc[len(graph_evaluation)] = content

graph_evaluation.to_csv("scripts/csl-experiments/bnlearn/results/graph_evaluation.csv", index=False)
