import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import sys
import os
import inspect


#current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parent_dir = os.path.dirname(current_dir)
#sys.path.insert(0, parent_dir)
#
#
#from graphs_py.utils import create_folder
#
#
#create_folder("visualization/graphs/")


g1 = pd.read_csv(f"bnlearn/results/tabu/bike_amat.csv")

#col_names_str = []
#for k in range(len(g1.columns)):
#    col_names_str.append(str(k + 1))
#g1.columns = col_names_str
var_names = g1.columns.to_list()
g1 = g1.set_axis(var_names, axis=0)
# g2 = pd.read_csv(f"results_py/tabu/dag_s_est.csv")


G = nx.DiGraph(g1)
nx.draw(G, with_labels=True)
plt.draw()
plt.savefig(f"visualization/graphs/graph_bike.png", transparent=True)
plt.clf()

#G = nx.DiGraph(g2)
#nx.draw(G, with_labels=True)
#plt.draw()
#plt.savefig(f"graphs/graph_dag_s_est.png", transparent=True)
#plt.clf()
