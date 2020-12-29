import numpy as np
from typing import List
from networkx import nx
import matplotlib.pyplot as plt


class DirectedAcyclicGraph:
    """
    Directed acyclic graph, used to define Structural Equation Model
    """

    def __init__(self, adjacency_matrix: np.array, var_names: List[str]):
        """
        Args:
            adjacency_matrix: Square adjacency matrix
            var_names: List of variable input_var_names
        """
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        assert adjacency_matrix.shape[0] == len(var_names)

        self.adjacency_matrix = adjacency_matrix.astype(int)
        self.DAG = nx.convert_matrix.from_numpy_matrix(self.adjacency_matrix, create_using=nx.DiGraph)
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.DAG)

        self.var_names = np.array(var_names, dtype=str)


    def plot_dag(self, ax=None):
        """
        Plots DAG with networkx tools
        Args:
            ax: Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots()
        labels_dict = {i: self.var_names[i] for i in range(len(self.DAG))}
        nx.draw_networkx(self.DAG, pos=nx.kamada_kawai_layout(self.DAG), ax=ax, labels=labels_dict, node_color='white',
                         arrowsize=15, edgecolors='b', node_size=800)

    @staticmethod
    def random_dag(n, p, seed=None):
        """
        Creates random Erdős-Rényi graph from G(size, p) sem
        (see https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)

        Args:
            seed: Random mc_seed
            n: Number of nodes
            p: Probability of creating an edge

        Returns: DirectedAcyclicGraph instance

        """
        G = nx.gnp_random_graph(n, p, seed, directed=True)
        G.remove_edges_from([(u, v) for (u, v) in G.edges() if u > v])
        adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(G).todense().astype(int)
        var_names = [f'x{i}' for i in range(n)]
        return DirectedAcyclicGraph(adjacency_matrix, var_names)
