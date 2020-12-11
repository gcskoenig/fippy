import numpy as np
from typing import List

class DirectedAssyclicGraph:

    def __init__(self, adjacency_matrix: np.array, var_names: List[str]):
        self.adjacency_matrix = adjacency_matrix
        self.var_names = var_names
