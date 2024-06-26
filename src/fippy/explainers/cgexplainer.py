"""Graph explainer implements explanation methods that rely
on knowledge of a (causal) graph.

# TODO: fix docstring here
Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""
from fippy.explainers.explainer import Explainer
import numpy as np
import pandas as pd
import logging
import networkx as nx
import fippy.explanation as explanation
from fippy import utils

idx = pd.IndexSlice
logger = logging.getLogger(__name__)

# TODO: rename to GraphExplainer?
class CGExplainer(Explainer):
    """Implements a number of feature importance algorithms

    # TODO: fix docstring here
    Default samplers or loss function can be defined.

    Attributes:
        adj_mat: Adjacency matrix of DAG used for d-separation/independence test
    """
    def __init__(self, predict, X_train, adj_mat, **kwargs):
        """Additionally adds adjacency matrix of graph to Explainer"""
        # TODO align signature with Explainer
        Explainer.__init__(self, predict, X_train, **kwargs)
        self.adj_mat = adj_mat
        self._check_valid_graph(self.adj_mat)
        self.g = nx.DiGraph(adj_mat)

    @staticmethod
    def _check_valid_graph(df):   # TODO(cl) better way to check? More flexible data type possible?
        # TODO check if column names of adjacency matrix == column names of data (X.columns + y.name)
        if isinstance(df, pd.DataFrame):
            if df.index.to_list() == df.columns.to_list():
                return True
            else:
                var_names = df.columns.to_list()
                df.set_axis(var_names, axis=0)
                return True
        else:
            raise TypeError('Adjacency matrix must be pandas.DataFrame.')

    def ai_via(self, J, C, K, X_eval, y_eval, nr_runs=10, **kwargs):
        d_sep = nx.d_separated(self.g, set(J), set(y_eval.name), set(C))
        if d_sep:
            desc = 'AR({} | {} -> {})'.format(J, C, K)
            nr_obs = X_eval.shape[0]
            index = utils.create_multiindex(['sample', 'i'],
                                            [np.arange(nr_runs),
                                             np.arange(nr_obs)])
            scores = pd.DataFrame([], index=index)
            for kk in np.arange(0, nr_runs, 1):
                scores.loc[(kk, slice(None)), 'score'] = 0
            ex_name = desc
            result = explanation.Explanation(
                self.fsoi, scores,
                ex_name=ex_name)
            return result
        else:
            result = super(CGExplainer, self).ai_via(J, C, K, X_eval, y_eval, nr_runs=nr_runs, **kwargs)
            return result
