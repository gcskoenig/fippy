"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""
# TODO(cl) Check how e.g. permutations are sampled -> seed for reproducibility
from rfi.explainers.explainer import Explainer
import numpy as np
import pandas as pd
import logging
import networkx as nx
from rfi.explanation import explanation
from rfi import utils
idx = pd.IndexSlice
logger = logging.getLogger(__name__)


class CGExplainer(Explainer):
    """Implements a number of feature importance algorithms

    Default samplers or loss function can be defined.

    Attributes:
        model: Model or predict function.
        fsoi: Features of interest. Columnnames.
        X_train: Training data for resampling. Pandas dataframe.
        adj_mat: Adjacency matrix of DAG used for d-separation/independence test
        sampler: default sampler.
        decorrelator: default decorrelator
        loss: default loss.
    """
    def __init__(self, model, fsoi, X_train, adj_mat, **kwargs):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        Explainer.__init__(self, model, fsoi, X_train, **kwargs)
        self.adj_mat = adj_mat
        self._check_valid_graph(self.adj_mat)
        self.g = nx.DiGraph(adj_mat)    # TODO(cl) what would be better style for sanity check + graph building?


    def _check_valid_graph(self, df):   # TODO(cl) better way to check? More flexible data type like in nx package?
        if isinstance(df, pd.DataFrame):
            if df.index.to_list() == df.columns.to_list():
                return True
            else:
                var_names = df.columns.to_list()
                df.set_axis(var_names, axis=0)
                return True
        else:
            raise NotImplementedError('Check not implemented yet.')

    # TODO(cl) Why is nr_runs (and other kwargs) not passed to ai_via below via **kwargs? cf explainer l. 749 & l. 483
    def ai_via(self, J, C, K, X_eval, y_eval, nr_runs=10, **kwargs):
        d_sep = nx.d_separated(self.g, set(J), {y_eval.name}, set(C))
        # d_sep = False
        print(d_sep)  # TODO(cl) delete print statements when done
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
            # TODO(cl) uncomment below if kwargs issue fixed
            # if return_perturbed:
            #     raise NotImplementedError('return_perturbed=True not implemented.')
            # else:
            #     return result
        # raise NotImplementedError('CSL of ai_via version no implemented yet.')
        # TODO why does **kwargs work here?
        else:
            result = super(CGExplainer, self).ai_via(J, C, K, X_eval, y_eval, **kwargs)
            return result
