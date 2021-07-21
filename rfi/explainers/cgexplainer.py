"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""

from rfi.explainers.explainer import Explainer
import numpy as np
import pandas as pd
import logging

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


class CGExplainer(Explainer):
    """Implements a number of feature importance algorithms

    Default samplers or loss function can be defined.

    Attributes:
        model: Model or predict function.
        fsoi: Features of interest. Columnnames.
        X_train: Training data for resampling. Pandas dataframe.
        sampler: default sampler.
        decorrelator: default decorrelator
        loss: default loss.
    """
    def __init__(self, model, fsoi, X_train, causal_graph, **kwargs):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        Explainer.__init__(model, fsoi, X_train, **kwargs)
        self.causal_graph = causal_graph

    def _check_valid_graph(self):
        # TODO make sanity checks
        raise NotImplementedError('Check not implemented yet.')

    def ai_via(self, J, C, K, X_eval, y_eval, D=None, **kwargs):
        # TODO d-separation test
        #  is J idp Y | C?
        #  if so, evaluate to zero?
        #  if not the case, call superclass method
        raise NotImplementedError('CSL of ai_via version no implemented yet.')
        super(CGExplainer, self).ai_via(J, C, K, X_eval, y_eval, D=D, **kwargs)