"""Explanations are the output of Explainers.

Aggregated or obser-wise wise results can be
accessed. Plotting functionality is available.
"""
import numpy as np

class Explanation():
    """Stores and provides access to results from Explainer.

    Aggregated as well as observation-wise results are stored.
    Plotting functionality is available.

    Attributes:
        fsoi: Features of interest.
        fsoi_names: FSOI names
        lss: losses on perturbed (# fsoi, # runs, # observations)
    """

    def __init__(self, fsoi, lss, fsoi_names=None):
        """Inits Explainer with model, mask and potentially sampler and loss"""
        self.fsoi = fsoi # TODO evaluate, do I need to make a copy?
        self.lss = lss # TODO evaluate, do I need to make a copy?
        self.fsoi_names = fsoi_names
        if self.fsoi_names is None:
            self.fsoi_names = fsoi

    def mean_rfis(self):
        """Computes Mean RFI over runs and observations

        Returns:
            A np.array with the relative feature importance values for
            features of interest.
        """
        return self.fsoi_names, np.mean(np.mean(self.lss, axis=2), axis=1)