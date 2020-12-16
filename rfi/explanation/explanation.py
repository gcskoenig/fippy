"""Explanations are the output of Explainers.

Aggregated or obser-wise wise results can be
accessed. Plotting functionality is available.
"""
import numpy as np
import rfi.plots._barplot as _barplot


class Explanation:
    """Stores and provides access to results from Explainer.

    Aggregated as well as observation-wise results are stored.
    Plotting functionality is available.

    Attributes:
        fsoi: Features of interest.
        lss: losses on perturbed (# fsoi, # runs, # observations)
        ex_name: Explanation description
        fsoi_names: FSOI input_var_names
    """

    def __init__(self, fsoi, lss, fs_names, ex_name=None):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        # TODO(gcsk): compress Explanation
        self.fsoi = fsoi # TODO evaluate, do I need to make a copy?
        self.lss = lss # TODO evaluate, do I need to make a copy?
        self.fs_names = fs_names
        if self.fs_names is None:
            self.fs_names = fsoi
        if ex_name is None:
            self.ex_name = 'Unknown'

    def rfi_names(self):
        """Return RFI input_var_names for feature of interest

        Returns:
            A np.array with the feature input_var_names for the
            features of interest
        """
        return self.fs_names[self.fsoi]

    def rfi_means(self):
        """Computes Mean RFI over all runs

        Returns:
            A np.array with the relative feature importance values for
            features of interest.
        """
        return np.mean(np.mean(self.lss, axis=2), axis=1)

    def rfi_stds(self):
        """Computes std of RFI over all runs

        Returns:
            A np.array with the std of RFI values for the features of interest
        """
        return np.std(np.mean(self.lss, axis=2), axis=1)

    def barplot(self, ax=None):
        return _barplot.rfi_hbarplot(self, ax=ax)
