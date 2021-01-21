"""Explanations are the output of Explainers.

Aggregated or obser-wise wise results can be
accessed. Plotting functionality is available.
"""
import numpy as np
import rfi.plots._barplot as _barplot
import logging


# TODO(gcsk): compute significance of the results using the multiple runs and marwin wright's procedure

class Explanation:
    """Stores and provides access to results from Explainer.

    Aggregated as well as observation-wise results are stored.
    Plotting functionality is available.

    Attributes:
        fsoi: Features of interest.
        lss: losses on perturbed (# fsoi, # runs, # observations, (#orderings))
        ex_name: Explanation description
        fsoi_names: feature of interest names
    """

    def __init__(self, fsoi, lss, fsoi_names, ex_name=None):
        """Inits Explanation with fsoi indices, fsoi names, """
        # TODO(gcsk): compress Explanation
        self.fsoi = fsoi # TODO evaluate, do I need to make a copy?
        self.lss = lss # TODO evaluate, do I need to make a copy?
        self.fsoi_names = fsoi_names
        if self.fsoi_names is None:
            self.fsoi_names = fsoi
        if ex_name is None:
            self.ex_name = 'Unknown'
        if len(self.lss.shape) == 3:
            self.lss = self.lss.reshape((self.lss.shape[0], self.lss.shape[1], 
                                         self.lss.shape[2], 1))
        if len(self.lss.shape) != 4:
            logging.debug('lss shape: {}'.format(lss.shape))
            raise RuntimeError('Lss has incorrect shape.')

    def fsoi_names(self):
        """Return RFI input_var_names for feature of interest

        Returns:
            A np.array with the feature input_var_names for the
            features of interest
        """
        return self.fsoi_names

    def fi_means(self):
        """Computes Mean RFI over all runs

        Returns:
            A np.array with the relative feature importance values for
            features of interest.
        """
        return np.mean(np.mean(np.mean(self.lss, axis=3), axis=2), axis=1)

    def fi_stds(self):
        """Computes std of RFI over all runs

        Returns:
            A np.array with the std of RFI values for the features of interest
        """
        return np.std(np.mean(np.mean(self.lss, axis=3), axis=2), axis=1)

    def barplot(self, ax=None):
        return _barplot.fi_hbarplot(self, ax=ax)
