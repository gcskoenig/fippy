"""Explanations are the output of Explainers.

Aggregated or obser-wise wise results can be
accessed. Plotting functionality is available.
"""
import numpy as np
import rfi.plots._barplot as _barplot
import pandas as pd
import itertools


class Explanation:
    """Stores and provides access to results from Explainer.

    Aggregated as well as observation-wise results are stored.
    Plotting functionality is available.

    Attributes:
        fsoi: Features of interest.
        lss: losses on perturbed (nr_fsoi, nr_runs, nr_obs)
        ex_name: Explanation description
        fsoi_names: feature of interest names
    """

    def __init__(self, fsoi, lss, fsoi_names, ex_name=None):
        """Inits Explanation with fsoi indices, fsoi names, """
        self.fsoi = fsoi  # TODO evaluate, do I need to make a copy?
        self.lss = lss  # TODO evaluate, do I need to make a copy?
        self.fsoi_names = fsoi_names
        if self.fsoi_names is None:
            self.fsoi_names = fsoi
        if ex_name is None:
            self.ex_name = 'Unknown'

    def _check_shape(self):
        """Checks whether the array confirms the
        specified shape (3 dimensional).
        Cannot tell whether the ordering
        (nr_fsoi, nr_runs, nr_obs) is correct.
        """
        if len(self.lss.shape) != 3:
            raise RuntimeError('.lss has shape {self.lss.shape}.'
                               'Expected 3-dim.')

    def fi_means(self):
        """Computes Mean RFI over all runs

        Returns:
            A np.array with the relative feature importance value for
            features of interest.
        """
        return np.mean(np.mean(np.mean(self.lss, axis=3), axis=2), axis=1)


    def fi_vals(self, return_np=False):
        """ Computes the sample-wide RFI for each run

        Returns:
            pd.DataFrame with (#fsoi, #runs)
        """
        self._check_shape()
        arr = np.mean(self.lss, axis=(2))
        if return_np:
            return arr
        else:
            runs = range(arr.shape[1])
            tuples = list(itertools.product(self.fsoi_names, runs))
            index = pd.MultiIndex.from_tuples(tuples, names=['feature', 'run'])
            arr = arr.reshape(-1)
            df = pd.DataFrame(arr, index=index, columns=['importance'])
        return df

    def fi_means_stds(self):
        """Computes Mean RFI over all runs

        Returns:
            A pd.DataFrame with the relative feature importance value for
            features of interest.
        """
        self._check_shape()
        means = np.mean(self.lss, axis=(2, 1))
        stds = np.std(np.mean(self.lss, axis=2), axis=(1))
        arr = np.array([means, stds]).T
        df = pd.DataFrame(arr,
                          index=self.fsoi_names,
                          columns=['mean', 'std'])
        df.index.set_names(['feature'], inplace=True)
        return df

    def hbarplot(self, ax=None, figsize=None):
        return _barplot.fi_sns_hbarplot(self, ax=ax, figsize=figsize)
