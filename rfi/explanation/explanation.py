"""Explanations are the output of Explainers.

Aggregated or observation-wise wise results can be
accessed. Plotting functionality is available.
"""
# import numpy as np
import numpy as np

import rfi.plots._barplot as _barplot
import pandas as pd
# import itertools
import rfi.utils as utils


class Explanation:
    """Stores and provides access to results from Explainer.

    Aggregated as well as observation-wise results are stored.
    Plotting functionality is available.

    Attributes:
        fsoi: Features of interest (column names)
        scores: DataFrame with Multiindex (sample, i)
            and one column per feature of interest
            deprecated: np.array with (nr_fsoi, nr_runs, nr_obs)
        ex_name: Explanation description
    """

    def __init__(self, fsoi, scores, ex_name=None):
        """Inits Explanation with fsoi indices, fsoi names, """
        self.fsoi = fsoi  # TODO evaluate, do I need to make a copy?
        self.scores = scores  # TODO evaluate, do I need to make a copy?
        if ex_name is None:
            self.ex_name = 'Unknown'

    @staticmethod
    def from_csv(path, ex_name=None):
        index_candidates = np.array(['ordering', 'sample', 'i'])
        scores = pd.read_csv(path)
        index_names = list(index_candidates[np.isin(index_candidates, scores.columns)])
        scores = scores.set_index(index_names)
        ex = Explanation(scores.columns, scores, ex_name=ex_name)
        return ex

    def _check_shape(self):
        """Checks whether the array confirms the
        specified shape (3 dimensional).
        Cannot tell whether the ordering
        (nr_fsoi, nr_runs, nr_obs) is correct.
        """
        raise NotImplementedError('Check shape has to be '
                                  'updated for Data Frame.')
        # if len(self.lss.shape) != 3:
        #     raise RuntimeError('.lss has shape {self.lss.shape}.'
        #                        'Expected 3-dim.')

    def to_csv(self, savepath=None, filename=None):
        if savepath is None:
            savepath = ''
        if filename is None:
            filename = 'scores_' + self.ex_name + '.csv'
        self.scores.to_csv(savepath + filename)

    def fi_vals(self, fnames_as_columns=True):
        """ Computes the sample-wide RFI for each run

        Returns:
            pd.DataFrame with index: sample and fsoi as columns
        """
        # self._check_shape()
        # arr = np.mean(self.scores, axis=(2))
        # if return_np:
        #     return arr
        # else:
        #     runs = range(arr.shape[1])
        #     index = utils.create_multiindex(['feature', 'run'],
        #                                     [self.fsoi_names, runs])
        #     arr = arr.reshape(-1)
        #     df = pd.DataFrame(arr, index=index, columns=['importance'])
        # return df
        df = self.scores.mean(level='sample')
        if fnames_as_columns:
            return df
        else:
            index = utils.create_multiindex([df.index.name, 'feature'],
                                            [df.index.values, df.columns])
            df2 = pd.DataFrame(df.to_numpy().reshape(-1),
                               index=index,
                               columns=['importance'])
            return df2

    def fi_means_stds(self):
        """Computes mean score over all runs, as well es the respective standard
        deviations.

        Returns:
            A pd.DataFrame with the mean score and std for
            all features.
        """
        df = pd.DataFrame(self.scores.mean(), columns=['mean'])
        df['std'] = self.scores.std()
        df.index.set_names(['feature'], inplace=True)
        return df

    def fi_means_quantiles(self):
        """Computes mean feature importance over all runs, as well as the
        respective .05 and .95 quantiles.

        Returns:
            A pd.DataFrame with the respective characteristics for every feature.
            features are rows, quantities are columns
        """
        scores_agg = self.scores.mean(level='sample')
        df = pd.DataFrame(scores_agg.mean(), columns=['mean'])
        df['q.05'] = scores_agg.quantile(0.05)
        df['q.95'] = scores_agg.quantile(0.95)
        df.index.set_names(['feature'], inplace=True)
        return df

    def hbarplot(self, ax=None, figsize=None):
        return _barplot.fi_sns_hbarplot(self, ax=ax, figsize=figsize)
