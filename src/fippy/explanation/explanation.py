"""Explanations are the output of Explainers.

Aggregated or observation-wise wise results can be
accessed. Plotting functionality is available.
"""
import numpy as np
import fippy.plots._barplot as _barplot
import pandas as pd
import fippy.utils as utils
import json


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

    def __init__(self, fsoi, scores, ex_name=None, ex_description=None):
        """Inits Explanation with fsoi indices, fsoi names, """
        self.fsoi = fsoi  # TODO evaluate, do I need to make a copy?
        self.scores = scores  # TODO evaluate, do I need to make a copy?
        self.ex_name = ex_name
        self.ex_description = ex_description
        self.groupbycol = 'sample'
        self.split = None
        if ex_name is None:
            self.ex_name = 'Unknown'
        if ex_description is None:
            self.ex_description = 'No description available.'

    @staticmethod
    def from_csv(path, ex_name=None, ex_description=None):
        index_candidates = np.array(['ordering', 'fit', 'sample', 'i'])
        scores = pd.read_csv(path)
        index_names = list(index_candidates[np.isin(index_candidates, scores.columns)])
        scores = scores.set_index(index_names)
        ex = Explanation(scores.columns, scores, ex_name=ex_name, ex_description=ex_description)
        metadata = {}
        with open(path + '.json', 'r') as file:
            metadata = json.load(file)
        ex.ex_name = metadata['ex_name']
        ex.ex_description = metadata['ex_description']
        ex.fsoi = metadata['fsoi']
        ex.groupbycol = metadata['groupbycol']
        ex.split = metadata['split']
        return ex

    def _check_shape(self):
        """Checks whether the array confirms the
        specified shape (3 dimensional).
        Cannot tell whether the ordering
        (nr_fsoi, nr_runs, nr_obs) is correct.
        """
        raise NotImplementedError('Check shape has to be '
                                  'updated for Data Frame.')

    def to_csv(self, savepath=None, filename=None):
        if savepath is None:
            savepath = ''
        if filename is None:
            filename = 'scores_' + self.ex_name + '.csv'
        metadata = {
            'ex_name': self.ex_name,
            'ex_description': self.ex_description,
            'fsoi': list(self.fsoi),
            'groupbycol': self.groupbycol,
            'split': self.split
        }
        self.scores.to_csv(savepath + filename)
        with open(savepath + filename + '.json', 'w') as file:
            json.dump(metadata, file)
        
    def fi_vals(self, fnames_as_columns=True):
        """ Computes the mean importance for each run.

        Returns:
            pd.DataFrame with index: sample and fsoi as columns
        """
        df = self.scores.groupby(level=self.groupbycol).mean()
        if fnames_as_columns:
            return df
        else:
            return df.stack()

    def fi_means_stds(self, level='over_groups'):
        """Computes mean score for each run as well as the respective standard deviations.

        Returns:
            A pd.DataFrame with the mean score and std for
            all features.
        """
        if level == 'over_groups':
            scores_agg = self.scores.groupby(level=self.groupbycol).mean()
            fi_vals = scores_agg.mean()
            fi_vals.columns = ['mean']
            fi_vals['std'] = scores_agg.std()
        elif level == 'per_group':
            fi_vals = self.scores.groupby(level=self.groupbycol).mean().stack()
            fi_vals.columns = ['mean']
            fi_vals['std'] = self.scores.groupby(level=self.groupbycol).std().stack()
        else:
            raise NotImplementedError(f'Level {level} not implemented.')
        return fi_vals

    # def fi_means_quantiles(self):
    #     """Computes mean feature importance over all runs, as well as the
    #     respective .05 and .95 quantiles.

    #     Returns:
    #         A pd.DataFrame with the respective characteristics for every feature.
    #         features are rows, quantities are columns
    #     """
    #     scores_agg = self.scores.groupby(level=self.groupbycol).mean()
    #     df = pd.DataFrame(scores_agg.mean(), columns=['mean'])
    #     df['q.05'] = scores_agg.quantile(0.05)
    #     df['q.95'] = scores_agg.quantile(0.95)
    #     df.index.set_names(['feature'], inplace=True)
    #     return df
        
    def hbarplot(self, ax=None, figsize=None, alpha=0.05, facecolor='darkgray', errcolor='black'):
        return _barplot.fi_sns_hbarplot(self, ax=ax, figsize=figsize, alpha=alpha,
                                        facecolor=facecolor, errcolor=errcolor)

    def cis(self, type='two-sided', alpha=0.05):
        raise NotImplementedError('This method is not implemented for the base class.')
