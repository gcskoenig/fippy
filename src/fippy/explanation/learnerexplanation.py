import numpy as np
import pandas as pd
from fippy.utils import create_multiindex
import fippy.plots._barplot as _barplot

class LearnerExplanation:
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

    def __init__(self, fsoi, scores, split, ex_name=None):
        """Inits Explanation with fsoi indices, fsoi names, """
        self.fsoi = fsoi  
        self.scores = scores 
        self.ex_name = ex_name
        self.split = split # tuple with (n_train, n_test)
        assert isinstance(self.split, tuple)
        if ex_name is None:
            self.ex_name = 'Unknown'

    @staticmethod
    def from_csv(path, ex_name=None):
        index_candidates = np.array(['ordering', 'fit', 'sample', 'i'])
        scores = pd.read_csv(path)
        index_names = list(index_candidates[np.isin(index_candidates, scores.columns)])
        scores = scores.set_index(index_names)
        ex = LearnerExplanation(scores.columns, scores, ex_name=ex_name)
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
        self.scores.to_csv(savepath + filename)

    def fi_vals(self, fnames_as_columns=True):
        """ Computes the sample-wide RFI for each run

        Returns:
            pd.DataFrame with index: sample and fsoi as columns
        """
        df = self.scores.groupby(level='fit').mean()
        if fnames_as_columns:
            return df
        else:
            index = create_multiindex([df.index.name, 'feature'],
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
        fi_vals = self.fi_vals(fnames_as_columns=True)
        df = pd.DataFrame(fi_vals.mean(), columns=['mean'])
        df['std'] = fi_vals.std()
        df.index.set_names(['feature'], inplace=True)
        return df

    def fi_means_quantiles(self):
        """Computes mean feature importance over all runs, as well as the
        respective .05 and .95 quantiles.

        Returns:
            A pd.DataFrame with the respective characteristics for every feature.
            features are rows, quantities are columns
        """
        scores_agg = self.scores.groupby(level='fit').mean()
        df = pd.DataFrame(scores_agg.mean(), columns=['mean'])
        df['q.05'] = scores_agg.quantile(0.05)
        df['q.95'] = scores_agg.quantile(0.95)
        df.index.set_names(['feature'], inplace=True)
        return df
    
    def cis(self, type='two-sided', alpha=0.05, c=None):
        """Computes confidence intervals for the feature importance.
        
        Args:
            type: Type of confidence interval. 'two-sided' or 'one-sided'
            c: correction term. Recommended to be set to ntest/train
            alpha: Significance level
        """
        agg = self.scores.groupby('fit').mean()
        var = agg.var()
        if (var == 0).all():
            raise RuntimeError('Variance of scores is zero. Did you specify only one fit?')
        
        means = agg.mean()
        count = agg.shape[0]
        
        cis = means.to_frame('importance')
        cis.index.name = 'feature'
        
        if type=='two-sided':
            # implements the learner ci procedure from Molnar et al. (2023)
            if c is None:
                c = self.split[1] / self.split[0]
            se = np.sqrt(var * (c + 1/count))

            from scipy.stats import t
            alpha = 0.05
            t_quant = t(df=count-1).ppf((1-(alpha/2)))
            
            ci_upper = means + se * t_quant
            ci_lower = means - se * t_quant
            cis['lower'] = ci_lower
            cis['upper'] = ci_upper              
        else:
            raise NotImplementedError('Type not implemented.') 
        
        cis.sort_values('importance', ascending=False, inplace=True)
        return cis

    def hbarplot(self, ax=None, figsize=None):
        return _barplot.fi_sns_hbarplot(self, ax=ax, figsize=figsize)
