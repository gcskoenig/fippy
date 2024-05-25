"""Explanations are the output of Explainers.

Aggregated or observation-wise wise results can be
accessed. Plotting functionality is available.
"""
import numpy as np
from fippy.explanation import Explanation


class ModelExplanation(Explanation):
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
        super().__init__(fsoi, scores, ex_name=ex_name, ex_description=ex_description)
        self.groupbycol = 'sample'
    
    def cis(self, type='two-sided', alpha=0.05):
        var = self.scores.groupby('sample').var()
        if (var == 0).all().all():
            raise RuntimeError('Variance of scores is zero. Cannot compute CI.'+
                               'Did you pass a population-wide loss function (e.g. mean_squared_error)?'+
                               'Please use square_error as loss instead.')
        
        means = self.scores.groupby('sample').mean()
        count = self.scores.groupby('sample').count().iloc[0, 0]
        
        cis = means.stack().to_frame('importance')
        cis.index.names = ['sample', 'feature']
        
        if type=='two-sided':
            # implements two-sided confidence interval by Molnar et al. (2023)
            se = np.sqrt(var / count)

            from scipy.stats import t
            alpha = 0.05
            t_quant = t(df=count-1).ppf((1-(alpha/2)))
            
            ci_upper = means + se * t_quant
            ci_lower = means - se * t_quant
            cis['lower'] = ci_lower.stack().values
            cis['upper'] = ci_upper.stack().values                  
        elif type=='one-sided':
            # implements one-sided test by Watson et al. (2019)
            stds = self.scores.groupby('sample').std()
            se = stds / np.sqrt(count)

            ts = means / se

            from scipy.stats import t
            alpha = 0.05
            cutoff = t(df=count-1).ppf((1-alpha))
            ts > cutoff
            ci_lower = means - se * cutoff      
            cis['lower'] = ci_lower.stack().values     
        else:
            raise NotImplementedError('Type not implemented.') 
        
        cis.sort_values('importance', ascending=False, inplace=True)
        return cis