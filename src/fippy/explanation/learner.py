import numpy as np
from fippy.explanation.explanation import Explanation

class LearnerExplanation(Explanation):
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

    def __init__(self, fsoi, scores, split, ex_name=None, ex_description=None):
        """Inits Explanation with fsoi indices, fsoi names, """
        super().__init__(fsoi, scores, ex_name=ex_name, ex_description=ex_description)
        self.split = split
        self.groupbycol = 'fit'
        
    @staticmethod 
    def from_csv(path, ex_name=None, ex_description=None):
        ex = Explanation.from_csv(path, ex_name=ex_name, ex_description=ex_description)
        lex = LearnerExplanation(ex.fsoi, ex.scores, ex.split, ex_name=ex.ex_name, ex_description=ex.ex_description)
        return lex
    
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
            t_quant = t(df=count-1).ppf((1-(alpha/2)))
            
            ci_upper = means + se * t_quant
            ci_lower = means - se * t_quant
            cis['lower'] = ci_lower
            cis['upper'] = ci_upper              
        else:
            raise NotImplementedError('Type not implemented.') 
        
        cis.sort_values('importance', ascending=False, inplace=True)
        return cis