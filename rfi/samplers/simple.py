"""Simple Conditional Sampling from observed data
Only recommended for categorical data with very few states and
all states being observed
"""
import pandas as pd
from rfi.samplers.sampler import Sampler

class SimpleSampler(Sampler):
    """
    Simple sampling from observed data
    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, **kwargs):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train, **kwargs)

    def train(self, J, G, verbose=True):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            J: features of interest
            G: arbitrary set of variables
            verbose: printing
        """

        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        if not self._train_J_degenerate(J, G, verbose=verbose):
            # TODO assert that variables are categorical
            # TODO raise error if that is not the case

            def samplefunc(eval_context, **kwargs):
                X_eval = pd.DataFrame(data=eval_context, columns=Sampler._order_fset(G))
                sample = pd.merge(X_eval[G].reset_index().reset_index(), X_train, on=G, how='left').groupby(['level_0']).sample(1)
                sample = sample.set_index('index')[Sampler._order_fset(J)]
                return sample.to_numpy()

            # TODO add alternative sampling function based on

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)
