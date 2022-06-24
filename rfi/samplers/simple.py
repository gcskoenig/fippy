"""Simple Conditional Sampling from observed data
Only recommended for categorical data with very few states and
all states being observed
"""
import numpy as np
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

        JuG = list(set(J).union(G))

        if not self._train_J_degenerate(J, G, verbose=verbose):
            # TODO assert that variables are categorical
            # TODO raise error if that is not the case

            def samplefunc(eval_context, num_samples=1, **kwargs):
                arrs = []
                for snr in range(num_samples):
                    if len(G) > 0:
                        X_eval = pd.DataFrame(data=eval_context, columns=Sampler._order_fset(G))
                        X_eval = X_eval.reset_index().reset_index().set_index(list(G))
                        X_train = self.X_train[JuG].set_index(list(G))
                        sample = X_eval.join(X_train, on=list(G), how='left').groupby(['level_0']).sample(1)
                        sample = sample.reset_index().set_index('index')[Sampler._order_fset(J)]
                        # sample = pd.merge(X_eval.reset_index().reset_index(), self.X_train[JuG], on=list(G), how='left').groupby(['level_0']).sample(1)
                        arrs.append(sample.to_numpy().reshape(1, -1, len(J)))
                    else:
                        sample = self.X_train[Sampler._order_fset(J)].sample(eval_context.shape[0])
                        arrs.append(sample.to_numpy().reshape(1, -1, len(J)))
                res = np.concatenate(arrs, axis=0)
                res = np.swapaxes(res, 0, 1)
                return res

            # TODO add alternative sampling function based on

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)