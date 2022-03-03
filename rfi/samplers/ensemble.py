"""Categorical Sampler
Random Forest based
only works for univariate targets.
"""
import numpy as np
import pandas as pd
from rfi.samplers.sampler import Sampler
from sklearn.ensemble import RandomForestClassifier
import torch

class UnivRFSampler(Sampler):
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
        assert len(J) == 1
        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        JuG = list(set(J).union(G))

        if not self._train_J_degenerate(J, G, verbose=verbose):
            # TODO assert that variables are categorical
            # TODO raise error if that is not the case

            # TODO train model to predict j from G
            model = RandomForestClassifier() # TODO set loss to be cross-entropy
            model.fit(self.X_train[Sampler._order_fset(G)], self.X_train[J])

            def samplefunc(eval_context, num_samples=1, **kwargs):
                arrs = []
                for snr in range(num_samples):
                    X_eval = pd.DataFrame(data=eval_context, columns=Sampler._order_fset(G))

                    # TODO: use the model to sample the target variable
                    pred_proba = torch.tensor(model.predict_proba(eval_context))
                    sample = torch.multinomial(pred_proba, 1).numpy().flatten()

                    arrs.append(sample.to_numpy().reshape(1, -1, len(J)))
                res = np.concatenate(arrs, axis=0)
                res = np.swapaxes(res, 0, 1)
                return res

            # TODO add alternative sampling function based on

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)