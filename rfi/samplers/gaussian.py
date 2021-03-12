"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to sem the
conditional distribution.
"""
from rfi.samplers.sampler import Sampler
from rfi.backend.gaussian import GaussianConditionalEstimator


class GaussianSampler(Sampler):
    """
    Second order Gaussian Sampler.

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

        J = Sampler._to_array(J)
        G = Sampler._to_array(G)
        super().train(J, G, verbose=verbose)

        if not self._train_J_degenerate(J, G, verbose=verbose):
            G_disjoint = set(G).isdisjoint(self.cat_inputs)
            J_disjoint = set(J).isdisjoint(self.cat_inputs)
            if not G_disjoint or not J_disjoint:
                raise NotImplementedError('GaussianConditionalEstimator does '
                                          'not support categorical variables.')

            gaussian_estimator = GaussianConditionalEstimator()
            train_inputs = self.X_train[Sampler._order_fset(J)].to_numpy()
            train_context = self.X_train[Sampler._order_fset(G)].to_numpy()

            gaussian_estimator.fit(train_inputs=train_inputs,
                                   train_context=train_context)

            def samplefunc(eval_context, **kwargs):
                return gaussian_estimator.sample(eval_context, **kwargs)

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)

            return gaussian_estimator
        else:
            return None
