"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to sem the
conditional distribution.
"""
import numpy as np

from fippy.samplers.sampler import Sampler
from fippy.backend.estimators import GaussianConditionalEstimator
from fippy.samplers._utils import sample_id
from fippy.utils import fset_to_ix


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

        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        if not self._train_J_degenerate(J, G):
            G_disjoint = set(G).isdisjoint(self.cat_inputs)
            J_disjoint = set(J).isdisjoint(self.cat_inputs)
            if not G_disjoint or not J_disjoint:
                raise NotImplementedError('GaussianConditionalEstimator does '
                                          'not support categorical variables.')

            # to be sampled using gaussian estimator
            J_R = list(set(J) - set(G))
            # to be ID returned
            J_G = list(set(J) - set(J_R))

            gaussian_estimator = GaussianConditionalEstimator()
            train_inputs = self.X_train[Sampler._order_fset(J_R)].to_numpy()
            train_context = self.X_train[Sampler._order_fset(G)].to_numpy()

            gaussian_estimator.fit(train_inputs=train_inputs,
                                   train_context=train_context)

            J_G_ixs = fset_to_ix(G, J)
            samplef_J_G = sample_id(J_G_ixs)

            ixs_J_G = fset_to_ix(J, J_G)
            ixs_J_R = fset_to_ix(J, J_R)

            def samplefunc(eval_context, **kwargs):
                sample_J_G = samplef_J_G(eval_context, **kwargs)
                sample_J_R = gaussian_estimator.sample(eval_context, **kwargs)
                sample = np.zeros((sample_J_R.shape[0], sample_J_R.shape[1],
                                   sample_J_R.shape[2] + sample_J_G.shape[2]))
                sample[:, :, ixs_J_G] = sample_J_G
                sample[:, :, ixs_J_R] = sample_J_R
                return sample

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)

            return gaussian_estimator
        else:
            return None
