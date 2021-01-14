"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to sem the
conditional distribution.
"""
from rfi.samplers.sampler import Sampler
from rfi.backend.gaussian import GaussianConditionalEstimator
import rfi.utils as utils
# from DeepKnockoffs import GaussianKnockoffs
import numpy as np
import logging


class GaussianSampler(Sampler):
    """
    Second order Gaussian Sampler.

    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, X_val=None, **kwargs):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train, X_val)
            

    def train(self, J, G, verbose=True):
        """Trains sampler using dataset to resample
        variable jj relative to G.

        Args:
            J: features of interest
            G: arbitrary set of variables
            verbose: printing
        """

        J = Sampler._to_array(J)
        G = Sampler._to_array(G)
        super().train(J, G, verbose=verbose)

        val_log_probs = []

        for j in J:
            j = Sampler._to_array([j])
            if not self._train_J_degenerate(j, G, verbose=verbose):

                gaussian_estimator = GaussianConditionalEstimator()
                gaussian_estimator.fit(train_inputs=self.X_train[:, j], train_context=self.X_train[:, G])

                def samplefunc(X_test, **kwargs):
                    return gaussian_estimator.sample(X_test[:, G], **kwargs)

                self._store_samplefunc(j, G, samplefunc, verbose=verbose)

                if self.X_val is not None:
                    val_log_prob = gaussian_estimator.log_prob(inputs=self.X_val[:, j], context=self.X_val[:, G]).mean()
                    val_log_probs.append(val_log_prob)

            elif self.X_val is not None:
                val_log_probs.append(None)

        if len(J) > 1:
            def samplefunc(X_test, **kwargs):
                sampled_data = []
                for j in J:
                    j = Sampler._to_array([j])
                    G_key, j_key = Sampler._to_key(G), Sampler._to_key(j)
                    sampled_data.append(np.squeeze(self._trained_sampling_funcs[(j_key, G_key)](X_test, **kwargs)))
                return np.stack(sampled_data, axis=-1)
            self._store_samplefunc(J, G, samplefunc, verbose=verbose)

        if self.X_val is not None:
            return val_log_probs


# def train_gaussian(X_train, J, G):
#     """Training conditional sampler under the
#     assumption of gaussianity
#
#     Args:
#         G: relative feature set
#         j: features of interest
#     """
#
#
#     return lambda X_test: sample(X_test[:, G]).reshape(-1)

# def train_gaussian(X_train, j, G):
#     '''Training a conditional sampler under the assumption
#     of gaussianity

#     Args:
#         G: relative feature set
#         j: feature of interest
#     '''
#     data = np.zeros((X_train.shape[0], G.shape[0]+1))
#     if j in G:
#         return sample_id(j)
#     elif G.size == 0:
#         return sample_perm(j)
#     else:
#         data[:, :-1] = X_train[:, G]
#         data[:, -1] = X_train[:, j]
#         SigmaHat = np.cov(data, rowvar=False)
#         second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data, 0))
#         def sample(X_test):
#             ixs = np.zeros((G.shape[0] + 1), dtype=np.int16)
#             ixs[:-1] = G
#             ixs[-1] = j
#             knockoffs = second_order.generate(X_test[:, ixs])
#             return knockoffs[:, -1]
#         return sample
