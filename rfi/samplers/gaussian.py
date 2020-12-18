"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to sem the
conditional distribution.
"""
from rfi.samplers.sampler import Sampler
from rfi.samplers._utils import sample_id, sample_perm
from rfi.backend.gaussian import GaussianConditionalEstimator
import rfi.utils as utils
# from DeepKnockoffs import GaussianKnockoffs
import numpy as np


class GaussianSampler(Sampler):
    """
    Second order Gaussian Sampler.

    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train)
            

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

        if not super()._train_J_degenerate(J, G, verbose=verbose):
            gaussian_estimator = GaussianConditionalEstimator()
            gaussian_estimator.fit(train_inputs=self.X_train[:, J], train_context=self.X_train[:, G])
            samplefunc = lambda X_test: gaussian_estimator.sample(X_test[:, G]).reshape(-1)
            super()._store_samplefunc(J, G, samplefunc, verbose=verbose)



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
