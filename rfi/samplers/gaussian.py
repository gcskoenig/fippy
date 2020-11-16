"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to model the
conditional distribution.
"""
from rfi.samplers.sampler import Sampler
from rfi.samplers._helpers import id
import rfi.utils as utils
from DeepKnockoffs import GaussianKnockoffs
import numpy as np

class GaussianSampler(Sampler):
    """
    Second order Gaussian Sampler.

    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, fsoi):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train, fsoi)

    def train(self, jj, G, verbose=True):
        """Trains sampler using dataset to resample
        variable jj relative to G.

        Args:
            jj: feature of interest
            G: arbitrary set of variables
            verbose: printing
        """
        G_key = utils.to_key(G)
        jj_key = utils.to_key([jj])
        if verbose:
            print('Start training')
            print('G: {}, fsoi: {}'.format(G, jj))
        self._trainedGs[(jj_key, G_key)] = train_gaussian_knockoff(self.X_train, G, jj)
        if verbose:
            print('Training ended. Sampler saved.')

    def train_fsoi(self, G, verbose=True):
        """Trains sampler using the training dataset to resample
        relative to any variable set G.

        Args:
            G: arbitrary set of variables.

        Returns:
            Nothing. Now the sample function can be used
            to resample on seen or unseen data.
        """
        for jj in self.fsoi:
            self.train(jj, G, verbose=verbose)

    def sample(self, X_test, J, G, verbose=True):
        """

        """
        # initialize numpy matrix
        J = np.array(J, dtype=np.int16)
        sampled_data = np.zeros((X_test.shape[0], J.shape[0]))

        #sample
        G_key = utils.to_key(G)
        for kk in range(J.shape[0]):
            jj_key = utils.to_key([J[kk]])
            if not super().is_trained([J[kk]], G):
                print('Sampler not trained yet.')
                self.train(J[kk], G, verbose=verbose)
            sample_func = self._trainedGs[(jj_key, G_key)]
            sampled_data[:, kk] = sample_func(X_test)
        return sampled_data

    def sample_fsoi(self, X_test, G, verbose=True):
        """Sample features of interest using trained resampler.

        Args:
            X_test: Data for which sampling shall be performed.

        Returns:
            Resampled data for the features of interest.
            np.array with shape (X_test.shape[0], # features of interest)
        """
        return self.sample(X_test, self.fsoi, G, verbose=verbose)


# auxilary functions below, TODO cleanup
# TODO(gcsk) handle corner cases like "j in G" or "G empty"

def train_gaussian_knockoff(X_train, G, j):
    data = np.zeros((X_train.shape[0], G.shape[0]+1))
    data[:, :-1] = X_train[:, G]
    data[:, -1] = X_train[:, j]
    SigmaHat = np.cov(data, rowvar=False)
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data, 0))
    def sample(X_test):
        ixs = np.zeros((G.shape[0] + 1), dtype=np.int16)
        ixs[:-1] = G
        ixs[-1] = j
        knockoffs = second_order.generate(X_test[:, ixs])
        return knockoffs[:, -1]
    return sample

# def train_gaussian_knockoffs(X_train, G, fsoi):
#     fs = []
#     for jj in fsoi:
#         fs.append(train_gaussian_knockoff(X_train, G, jj))
#     def sample(X_test):
#         knockoffs = np.zeros((X_test.shape[0], fsoi.shape[0]))
#         for jj in range(len(fsoi)):
#             knockoffs[:, jj] = fs[jj](X_test)
#         return knockoffs
#     return sample
