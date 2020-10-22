"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to model the
conditional distribution.
"""
from rfi.samplers.sampler import Sampler
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

    def train(self, G, verbose=True):
        """Trains sampler using the training dataset to resample
        relative to any variable set G.

        Args:
            G: arbitrary set of variables.

        Returns:
            Nothing. Now the sample function can be used
            to resample on seen or unseen data.
        """
        if verbose:
            print('Start training relative to G.')
            print('G: {}'.format(G))
            print('fsoi: {}'.format(self.fsoi))
        sample_func = train_gaussian_knockoffs(self.X_train, G, self.fsoi)
        if verbose:
            print('End training. Save sampler.')
        key = utils.to_key(G)
        self._trainedGs[key] = sample_func

    def sample(self, X_test, G):
        """Sample features of interest using trained resampler.

        Args:
            X_test: Data for which sampling shall be performed.

        Returns:
            Resampled data for the features of interest.
            np.array with shape (X_test.shape[0], # features of interest)
        """
        if not super().is_trained(G):  # asserts that it was trained
            print('Sampler not trained yet.')
            self.train(G)
        key = utils.to_key(G)
        sample_func = self._trainedGs[key]
        return sample_func(X_test)


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

def train_gaussian_knockoffs(X_train, G, fsoi):
    fs = []
    for jj in fsoi:
        fs.append(train_gaussian_knockoff(X_train, G, jj))
    def sample(X_test):
        knockoffs = np.zeros((X_test.shape[0], fsoi.shape[0]))
        for jj in range(len(fsoi)):
            knockoffs[:, jj] = fs[jj](X_test)
        return knockoffs
    return sample
