import numpy as np
from rfi.backend.gaussian.gaussian_estimator import GaussianConditionalEstimator
import math

name = 'chain2'

N=10**3
dataset = np.loadtxt('data/{}.csv'.format(name), dtype=np.float32)
D = np.arange(0, 4)

splitpoint = math.floor(N*0.9)
ix_train = np.arange(0, splitpoint, 1)
ix_test = np.arange(splitpoint, N, 1)

X_train, y_train = dataset[ix_train, :-1], dataset[ix_train,-1]
X_test, y_test = dataset[ix_test, :-1], dataset[ix_test,-1]

J, G = [0,1], [2, 3] # TODO(gcsk): remove before release

class TestGaussian:
    thresh = 0.1 # TODO(gcsk): remove before release

    def compute_sample(self, J, G):
        X_G = X_test[:, G]
        X_J = X_test[:, J]

        gaussian_estimator = GaussianConditionalEstimator()
        gaussian_estimator.fit(train_inputs=X_train[:, J], train_context=X_train[:, G])
        X_J_recovered = gaussian_estimator.sample(X_test[:, G])

        sample = train_gaussian(X_train, J, G)
        X_J_recovered = sample(X_G)

        X_orig = np.concatenate((X_J, X_G), axis=1)
        X_recov = np.concatenate((X_J_recovered, X_G), axis=1)

        return X_orig, X_recov

    def compute_cov_mean_diff(self, X_orig, X_recov):
        cov_orig = np.cov(X_orig.T)
        cov_recov = np.cov(X_recov.T)
        mean_orig = np.mean(X_orig, axis=0)
        mean_recov = np.mean(X_recov, axis=0)

        cov_diff = cov_orig - cov_recov
        mean_diff = mean_orig - mean_recov
        return mean_diff, cov_diff

    def test_cov_1d_same(self):
        J = np.array([1])
        G = np.array([1])

        X_orig, X_recov = self.compute_sample(J, G)
        mean_diff, cov_diff = self.compute_cov_mean_diff(X_orig, X_recov)
        
        assert np.sum(cov_diff > self.thresh) == 0
        assert np.sum(mean_diff > self.thresh) == 0

    def test_cov_1d_distinct(self):
        J = np.array([0])
        G = np.array([1])

        X_orig, X_recov = self.compute_sample(J, G)
        mean_diff, cov_diff = self.compute_cov_mean_diff(X_orig, X_recov)
        
        assert np.sum(cov_diff > self.thresh) == 0
        assert np.sum(mean_diff > self.thresh) == 0

    def test_cov_2d_distinct(self):
        J = np.array([0, 1])
        G = np.array([2, 3])

        X_orig, X_recov = self.compute_sample(J, G)
        mean_diff, cov_diff = self.compute_cov_mean_diff(X_orig, X_recov)
        
        assert np.sum(cov_diff > self.thresh) == 0
        assert np.sum(mean_diff > self.thresh) == 0
