import numpy as np
from scipy.stats import multivariate_normal, norm
import logging

import fippy.backend.utils as utils

logger = logging.getLogger(__name__)


class GaussianConditionalEstimator:
    """Conditional density estimation for joint normal distribution."""

    def __init__(self, **kwargs):
        pass

    def fit(self, train_inputs: np.ndarray, train_context: np.ndarray, **kwargs):
        """Fit Gaussian conditional estimator.

        Args:
            train_inputs: variables to be resampled
            train_context: conditioning set
        """
        train_inputs = train_inputs.reshape((train_inputs.shape[0], -1))
        train_context = train_context.reshape((train_context.shape[0], -1))
        X_train = np.concatenate([train_inputs, train_context], axis=1)

        mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train.T)

        n_in, n_co = train_inputs.shape[1], train_context.shape[1]
        cov = cov.reshape((n_in + n_co, n_in + n_co))
        inp_ind = np.arange(0, n_in, 1)
        cont_ind = np.arange(n_in, n_in + n_co, 1)

        return self.fit_mean_cov(mean, cov, inp_ind, cont_ind)

    def fit_mean_cov(self, joint_mean, joint_cov, inp_ind, cont_ind):
        """Fit using mean vector and covariance matrix.

        Args:
            joint_mean: means for all variables
            joint_cov: covariance for all variables
            inp_ind: indices of variables to be sampled
            cont_ind: context variable indices (conditioning set)
        """
        self.inp_ind = inp_ind
        if cont_ind.shape[0] == 0:
            self.RegrCoeff = np.zeros((inp_ind.shape[0], 0))
        else:
            if cont_ind.shape[0] == 1:
                Sigma_GG_inv = 1 / joint_cov[np.ix_(cont_ind, cont_ind)]
            else:
                cov_context = joint_cov[np.ix_(cont_ind, cont_ind)]
                Sigma_GG_inv = np.linalg.pinv(cov_context)
            cov_ip_con = joint_cov[np.ix_(inp_ind, cont_ind)]
            tmp = cov_ip_con @ Sigma_GG_inv
            self.RegrCoeff = tmp.reshape((len(inp_ind), len(cont_ind)))

        cov_inp = joint_cov[np.ix_(inp_ind, inp_ind)]
        cov_cont_inp = joint_cov[np.ix_(cont_ind, inp_ind)]
        self.Sigma = cov_inp - self.RegrCoeff @ cov_cont_inp
        mean_inp, mean_cont = joint_mean[inp_ind], joint_mean[cont_ind]
        self.mu_part = mean_inp - self.RegrCoeff @ mean_cont
        if not utils.isPD(self.Sigma):
            logger.info('Making Sigma positive definite')
            self.Sigma = utils.nearestPD(self.Sigma)
        return self

    def _conditional_means(self, context):
        """Compute conditional means for each observation. Returns (n_inputs, n_obs)."""
        mu_part2 = self.RegrCoeff @ context.T
        return self.mu_part.reshape((-1, 1)) + mu_part2

    def log_prob(self, inputs: np.ndarray, context: np.ndarray = None) -> np.ndarray:
        """Compute log-likelihood of inputs given context."""
        assert context is not None
        assert len(context.shape) == 2

        means = self._conditional_means(context)  # (n_inputs, n_obs)
        log_probs = np.zeros((context.shape[0],))
        for j in range(len(context)):
            mu = means[:, j]
            log_probs[j] = multivariate_normal.logpdf(inputs[j], mean=mu, cov=self.Sigma)
        return log_probs

    def sample(self, context: np.ndarray = None, num_samples=1) -> np.ndarray:
        """Sample from conditional distribution.

        Returns:
            np.ndarray of shape (n_obs, num_samples, n_inputs).
        """
        means = self._conditional_means(context)  # (n_inputs, n_obs)
        n_obs = context.shape[0]
        n_inputs = self.inp_ind.shape[0]
        res = np.empty((n_obs, num_samples, n_inputs))
        for j in range(n_obs):
            mu = means[:, j]
            res[j, :, :] = np.random.multivariate_normal(mu, self.Sigma, num_samples)
        return res
