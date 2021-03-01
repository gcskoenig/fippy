from typing import Type, Union, Tuple, List
import numpy as np
from scipy.stats import norm, multivariate_normal
import torch
from torch.distributions import Normal, MultivariateNormal, Distribution
from statsmodels.stats.correlation_tools import cov_nearest
import logging

from rfi.backend import ConditionalDistributionEstimator

logger = logging.getLogger(__name__)


class GaussianConditionalEstimator(ConditionalDistributionEstimator):
    """
    Conditional density estimation for joint normal distribution
    """
    default_hparam_grid = {}

    def __init__(self, **kwargs):
        super(GaussianConditionalEstimator, self).__init__()

    def __check_target_1d(self):
        """Assess whether target distribution is univariate.
        Throws an RuntimeError if not.
        """
        if np.prod(self.Sigma.shape[0] != 1):
            raise RuntimeError('The target distribution is required '
                               'to be univariate. Dimensionality of '
                               'the target distribution: '
                               '{}'.format(self.Sigma.shape[0]))
        else:
            logger.debug('Passed: Target distribution dimensionality '
                         '= {}. Continue.'.format(self.Sigma.shape[0]))

    def __check_positive_variance(self, adjust=False):
        all_positive = np.prod(np.diag(self.Sigma) > 0) == 1
        if not all_positive and adjust:  # take absolute values on diagonal
            diag_ixs = np.diag_indices(self.Sigma.shape[0], ndim=2)
            diag_vals = np.abs(np.diag(self.Sigma)) + np.finfo(float).eps
            self.Sigma[diag_ixs] = diag_vals
        return all_positive

    def fit(self, train_inputs: np.array, train_context: np.array, **kwargs):
        """Fit Gaussian Sampler.

        Args:
            train_inputs: variables to be resampled
            train_context: conditioning set
        """
        # make sure arrays are 2d and concatenate into one array
        train_inputs = train_inputs.reshape((train_inputs.shape[0], -1))
        train_context = train_context.reshape((train_context.shape[0], -1))
        X_train = np.concatenate([train_inputs, train_context], axis=1)

        mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train.T)

        n_in, n_co = train_inputs.shape[1], train_context.shape[1]
        cov = cov.reshape((n_in+n_co, n_in+n_co)) # make sure its 2d
        inp_ind = np.arange(0, n_in, 1)
        cont_ind = np.arange(n_in, n_in + n_co, 1)

        return self.fit_mean_cov(mean, cov, inp_ind, cont_ind)

    def fit_mean_cov(self, joint_mean, joint_cov, inp_ind, cont_ind):
        """Fit using mean vector and covariate matrix.

        Args:
            joint_mean: means for all variables
            cov: cov for all variables
            inp_ind: indices of variables to be sampled
            cont_ind: "context" variable indexes (conditioning set)
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

        # What if joint_cov has more vars, than in inp_ind and cont_ind ???
        # I got an Exception due to this ...
        # size = inp_ind.shape[0] + cont_ind.shape[0]
        # joint_cov = joint_cov.reshape((size, size))
        cov_inp = joint_cov[np.ix_(inp_ind, inp_ind)]
        cov_cont_inp = joint_cov[np.ix_(cont_ind, inp_ind)]
        self.Sigma = cov_inp - self.RegrCoeff @ cov_cont_inp
        mean_inp, mean_cont = joint_mean[inp_ind], joint_mean[cont_ind]
        self.mu_part = mean_inp - self.RegrCoeff @ mean_cont
        return self

    def log_prob(self, inputs: np.array, context: np.array = None) -> np.array:
        """
        Calculates log-likelihood of context_vars
        @param inputs: np.array with random variable sample, shape = (-1, 1)
        @param context: np.array, conditioning sample
        """
        inputs = inputs.reshape(-1, 1)
        assert context is not None
        # assert context_vars.shape[1] == 1
        assert len(context.shape) == 2

        mu_part2 = self.RegrCoeff @ context.T
        log_probs = np.zeros((context.shape[0],))
        for j in range(len(context)):
            mu = self.mu_part + mu_part2[:, j]
            log_probs[j] = np.log(multivariate_normal.pdf(inputs[j],
                                                          mean=mu,
                                                          cov=self.Sigma))
        return log_probs

    def cdf(self, inputs: np.array, context: np.array) -> np.array:
        """Calulates the quantile (cumulative distribution function)
        Only works for 1d inputs/targets

        Args:
            inputs: np.array with values, shape = (-1)
            context: np.array with context values, shape = (-1, d_context)
        """
        self.__check_target_1d()
        self.__check_positive_variance(adjust=True)
        qs = np.zeros(inputs.shape[0])
        mu_part2 = self.RegrCoeff @ context.T
        for j in range(len(context)):
            mu = self.mu_part + mu_part2[:, j]
            qs[j] = norm.cdf(inputs[j], loc=mu, scale=np.sqrt(self.Sigma))
        return qs

    def icdf(self, quantiles: np.array, context: np.array) -> np.array:
        """Calulates the quantile (cumulative distribution function)
        Only works for 1d inputs/targets

        Args:
            inputs: np.array with quantiles, shape = (-1)
            context: np.array with context values, shape = (-1, d_context)
        """
        self.__check_target_1d()
        self.__check_positive_variance(adjust=True)
        values = np.zeros(quantiles.shape[0])
        mu_part2 = self.RegrCoeff @ context.T
        for j in range(len(context)):
            mu = self.mu_part + mu_part2[:, j]
            values[j] = norm.ppf(q=quantiles[j],
                                 loc=mu,
                                 scale=np.sqrt(self.Sigma))
        return values

    def conditional_distribution(self,
                                 context: np.array = None) -> Distribution:
        mu_part2 = self.RegrCoeff @ context.T
        mu = self.mu_part + mu_part2
        return MultivariateNormal(torch.tensor(mu).T, torch.tensor(self.Sigma))

    def sample(self, context: np.array = None, num_samples=1) -> np.array:
        res = np.zeros((context.shape[0], num_samples, self.inp_ind.shape[0]))
        mu_part2 = self.RegrCoeff @ context.T
        for j in range(len(context)):
            mu = self.mu_part + mu_part2[:, j]
            res[j, :, :] = np.random.multivariate_normal(mu,
                                                         self.Sigma,
                                                         num_samples)
        return res
