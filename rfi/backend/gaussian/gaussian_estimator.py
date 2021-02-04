from nflows.flows.base import Flow, Distribution
from typing import Type, Union, Tuple, List
import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.distributions import Normal, MultivariateNormal, Distribution
from statsmodels.stats.correlation_tools import cov_nearest


class GaussianConditionalEstimator(Distribution):
    """
    Conditional density estimation for joint normal distribution
    """

    def __init__(self, **kwargs):
        super(GaussianConditionalEstimator, self).__init__()

    def fit(self, train_inputs: np.array, train_context: np.array):
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
        inp_ind, cont_ind = np.arange(0, n_in, 1), np.arange(n_in, n_in+n_co, 1)

        self.fit_mean_cov(mean, cov, inp_ind, cont_ind)

    def fit_mean_cov(self, joint_mean, joint_cov, inp_ind, cont_ind):
        """Fit using mean vector and covariate matrix.

        Args:
            joint_mean: means for all variables
            cov: cov for all variables
            inp_ind: indices of variables to be sampled
            cont_ind: "context" variable indexes (conditioning set)
        """
        self.inp_ind = inp_ind
        if cont_ind.shape[0] == 1:
            Sigma_GG_inv = 1 / joint_cov[np.ix_(cont_ind, cont_ind)]
        else:
            Sigma_GG_inv = np.linalg.pinv(joint_cov[np.ix_(cont_ind, cont_ind)])
        self.RegrCoeff = (joint_cov[np.ix_(inp_ind, cont_ind)] @ Sigma_GG_inv).reshape((len(inp_ind), len(cont_ind)))
        self.Sigma = joint_cov[np.ix_(inp_ind, inp_ind)] - self.RegrCoeff @ joint_cov[np.ix_(cont_ind, inp_ind)]
        # self.Sigma = cov_nearest(self.Sigma, threshold=1e-16)
        self.mu_part = joint_mean[inp_ind] - self.RegrCoeff @ joint_mean[cont_ind]

    def log_prob(self, inputs: np.array, context: np.array = None):
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
            log_probs[j] = np.log(multivariate_normal.pdf(inputs[j], mean=mu, cov=self.Sigma))
        return log_probs

    def conditional_distribution(self, context: np.array = None) -> Distribution:
        mu_part2 = self.RegrCoeff @ context.T
        mu = self.mu_part + mu_part2
        # if len(self.inp_ind) == 1:
        #     return Normal(torch.tensor(mu[0]), torch.sqrt(torch.tensor(self.Sigma[0, 0])))
        # else:
        return MultivariateNormal(torch.tensor(mu), torch.tensor(self.Sigma))

    def sample(self, context: np.array, num_samples=1) -> np.array:
        res = np.zeros((context.shape[0], num_samples, self.inp_ind.shape[0]))
        mu_part2 = self.RegrCoeff @ context.T
        for j in range(len(context)):
            mu = self.mu_part + mu_part2[:, j]
            res[j, :, :] = np.random.multivariate_normal(mu, self.Sigma, num_samples)
        return res
