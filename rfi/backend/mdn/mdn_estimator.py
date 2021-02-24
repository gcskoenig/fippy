from typing import Union
import numpy as np
import logging
from torch import Tensor
import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical, MixtureSameFamily, Independent
from ray import tune
import torch.nn.init as init
from pyro.contrib.randomvariable import RandomVariable

from rfi.backend import ConditionalDistributionEstimator

logger = logging.getLogger(__name__)


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, context_size, inputs_size, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = context_size
        if context_size > 0:
            self.network = nn.Sequential(
                nn.Linear(context_size, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2 * inputs_size * n_components),
            )
        else:  # Unconditional distribution
            self.mean = torch.nn.Parameter(torch.empty((n_components, inputs_size)))
            self.log_std = torch.nn.Parameter(torch.empty((n_components, inputs_size)))

            init.normal_(self.mean, -1.0, 1.0)
            init.normal_(self.log_std, 0.0, 1.0)

    def forward(self, context=None):

        if context is not None:
            params = self.network(context)
            mean, log_std = torch.split(params, params.shape[1] // 2, dim=1)
            mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
            log_std = torch.stack(log_std.split(log_std.shape[1] // self.n_components, 1))
            self.mean, self.log_std = mean.transpose(0, 1), log_std.transpose(0, 1)

        return Normal(self.mean, torch.exp(self.log_std))


class CategoricalNetwork(nn.Module):

    def __init__(self, context_size, n_components, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = context_size
        if context_size > 0:
            self.network = nn.Sequential(
                nn.Linear(context_size, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, n_components)
            )
        else:  # Unconditional distribution
            self.logits = torch.nn.Parameter(torch.empty((n_components, )))
            init.normal_(self.logits, -1.0, 1.0)

    def forward(self, context=None):
        if context is not None:
            self.logits = self.network(context)
        return OneHotCategorical(logits=self.logits)


class MixtureDensityNetworkEstimator(ConditionalDistributionEstimator, nn.Module):
    """
    Conditional density estimator based on Mixture Density Networks [ Bishop, 1994 ]
    """

    default_hparam_grid = {
        'n_epochs': tune.grid_search([500, 1000, 1500]),
        'n_components': tune.grid_search([3, 5, 10]),
        'hidden_dim': tune.grid_search([None, 8, 16]),
        'context_noise_std': tune.grid_search([0.1, 0.2, 0.3]),
        'input_noise_std': tune.grid_search([0.01, 0.05, 0.1]),
        'weight_decay': tune.grid_search([0.0, 1e-4])
    }

    def __init__(self,
                 context_size: int,
                 inputs_size: int = 1,
                 n_components: int = 5,
                 hidden_dim: int = None,
                 n_epochs: int = 1000,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 input_noise_std: float = 0.05,
                 context_noise_std: float = 0.1,
                 cat_context: np.array = None,
                 device='cpu',
                 context_normalization=True,
                 inputs_normalization=True,
                 **kwargs):
        super().__init__(context_size=context_size, inputs_size=inputs_size, cat_context=cat_context,
                         context_normalization=context_normalization, inputs_normalization=inputs_normalization)
        self.pi_network = CategoricalNetwork(context_size, n_components, hidden_dim=hidden_dim)
        self.normal_network = MixtureDiagNormalNetwork(context_size, inputs_size, n_components, hidden_dim=hidden_dim)
        self.n_epochs = n_epochs
        self.hidden_dim = hidden_dim
        self.n_components = n_components

        # Training details
        self.lr = lr
        self.weight_decay = weight_decay
        self._init_optimizer(lr, weight_decay)
        self.n_epochs = n_epochs
        self.device = device
        self.to(self.device)

        # Regularisation
        self.input_noise_std = input_noise_std
        self.context_noise_std = context_noise_std

    @property
    def context_size(self):
        return self._context_size

    @context_size.setter
    def context_size(self, context_size):
        self._context_size = context_size
        if hasattr(self, 'pi_network') and context_size > 0:
            # While changing the context_size we also need to change the CategoricalNetwork
            self.pi_network = CategoricalNetwork(context_size, self.n_components, hidden_dim=self.hidden_dim)
            self.normal_network = MixtureDiagNormalNetwork(context_size, self.inputs_size, self.n_components,
                                                           hidden_dim=self.hidden_dim)
            self._init_optimizer(self.lr, self.weight_decay)

    def fit(self,
            train_inputs: Union[np.array, Tensor],
            train_context: [np.array, Tensor] = None,
            verbose=False,
            val_inputs: Union[np.array, Tensor] = None,
            val_context: Union[np.array, Tensor] = None,
            log_frequency: int = 100,
            **kwargs):
        """
        Method to fit Conditional Normalizing Flow density estimator
        Args:
            train_inputs: Train input
            train_context: Train conditioning global_context
            verbose: True - prints train (value) log-likelihood every log_frequency epoch
            val_inputs: Validation input
            val_context: Validation conditioning global_context
            log_frequency: Frequency of logging, only works, when verbose == True

        Returns: self
        """

        _, train_context = self._fit_transform_onehot_encode(None, train_context)
        train_inputs, train_context = self._input_to_tensor(train_inputs, train_context)
        train_inputs, train_context = self._fit_transform_normalise(train_inputs, train_context)

        _, val_context = self._transform_onehot_encode(None, val_context)
        val_inputs, val_context = self._input_to_tensor(val_inputs, val_context)
        val_inputs, val_context = self._transform_normalise(val_inputs, val_context)

        for i in range(self.n_epochs):
            self.optimizer.zero_grad()
            # Adding noise to data
            noised_train_inputs = self._add_noise(train_inputs, self.input_noise_std)
            noised_train_context = self._add_noise(train_context, self.context_noise_std)

            # Forward pass
            loss = - self.log_prob(inputs=noised_train_inputs, context=noised_train_context,
                                   data_normalization=False, context_one_hot_encoding=False).mean()
            loss.backward()
            self.optimizer.step()

            if verbose and (i + 1) % log_frequency == 0:
                with torch.no_grad():
                    train_log_lik = self.log_prob(inputs=train_inputs, context=train_context, data_normalization=False,
                                                  context_one_hot_encoding=False).mean()

                    if val_inputs is not None:
                        val_log_lik = self.log_prob(inputs=val_inputs, context=val_context, data_normalization=False,
                                                    context_one_hot_encoding=False).mean()
                        logger.info(f'{i}: train log-likelihood: {train_log_lik}, val log-likelihood: {val_log_lik}')
                    else:
                        logger.info(f'{i}: train log-likelihood: {train_log_lik}')

        return self

    def log_prob(self, inputs: Union[np.array, Tensor], context: Union[np.array, Tensor] = None,
                 data_normalization=True, context_one_hot_encoding=True) -> Union[np.array, Tensor]:

        inputs, context, return_numpy = self._preprocess_data(inputs, context, data_normalization, context_one_hot_encoding,
                                                              False)

        pi, normal = self.pi_network(context), self.normal_network(context)
        if context is not None:
            loglik = normal.log_prob(inputs.unsqueeze(1).expand_as(normal.loc))
        else:
            loglik = normal.log_prob(inputs.unsqueeze(1))
        result = torch.logsumexp(pi.probs.log() + loglik.sum(dim=2), dim=1)

        if self.inputs_normalization:
            result = result - torch.log(self.inputs_std).sum()

        result = self._postprocess_result(result, return_numpy)
        return result

    def conditional_distribution(self, context: Union[np.array, Tensor] = None, data_normalization=True,
                                 context_one_hot_encoding=True) -> torch.distributions.Distribution:

        _, context, _ = self._preprocess_data(None, context, data_normalization, context_one_hot_encoding, False)

        pi, normal = self.pi_network(context), self.normal_network(context)
        mixture = RandomVariable(MixtureSameFamily(pi._categorical, Independent(normal, 1)))

        if self.inputs_normalization:
            mixture, _ = self._transform_inverse_normalise(mixture, None)

        return mixture.dist

    def sample(self, context: Union[np.array, Tensor] = None, num_samples: int = 1, data_normalization=True,
               context_one_hot_encoding=True) -> Union[np.array, Tensor]:

        return_numpy, context, _ = self._preprocess_data(None, context, data_normalization, context_one_hot_encoding, False)

        pi, normal = self.pi_network(context), self.normal_network(context)
        result = torch.sum(pi.sample((num_samples, )).unsqueeze(-1) * normal.sample((num_samples, )), dim=-2)

        if self.inputs_normalization:
            result, _ = self._transform_inverse_normalise(result, None)

        result = self._postprocess_result(result, True)
        return result
