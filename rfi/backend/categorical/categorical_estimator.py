from nflows.flows.base import Flow, Distribution
from nflows.distributions import StandardNormal
from torch import optim, Tensor
import numpy as np
from torch.distributions import Normal, OneHotCategorical, MixtureSameFamily, Independent

from typing import Type, Union, Tuple, List
import torch
from copy import deepcopy
from sklearn.model_selection import KFold
import logging
from ray import tune
import ray

from rfi.backend import ConditionalDistributionEstimator
from rfi.backend.cnf.context_embedding import ContextEmbedding
from rfi.backend.mdn.mdn_estimator import CategoricalNetwork


logger = logging.getLogger(__name__)


class CategoricalEstimator(ConditionalDistributionEstimator):

    default_hparam_grid = {
        'n_epochs': tune.grid_search([500, 1000, 1500]),
        'hidden_dim': tune.grid_search([None, 8, 16]),
        'context_noise_std': tune.grid_search([0.1, 0.2, 0.3]),
        'input_noise_std': tune.grid_search([0.01, 0.05, 0.1]),
        'weight_decay': tune.grid_search([0.0, 1e-4])
    }

    def __init__(self,
                 context_size: int,
                 inputs_size: int = 1,
                 hidden_dim: int = None,
                 n_epochs: int = 1000,
                 lr: float = 0.005,
                 weight_decay: float = 0.0,
                 context_noise_std: float = 0.1,
                 cat_context: np.array = None,
                 device='cpu',
                 context_normalization=True,
                 **kwargs):
        super().__init__(context_size=context_size, inputs_size=inputs_size, context_normalization=context_normalization,
                         inputs_normalization=False, cat_context=cat_context)
        self.pi_network = CategoricalNetwork(context_size, 1, hidden_dim=hidden_dim)
        self.n_epochs = n_epochs
        self.hidden_dim = hidden_dim

        # Training
        self.lr = lr
        self.weight_decay = weight_decay
        self._init_optimizer(lr, weight_decay)
        self.n_epochs = n_epochs
        self.device = device
        self.to(self.device)

        # Regularisation
        self.context_noise_std = context_noise_std

    @property
    def context_size(self):
        return self._context_size

    @property
    def inputs_size(self):
        return self._inputs_size

    @context_size.setter
    def context_size(self, context_size):
        self._context_size = context_size
        if hasattr(self, 'pi_network') and context_size > 0:
            # While changing the context_size we also need to change the CategoricalNetwork
            self.pi_network = CategoricalNetwork(context_size, self.inputs_size, hidden_dim=self.hidden_dim)
            self._init_optimizer(self.lr, self.weight_decay)

    @inputs_size.setter
    def inputs_size(self, inputs_size):
        self._inputs_size = inputs_size
        if hasattr(self, 'pi_network') and inputs_size > 0:
            # While changing inputs_size we also need to change the CategoricalNetwork
            self.pi_network = CategoricalNetwork(self.context_size, inputs_size, hidden_dim=self.hidden_dim)
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

        train_inputs, train_context = self._fit_transform_onehot_encode(train_inputs, train_context)
        train_inputs, train_context = self._input_to_tensor(train_inputs, train_context)
        _, train_context = self._fit_transform_normalise(None, train_context)

        val_inputs, val_context = self._transform_onehot_encode(val_inputs, val_context)
        val_inputs, val_context = self._input_to_tensor(val_inputs, val_context)
        _, val_context = self._transform_normalise(None, val_context)

        for i in range(self.n_epochs):
            self.optimizer.zero_grad()
            # Adding noise to data
            noised_train_context = self._add_noise(train_context, self.context_noise_std)
            # Forward pass
            loss = - self.log_prob(inputs=train_inputs, context=noised_train_context, data_normalization=False,
                                   inputs_one_hot_ecoding=False, context_one_hot_encoding=False).mean()
            loss.backward()
            self.optimizer.step()

            if verbose and (i + 1) % log_frequency == 0:
                with torch.no_grad():
                    train_log_lik = self.log_prob(inputs=train_inputs, context=train_context, data_normalization=False,
                                                  inputs_one_hot_ecoding=False, context_one_hot_encoding=False).mean()

                    if val_inputs is not None:
                        val_log_lik = self.log_prob(inputs=val_inputs, context=val_context, data_normalization=False,
                                                    inputs_one_hot_ecoding=False, context_one_hot_encoding=False).mean()
                        logger.info(f'{i}: train log-likelihood: {train_log_lik}, val log-likelihood: {val_log_lik}')
                    else:
                        logger.info(f'{i}: train log-likelihood: {train_log_lik}')

        return self

    def log_prob(self, inputs: Union[np.array, Tensor], context: Union[np.array, Tensor] = None,
                 data_normalization=True, context_one_hot_encoding=True, inputs_one_hot_ecoding=True) -> Union[np.array, Tensor]:
        inputs, context, return_numpy = self._preprocess_data(inputs, context, data_normalization, context_one_hot_encoding,
                                                              inputs_one_hot_ecoding)
        result = self.pi_network(context).log_prob(inputs)
        result = self._postprocess_result(result, return_numpy)
        return result

    def conditional_distribution(self, context: Union[np.array, Tensor] = None, data_normalization=True,
                                 context_one_hot_encoding=True) -> torch.distributions.Distribution:

        inputs, context, return_numpy = self._preprocess_data(None, context, data_normalization, context_one_hot_encoding, False)
        return self.pi_network(context)

    def sample(self, context: Union[np.array, Tensor] = None, num_samples: int = 1, data_normalization=True,
               context_one_hot_encoding=True) -> Union[np.array, Tensor]:

        inputs, context, return_numpy = self._preprocess_data(None, context, data_normalization, context_one_hot_encoding, False)
        result = self.pi_network(context).sample((num_samples, ))
        result = np.squeeze(np.array([self._inverse_onehot_encode(res) for res in result]))
        return result
