"""
Conditional density estimation with normalizing flows. Using affine and invertable radial transformations.
"""

from nflows.flows.base import Flow, Distribution
from nflows.distributions import StandardNormal
from torch import optim, Tensor
import numpy as np
from nflows.transforms import CompositeTransform, PointwiseAffineTransform
from typing import Type, Union, Tuple, List
import torch
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import logging
from ray import tune
import ray

from rfi.backend import ConditionalDistributionEstimator
from rfi.backend.cnf.context_embedding import ContextEmbedding
from rfi.backend.cnf.transforms import ContextualInvertableRadialTransform, ContextualPointwiseAffineTransform, \
    ContextualCompositeTransform


logger = logging.getLogger(__name__)


class NormalisingFlowEstimator(Flow, ConditionalDistributionEstimator):
    """
    Conditional density estimator based on Normalising Flows
    """

    default_hparam_grid = {
        'n_epochs': tune.grid_search([500, 1000, 1500]),
        'hidden_units': tune.grid_search([(8,), (16,)]),
        'transform_classes': tune.grid_search([(ContextualPointwiseAffineTransform,),
                                               2 * (ContextualInvertableRadialTransform,) + (
                                               ContextualPointwiseAffineTransform,)]),
        'context_noise_std': tune.grid_search([0.1, 0.2, 0.3]),
        'input_noise_std': tune.grid_search([0.01, 0.05, 0.1]),
        'weight_decay': tune.grid_search([0.0, 1e-4])
    }

    def __init__(
            self,
            context_size: int,
            inputs_size: int = 1,
            transform_classes: Tuple[Type] = 2 * (ContextualInvertableRadialTransform,) + (ContextualPointwiseAffineTransform,),
            hidden_units: Tuple[int] = (16,),
            base_distribution: Distribution = StandardNormal(shape=[1]),
            n_epochs: int = 1000,
            lr: float = 0.001,
            weight_decay: float = 0.0,
            input_noise_std: float = 0.05,
            context_noise_std: float = 0.1,
            cat_context: np.array = None,
            context_normalization=True,
            inputs_normalization=True,
            device='cpu',
            **kwargs
    ):
        """
        PyTorch implementation of Noise Regularization for Conditional Density Estimation. Also works unconditionally
        (https://github.com/freelunchtheorem/Conditional_Density_Estimation)
         [Rothfuss et al, 2020; https://arxiv.org/pdf/1907.08982.pdf]
        Args:
            context_size: Dimensionality of global_context
            transform_classes: Contextual transformations list
            hidden_units: Tuple of hidden sizes for global_context embedding network
            n_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay (not applied to bias)
            input_noise_std: Noise regularisation for input
            context_noise_std: Noise regularisation for global_context
            base_distribution: Base distribution of normalising flow
            context_normalization: Mean-std normalisation of global_context
            inputs_normalization: Mean-std normalisation of context_vars
            device: cpu / cuda
        """
        # Constructing composite transformation & Initialisation of global_context embedding network
        self.hidden_units = hidden_units
        if context_size > 0:
            transform = ContextualCompositeTransform(
                [transform_cls(inputs_size=inputs_size) for transform_cls in transform_classes])
            embedding_net = ContextEmbedding(transform, input_units=context_size, hidden_units=hidden_units)
        else:
            transform = CompositeTransform(
                [transform_cls(inputs_size=inputs_size, conditional=False) for transform_cls in transform_classes])
            embedding_net = None

        assert base_distribution._shape[0] == inputs_size
        Flow.__init__(self, transform, base_distribution, embedding_net)

        # Training params
        self.lr = lr
        self.weight_decay = weight_decay
        self._init_optimizer(lr, weight_decay)
        self.n_epochs = n_epochs
        self.device = device
        self.to(self.device)

        # Regularisation
        self.input_noise_std = input_noise_std
        self.context_noise_std = context_noise_std

        if cat_context is None or len(cat_context) == 0:
            self.cat_context = []
            self.cont_context = np.arange(0, context_size)
        else:
            self.cat_context = cat_context
            self.cont_context = np.array([i for i in range(context_size) if i not in cat_context])
            self.cont_context = [] if len(self.cont_context) == 0 else self.cont_context

        # Inputs / Context one-hot encoders
        self.context_enc = OneHotEncoder(drop='if_binary', sparse=False)
        self.inputs_enc = OneHotEncoder(drop='if_binary', sparse=False)

        # Normalisation
        self.context_normalization = context_normalization
        self.inputs_normalization = inputs_normalization
        self.inputs_mean, self.inputs_std = None, None
        self.context_mean, self.context_std = None, None

        self.inputs_size = inputs_size
        self.context_size = context_size

    @property
    def context_size(self):
        return self._context_size

    @context_size.setter
    def context_size(self, context_size):
        self._context_size = context_size
        if hasattr(self, '_embedding_net') and context_size > 0:
            # While changing the context size we also need to change the Embedding network
            self._embedding_net = ContextEmbedding(self._transform, input_units=context_size, hidden_units=self.hidden_units)
            self._init_optimizer(self.lr, self.weight_decay)

    def fit(self,
            train_inputs: Union[np.array, Tensor],
            train_context: Union[np.array, Tensor] = None,
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
        """
        Log pdf function
        Args:
            inputs: Input
            context: Conditioning global_context
            data_normalization: Perform data normalisation

        Returns: np.array or Tensor with shape (inputs_size, )

        """

        inputs, context, return_numpy = self._preprocess_data(inputs, context, data_normalization, context_one_hot_encoding,
                                                              False)
        result = super().log_prob(inputs, context)

        if self.inputs_normalization:
            result = result - torch.log(self.inputs_std).sum()

        result = self._postprocess_result(result, return_numpy)
        return result

    def conditional_distribution(self, context: Union[np.array, Tensor] = None, data_normalization=True,
                                 context_one_hot_encoding=True) -> Flow:

        _, context, _ = self._preprocess_data(None, context, data_normalization, context_one_hot_encoding, False)

        transforms_list = torch.nn.ModuleList()

        if self.inputs_normalization:
            # Inverse normalisation
            transforms_list.append(PointwiseAffineTransform(shift=-self.inputs_mean / self.inputs_std,
                                                            scale=1 / self.inputs_std))

        # Forward pass, to init conditional parameters
        with torch.no_grad():
            _ = super().log_prob(torch.zeros(len(context), 1), context)

        transforms_list.extend(deepcopy(self._transform._transforms))
        cond_dist = Flow(CompositeTransform(transforms_list), self._distribution)

        return cond_dist

    def sample(self, context: Union[np.array, Tensor] = None, num_samples: int = 1, data_normalization=True,
               context_one_hot_encoding=True) -> Union[np.array, Tensor]:
        """
        Sampling from conditional distribution

        Args:
            num_samples: Number of samples per global_context
            context: Conditioning global_context
            data_normalization: Perform data normalisation

        Returns: np.array or Tensor with shape (context.shape[0], num_samples)

        """
        return_numpy, context, _ = self._preprocess_data(None, context, data_normalization, context_one_hot_encoding, False)

        result = super().sample(num_samples, context).squeeze(-1)

        if self.inputs_normalization:
            result, _ = self._transform_inverse_normalise(result, None)

        result = self._postprocess_result(result, True)
        return result
