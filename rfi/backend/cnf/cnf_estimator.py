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

from rfi.backend.cnf.context_embedding import ContextEmbedding
from rfi.backend.cnf.transforms import ContextualInvertableRadialTransform, ContextualAffineTransform, \
    ContextualCompositeTransform

from sklearn.model_selection import KFold
import logging
from ray import tune
import ray

logger = logging.getLogger(__name__)


class ConditionalNormalisingFlowEstimator(Flow):
    """
    Conditional density estimator based on Normalising Flows
    """

    default_hparam_grid = {
        'n_epochs': tune.grid_search([500, 1000, 1500]),
        'hidden_units': tune.grid_search([(8,), (16,)]),
        'transform_classes': tune.grid_search([(ContextualAffineTransform,),
                                               2 * (ContextualInvertableRadialTransform,) + (ContextualAffineTransform,)]),
        'context_noise_std': tune.grid_search([0.1, 0.2, 0.3]),
        'input_noise_std': tune.grid_search([0.01, 0.05, 0.1]),
        'weight_decay': tune.grid_search([0.0, 1e-4])
    }

    def __init__(self,
                 context_size: int,
                 transform_classes: Tuple[Type] = 2 * (ContextualInvertableRadialTransform,) + (ContextualAffineTransform,),
                 hidden_units: Tuple[int] = (16,),
                 n_epochs: int = 1000,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 input_noise_std: float = 0.05,
                 context_noise_std: float = 0.1,
                 base_distribution: Distribution = StandardNormal(shape=[1]),
                 context_normalization=True,
                 inputs_normalization=True,
                 device='cpu',
                 **kwargs):
        """
        PyTorch implementation of Noise Regularization for Conditional Density Estimation
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
        # Constructing composite transformation
        transform = ContextualCompositeTransform([transform_cls() for transform_cls in transform_classes])

        # Initialisation of global_context embedding network
        self.context_size = context_size
        embedding_net = ContextEmbedding(transform, input_units=context_size, hidden_units=hidden_units)

        super().__init__(transform, base_distribution, embedding_net)

        # Training
        self._init_optimizer(lr, weight_decay)
        self.n_epochs = n_epochs
        self.device = device
        self.to(self.device)

        # Regularisation
        self.input_noise_std = input_noise_std
        self.context_noise_std = context_noise_std

        # Normalisation
        self.context_normalization = context_normalization
        self.inputs_normalization = inputs_normalization
        self.inputs_mean, self.inputs_std = None, None
        self.context_mean, self.context_std = None, None

    def _init_optimizer(self, lr, weight_decay):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self.optimizer = optim.Adam(optimizer_grouped_parameters, betas=(0.9, 0.99), lr=lr)

    def _input_to_tensor(self, inputs: np.array, context: np.array) -> [Tensor, Tensor]:
        if inputs is not None and context is not None:
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device).reshape(-1, 1)
            context = torch.tensor(context, dtype=torch.float32, device=self.device).reshape(len(inputs), -1)
        return inputs, context

    @staticmethod
    def _add_noise(data: Tensor, std: float) -> Tensor:
        return data + torch.randn(data.size()).type_as(data) * std

    def _fit_transform_normalise(self, train_inputs: Tensor, train_context: Tensor):
        if train_inputs is not None and self.inputs_normalization:
            self.inputs_mean, self.inputs_std = train_inputs.mean(0), train_inputs.std(0)
            train_inputs = (train_inputs - self.inputs_mean) / self.inputs_std
        if train_context is not None and self.context_normalization:
            self.context_mean, self.context_std = train_context.mean(0), train_context.std(0)
            train_context = (train_context - self.context_mean) / self.context_std
        return train_inputs, train_context

    def _transform_normalise(self, inputs: Tensor, context: Tensor):
        if inputs is not None and self.inputs_normalization:
            inputs = (inputs - self.inputs_mean) / self.inputs_std
        if context is not None and self.context_normalization:
            context = (context - self.context_mean) / self.context_std
        return inputs, context

    def _transform_inverse_normalise(self, inputs: Tensor, context: Tensor):
        if inputs is not None and self.inputs_normalization:
            inputs = inputs * self.inputs_std + self.inputs_mean
        if context is not None and self.context_normalization:
            context = context * self.context_std + self.context_mean
        return inputs, context

    def _reset_parameters(self):
        self._embedding_net.reset_parameters()

    def fit(self,
            train_inputs: Union[np.array, Tensor],
            train_context: Union[np.array, Tensor],
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

        train_inputs, train_context = self._input_to_tensor(train_inputs, train_context)
        train_inputs, train_context = self._fit_transform_normalise(train_inputs, train_context)

        val_inputs, val_context = self._input_to_tensor(val_inputs, val_context)
        val_inputs, val_context = self._transform_normalise(val_inputs, val_context)

        for i in range(self.n_epochs):
            self.optimizer.zero_grad()
            # Adding noise to data
            noised_train_inputs = self._add_noise(train_inputs, self.input_noise_std)
            noised_train_context = self._add_noise(train_context, self.context_noise_std)
            # Forward pass
            loss = - self.log_prob(inputs=noised_train_inputs, context=noised_train_context, data_normalization=False).mean()
            loss.backward()
            self.optimizer.step()

            if verbose and (i + 1) % log_frequency == 0:
                with torch.no_grad():
                    train_log_lik = self.log_prob(inputs=train_inputs, context=train_context, data_normalization=False).mean()

                    if val_inputs is not None and val_context is not None:
                        val_log_lik = self.log_prob(inputs=val_inputs, context=val_context, data_normalization=False).mean()
                        print(f'{i}: train log-likelihood: {train_log_lik}, value log likelihood: {val_log_lik}')
                    else:
                        print(f'{i}: train log-likelihood: {train_log_lik}')

        return self

    def fit_by_cv(self, train_inputs: Union[np.array, Tensor], train_context: Union[np.array, Tensor],
                  hparam_grid=None, n_splits=5, resources_per_trial={"cpu": 0.5}, time_budget_s=None, num_cpus=15):
        """
        Method for hyper-parameter search for Conditional Normalizing Flow density estimator, performs K-fold cross-validation.
        After the hyper-parameter search, fits the best sem on full train dataset.
        Using ray.tune as backend

        Args:
            train_inputs: Train input
            train_context: Train conditioning global_context
            hparam_grid: Hyper-parameter grid. If None, default is used
            n_splits: Number of splits for K-Fold cross-validation
            resources_per_trial: Ray tune parameter
            time_budget_s: Total time budget (Ray tune parameter)
            num_cpus: Number of CPUs to employ

        Returns: self
        """

        ray.init(logging_level=logging.WARN, num_cpus=num_cpus)
        logger.info(f'Start fitting, using {n_splits}-fold split, time budget: {time_budget_s}')

        if hparam_grid is None:
            hparam_grid = ConditionalNormalisingFlowEstimator.default_hparam_grid

        def ray_fit(config):
            val_log_liks = []
            splitter = KFold(n_splits=n_splits)
            for (train_ind, val_ind) in splitter.split(X=train_context, y=train_inputs):
                train_inputs_, train_context_ = train_inputs[train_ind], train_context[train_ind]
                val_inputs_, val_context_ = train_inputs[val_ind], train_context[val_ind]

                flow = ConditionalNormalisingFlowEstimator(context_size=self.context_size, device=self.device, **config)
                flow.fit(train_inputs_, train_context_, False)
                val_log_liks.append(flow.log_prob(val_inputs_, val_context_).mean())

            tune.report(log_lik=np.mean(val_log_liks))

        result = tune.run(
            ray_fit,
            resources_per_trial=resources_per_trial,
            config=hparam_grid,
            mode='max',
            metric='log_lik',
            verbose=0,
            reuse_actors=True,
            time_budget_s=time_budget_s
        )
        ray.shutdown()

        logger.info(f"Models evaluated: {result.results_df['done'].sum()} / {len(result.results_df)}, "
                    f"Best config: {result.get_best_config()}. Refitting the best model.")
        self.__init__(self.context_size, device=self.device, **result.get_best_config())
        self.fit(train_inputs, train_context)
        return self

    def log_prob(self, inputs: Union[np.array, Tensor], context: Union[np.array, Tensor],
                 data_normalization=True) -> Union[np.array, Tensor]:
        """
        Log pdf function
        Args:
            inputs: Input
            context: Conditioning global_context
            data_normalization: Perform data normalisation

        Returns: np.array or Tensor with shape (inputs_dim, )

        """
        return_numpy = False
        if not isinstance(inputs, torch.Tensor):
            inputs, context = self._input_to_tensor(inputs, context)
            return_numpy = True

        if data_normalization:
            inputs, context = self._transform_normalise(inputs, context)

        result = super().log_prob(inputs, context)

        if self.inputs_normalization:
            result -= torch.log(self.inputs_std)

        if return_numpy:
            return result.detach().cpu().numpy()
        else:
            return result

    def conditional_distribution(self, context: Union[np.array, Tensor], data_normalization=True) -> Flow:
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32, device=self.device)

        if data_normalization:
            _, context = self._transform_normalise(None, context)

        # for cont in context:
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


    def sample(self, context: Union[np.array, Tensor], num_samples=1, data_normalization=True) -> Union[np.array, Tensor]:
        """
        Sampling from conditional distribution

        Args:
            num_samples: Number of samples per global_context
            context: Conditioning global_context
            data_normalization: Perform data normalisation

        Returns: np.array or Tensor with shape (global_context.shape[0], num_samples)

        """
        return_numpy = False
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)
            return_numpy = True

        if data_normalization:
            _, context = self._transform_normalise(None, context)

        result = super().sample(num_samples, context).squeeze(-1)

        if self.inputs_normalization:
            result, _ = self._transform_inverse_normalise(result, None)

        if return_numpy:
            return result.detach().cpu().numpy()
        else:
            return result
