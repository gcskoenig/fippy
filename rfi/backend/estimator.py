import numpy as np
import torch
from torch import Tensor
from torch import nn
from typing import Union
from sklearn.model_selection import KFold
import logging
from ray import tune
import ray

logger = logging.getLogger(__name__)


class ConditionalDistributionEstimator(nn.Module):

    default_hparam_grid = {}

    def __init__(self,
                 context_size: int = 0,
                 inputs_size: int = 1,
                 context_normalization=False,
                 inputs_normalization=False,
                 **kwargs):
        super().__init__()
        self.inputs_size = inputs_size
        self.context_size = context_size
        self.device = 'cpu'

        # Normalisation
        self.context_normalization = context_normalization
        self.inputs_normalization = inputs_normalization
        self.inputs_mean, self.inputs_std = None, None
        self.context_mean, self.context_std = None, None

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a ConditionalDistributionEstimator object.")

    def _input_to_tensor(self, inputs: np.array, context: np.array) -> [Tensor, Tensor]:
        if inputs is not None:
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device).reshape(-1, self.inputs_size)
        if context is not None:
            context = torch.tensor(context, dtype=torch.float32, device=self.device).reshape(-1, self.context_size)
        return inputs, context

    def _init_optimizer(self, lr, weight_decay):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, betas=(0.9, 0.99), lr=lr)

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

    @staticmethod
    def _add_noise(data: Tensor, std: float) -> Tensor:
        if data is not None:
            return data + torch.randn(data.size()).type_as(data) * std
        else:
            return None

    def fit_by_cv(self,
                  train_inputs: Union[np.array, Tensor],
                  train_context: Union[np.array, Tensor] = None,
                  hparam_grid=None,
                  n_splits=5,
                  resources_per_trial={"cpu": 0.5},
                  time_budget_s=None,
                  num_cpus=15,
                  **kwargs):
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

        cls = self.__class__

        if hparam_grid is None:
            hparam_grid = cls.default_hparam_grid

        def ray_fit(config):
            val_log_liks = []
            splitter = KFold(n_splits=n_splits)
            for (train_ind, val_ind) in splitter.split(X=train_context, y=train_inputs):
                train_inputs_, train_context_ = train_inputs[train_ind], train_context[train_ind]
                val_inputs_, val_context_ = train_inputs[val_ind], train_context[val_ind]

                flow = cls(inputs_size=self.inputs_size, context_size=self.context_size, device=self.device, **config)
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
        self.__init__(self.context_size, self.inputs_size, device=self.device, **result.get_best_config())
        self.fit(train_inputs, train_context)
        return self

    def fit(self, train_inputs: Union[np.array, Tensor], train_context: [np.array, Tensor], verbose=False,
            val_inputs: Union[np.array, Tensor] = None, val_context: Union[np.array, Tensor] = None):
        raise NotImplementedError()

    def log_prob(self, inputs: Union[np.array, Tensor], context: Union[np.array, Tensor] = None) -> Union[np.array, Tensor]:
        raise NotImplementedError()

    def conditional_distribution(self, context: Union[np.array, Tensor] = None) -> torch.distributions.Distribution:
        raise NotImplementedError()

    def sample(self, context: Union[np.array, Tensor] = None, num_samples: int = 1) -> Union[np.array, Tensor]:
        raise NotImplementedError()
