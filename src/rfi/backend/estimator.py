import numpy as np
import torch
from torch import Tensor
from torch import nn
from typing import Union
from sklearn.model_selection import KFold
import logging
from ray import tune
import ray
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger(__name__)


class ConditionalDistributionEstimator(nn.Module):

    default_hparam_grid = {}

    def __init__(self,
                 context_size: int = 0,
                 inputs_size: int = 1,
                 context_normalization=False,
                 inputs_normalization=False,
                 cat_context: np.array = None,
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

        # Categorical / Continuous context
        if cat_context is None or len(cat_context) == 0:
            self.cat_context = []
            self.cont_context = np.arange(0, context_size)
        else:
            self.cat_context = cat_context
            self.cont_context = np.array([i for i in range(context_size) if i not in cat_context])
            self.cont_context = [] if len(self.cont_context) == 0 else self.cont_context

        # Inputs / Context one-hot encoders
        self.context_enc = OneHotEncoder(drop='if_binary', sparse=False)
        self.inputs_enc = OneHotEncoder(sparse=False)

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a ConditionalDistributionEstimator object.")

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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
            self.context_mean, self.context_std = \
                train_context[:, :len(self.cont_context)].mean(0), train_context[:, :len(self.cont_context)].std(0)
            train_context[:, :len(self.cont_context)] = \
                (train_context[:, :len(self.cont_context)] - self.context_mean) / self.context_std
        return train_inputs, train_context

    def _transform_normalise(self, inputs: Tensor, context: Tensor):
        if inputs is not None and self.inputs_normalization:
            inputs = (inputs - self.inputs_mean) / self.inputs_std
        if context is not None and self.context_normalization:
            context[:, :len(self.cont_context)] = (context[:, :len(self.cont_context)] - self.context_mean) / self.context_std
        return inputs, context

    def _transform_inverse_normalise(self, inputs: Tensor, context: Tensor):
        if inputs is not None and self.inputs_normalization:
            inputs = inputs * self.inputs_std + self.inputs_mean
        if context is not None and self.context_normalization:
            context = context * self.context_std + self.context_mean
        return inputs, context

    def _add_noise(self, data: Tensor, std: float, non_lin=nn.Identity()) -> Tensor:
        if data is not None:
            return data + non_lin(torch.randn(data.size()).type_as(data) * std)
        else:
            return None

    def _fit_transform_onehot_encode(self, train_inputs: np.array, train_context: np.array) -> np.array:
        if len(self.cat_context) > 0 and train_context is not None:
            train_context_cat = self.context_enc.fit_transform(train_context[:, self.cat_context])
            train_context_cont = train_context[:, self.cont_context]
            train_context = np.concatenate([train_context_cont, train_context_cat], axis=1)
            self.context_size = train_context.shape[1]
        if train_inputs is not None:
            train_inputs = self.inputs_enc.fit_transform(train_inputs)
            self.inputs_size = train_inputs.shape[1]
        return train_inputs, train_context

    def _transform_onehot_encode(self, inputs: np.array, context: np.array) -> np.array:
        if len(self.cat_context) > 0 and context is not None:
            context = np.concatenate([context[:, self.cont_context],
                                      self.context_enc.transform(context[:, self.cat_context])], axis=1)
        if inputs is not None:
            inputs = self.inputs_enc.transform(inputs)
        return inputs, context

    def _inverse_onehot_encode(self, inputs: np.array):
        if inputs is not None:
            inputs = self.inputs_enc.inverse_transform(inputs)
        return inputs

    def _preprocess_data(self, inputs, context, data_normalization, context_one_hot_encoding, inputs_one_hot_ecoding):
        return_numpy = False

        if inputs_one_hot_ecoding:
            inputs = inputs.reshape(-1, 1)
            inputs, _ = self._transform_onehot_encode(inputs, None)

        if context_one_hot_encoding:
            _, context = self._transform_onehot_encode(None, context)

        if not isinstance(inputs, torch.Tensor):
            inputs, context = self._input_to_tensor(inputs, context)
            return_numpy = True

        if data_normalization:
            inputs, context = self._transform_normalise(inputs, context)

        return inputs, context, return_numpy

    def _postprocess_result(self, result, return_numpy):
        if return_numpy:
            return result.detach().cpu().numpy()
        else:
            return result

    def fit_by_cv(self,
                  train_inputs: Union[np.array, Tensor],
                  train_context: Union[np.array, Tensor] = None,
                  hparam_grid=None,
                  n_splits=5,
                  resources_per_trial={"cpu": 0.33},
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

                estimator = cls(inputs_size=self.inputs_size, context_size=self.context_size, device=self.device,
                                base_distribution=self._distribution if hasattr(self, '_distribution') else None,
                                inputs_noise_nonlinearity=(self.inputs_noise_nonlinearity
                                                           if hasattr(self, 'inputs_noise_nonlinearity') else None),
                                context_normalization=self.context_normalization,
                                inputs_normalization=self.inputs_normalization,
                                batch_size=self.batch_size, cat_context=self.cat_context, **config)
                estimator.fit(train_inputs_, train_context_, False)
                val_log_liks.append(estimator.log_prob(val_inputs_, val_context_).mean())

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
        self.__init__(self.context_size, self.inputs_size, device=self.device,
                      base_distribution=self._distribution if hasattr(self, '_distribution') else None,
                      inputs_noise_nonlinearity=(self.inputs_noise_nonlinearity
                                                 if hasattr(self, 'inputs_noise_nonlinearity') else None),
                      context_normalization=self.context_normalization,
                      inputs_normalization=self.inputs_normalization,
                      batch_size=self.batch_size, cat_context=self.cat_context,
                      **result.get_best_config())
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
