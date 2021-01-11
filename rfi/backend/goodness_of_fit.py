import torch
from torch.distributions import Distribution
from nflows.flows import Flow
import numpy as np
import pandas as pd
from scipy import integrate
from tqdm import tqdm
from omegaconf import DictConfig
from typing import List, Union, Tuple
from dataclasses import dataclass
import logging

from rfi.backend.causality import StructuralEquationModel
from rfi.backend.cnf import ConditionalNormalisingFlowEstimator
from rfi.backend.gaussian.gaussian_estimator import GaussianConditionalEstimator

logger = logging.getLogger(__name__)


@dataclass
class ConditionalGoodnessOfFit:
    name: str

    # TODO Calculation not with test_df, but with fixed error tolerance
    def __call__(self, estimator: Union[ConditionalNormalisingFlowEstimator, GaussianConditionalEstimator],
                 sem: StructuralEquationModel,
                 target_var: str,
                 context_vars: Tuple[str],
                 exp_args: DictConfig,
                 test_df: pd.DataFrame = None):

        logger.info(f"Calculating {self.name} for {target_var} / {context_vars}")
        assert target_var not in context_vars

        context = {node: torch.tensor(test_df.loc[:, node]) for node in context_vars}
        context_size = len(test_df)

        logger.info("Initializing SEM conditional distributions")
        if exp_args.conditioning_mode == 'true_parents':
            data_log_prob = sem.parents_conditional_distribution(target_var, parents_context=context).log_prob
        elif exp_args.conditioning_mode == 'true_markov_blanket' or exp_args.conditioning_mode == 'all':
            data_log_prob = sem.mb_conditional_log_prob(target_var, global_context=context, **exp_args.mb_dist)

        def data_log_prob_from_np(value):
            value = torch.tensor([[value]])
            with torch.no_grad():
                return data_log_prob(value.repeat(1, context_size)).squeeze()


        logger.info("Initializing estimator's conditional distributions")
        model_cond_dist = estimator.conditional_distribution(test_df.loc[:, context_vars].values).log_prob

        def model_log_prob_from_np(value):
            value = torch.tensor([[value]])
            with torch.no_grad():
                if isinstance(estimator, ConditionalNormalisingFlowEstimator):
                    return model_cond_dist(value.repeat(context_size, 1)).squeeze()
                else:
                    return model_cond_dist(value.repeat(1, context_size)).squeeze()


        logger.info(f"Calculating integral")
        if self.name == 'conditional_kl_divergence' or self.name == 'conditional_hellinger_distance':

            def integrand(value):
                data_log_p = data_log_prob_from_np(value)
                model_log_p = model_log_prob_from_np(value)

                if self.name == 'conditional_kl_divergence':
                    res = (data_log_p - model_log_p) * data_log_p.exp()
                elif self.name == 'conditional_hellinger_distance':
                    res = (torch.sqrt(data_log_p.exp()) - torch.sqrt(model_log_p.exp())) ** 2

                res[torch.isnan(res)] = 0.0  # Out of support values
                return res.numpy()

            if self.name == 'conditional_kl_divergence':
                result = integrate.quad_vec(integrand, *sem.support_bounds, epsabs=exp_args.metrics.epsabs)[0]
            else:
                result = integrate.quad_vec(integrand, -np.inf, np.inf, epsabs=exp_args.metrics.epsabs)[0]

            if self.name == 'conditional_hellinger_distance':
                result = np.sqrt(0.5 * result)

        elif self.name == 'conditional_js_divergence':

            # functions to integrate
            def integrand1(value):
                data_log_p = data_log_prob_from_np(value)
                model_log_p = model_log_prob_from_np(value)
                log_mixture = np.log(0.5) + torch.logsumexp(torch.stack([data_log_p, model_log_p]), 0)
                res = (data_log_p - log_mixture) * data_log_p.exp()
                res[torch.isnan(res)] = 0.0  # Out of support values
                return res.numpy()

            def integrand2(value):
                data_log_p = data_log_prob_from_np(value)
                model_log_p = model_log_prob_from_np(value)
                log_mixture = np.log(0.5) + torch.logsumexp(torch.stack([data_log_p, model_log_p]), 0)
                res = (model_log_p - log_mixture) * model_log_p.exp()
                res[torch.isnan(res)] = 0.0  # Out of support values
                return res.numpy()


            result = 0.5 * (integrate.quad_vec(integrand1, -np.inf, np.inf, epsabs=exp_args.metrics.epsabs)[0] +
                            integrate.quad_vec(integrand2, -np.inf, np.inf, epsabs=exp_args.metrics.epsabs)[0])

        else:
            raise NotImplementedError()

        return result.mean()


conditional_kl_divergence = ConditionalGoodnessOfFit(name='conditional_kl_divergence')
conditional_hellinger_distance = ConditionalGoodnessOfFit(name='conditional_hellinger_distance')
conditional_js_divergence = ConditionalGoodnessOfFit(name='conditional_js_divergence')
