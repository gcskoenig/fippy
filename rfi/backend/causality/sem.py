import networkx as nx
from typing import Dict, Tuple, List, Union
import collections
import numpy as np
import torch
from torch.distributions import Distribution, Normal, MultivariateNormal, constraints, TransformedDistribution
from scipy.integrate import quad, quad_vec
from scipy.interpolate import LinearNDInterpolator, interp1d, RegularGridInterpolator
from torch import Tensor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pyro.contrib.randomvariable import RandomVariable
import logging
from tqdm import tqdm

from rfi.backend.causality.dags import DirectedAcyclicGraph
from rfi.backend.gaussian import GaussianConditionalEstimator
from rfi.utils import search_nonsorted

logger = logging.getLogger(__name__)


class StructuralEquationModel:
    """
    Semi-abstract class for Structural Equation Model, defined by DAG. Specific assignments should be redefined in subclasses
    """

    name = 'sem'
    is_batched = False

    def __init__(self, dag: DirectedAcyclicGraph):
        """
        Args:
            dag: DAG, defining SEM
        """
        # Model init
        self.dag = dag
        self.model = {}
        self.topological_order = []
        self.parents_conditional_support = constraints.real

        # Model and topological order init
        for node in nx.algorithms.dag.topological_sort(dag.DAG):
            parents = dag.DAG.predecessors(node)
            children = dag.DAG.successors(node)
            node, parents, children = dag.var_names[node], [dag.var_names[par] for par in parents], \
                [dag.var_names[par] for par in children]

            self.model[node] = {
                'parents': tuple(parents),
                'children_nodes': tuple(children),
                'value': None
            }
            self.topological_order.append(node)

    @property
    def support_bounds(self):
        support_bounds = [-np.inf, np.inf]
        if hasattr(self.parents_conditional_support, 'lower_bound'):
            support_bounds[0] = self.parents_conditional_support.lower_bound
        if hasattr(self.parents_conditional_support, 'upper_bound'):
            support_bounds[1] = self.parents_conditional_support.upper_bound
        return support_bounds

    def parents_conditional_sample(self, node: str, parents_context: Dict[str, Tensor] = None, sample_shape: tuple = (1, )) \
            -> Tensor:
        cond_dist = self.parents_conditional_distribution(node, parents_context)
        return cond_dist.sample(sample_shape=sample_shape)

    def sample(self, size: int, seed: int = None) -> Tensor:
        """
        Returns a sample from SEM, columns are ordered as var input_var_names in DAG
        Args:
            seed: Random mc_seed
            size: Sample size

        Returns: torch.Tensor with sampled value of shape (size, n_vars))
        """
        torch.manual_seed(seed) if seed is not None else None
        for node in self.topological_order:
            parent_values = {par_node: self.model[par_node]['value'] for par_node in self.model[node]['parents']}
            if len(parent_values) == 0:
                sample_shape = (size,)
            elif self.is_batched:
                sample_shape = (1, )
            else:
                sample_shape = (1, size)
            self.model[node]['value'] = self.parents_conditional_sample(node, parent_values, sample_shape).reshape(-1)
            assert (~self.model[node]['value'].isnan()).all()
        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)

    @staticmethod
    def _support_check_wrapper(dist: Distribution):

        old_log_prob = dist.log_prob

        def new_log_prob_method(value):
            result = old_log_prob(value)

            # Out of support values
            result[torch.isnan(result)] = - float("Inf")
            result[~dist.support.check(value)] = - float("Inf")

            return result

        dist.log_prob = new_log_prob_method

    def parents_conditional_distribution(self, node: str, parents_context: Dict[str, Tensor] = None) -> Distribution:
        """
        Conditional probability distribution of node, conditioning only on parent nodes
        Args:
            node: Node
            parents_context: Conditioning global_context, as dict of all the other variables,
                    each value should be torch.Tensor of shape (n, )

        Returns: torch.distribution.Distribution with batch_shape == n
        """
        raise NotImplementedError()

    def joint_log_prob(self, value: Tensor) -> Tensor:
        global_context = {node: value[:, node_ind] for node_ind, node in enumerate(self.dag.var_names)}

        log_prob = torch.zeros_like(value)
        for node in self.topological_order:
            parent_values = {par_node: global_context[par_node] for par_node in self.model[node]['parents']}
            log_prob += self.parents_conditional_distribution(node, parents_context=parent_values).log_prob(global_context[node])
        return log_prob

    def get_markov_blanket(self, node: str) -> set:
        return self.dag.get_markov_blanket(node)

    def children_log_prob(self, node, value: Tensor, global_context: Dict[str, Tensor] = None):
        log_prob = torch.zeros_like(value)
        for child in self.model[node]['children_nodes']:
            child_par_context = {par: global_context[par].repeat(len(value))
                                 for par in self.model[child]['parents'] if par != node}
            child_par_context[node] = value.flatten()
            cond_dist = self.parents_conditional_distribution(child, parents_context=child_par_context)

            child_log_prob = cond_dist.log_prob(global_context[child].repeat(len(value)))

            log_prob += child_log_prob.reshape(value.shape)

        return log_prob

    def mb_conditional_log_prob(self, node: str, global_context: Dict[str, Tensor] = None,
                                method='mc', mc_size=500, quad_epsabs=1e-2, mc_seed=None) -> callable:
        """
        Conditional log-probability function (density) of node, conditioning on Markov-Blanket of node
        Args:
            node: Node
            global_context: Conditioning global_context, as dict of all the other variables,
                each value should be torch.Tensor of shape (n, )
            method: 'mc' or 'quad', method to compute normalisation constant (Monte-Carlo integration or quadrature integration)
            mc_size: if method == 'mc', MC samples for integration
            mc_seed: mc_seed for MC sampling
            quad_epsabs: epsabs for scipy.integrate.quad
        Returns:

        """
        global_context = {} if global_context is None else global_context

        # Checking, if all vars from MarkovBlanket(node) are present
        assert all([mb_var in global_context.keys() for mb_var in self.get_markov_blanket(node)])

        parents_context = {par: global_context[par] for par in self.model[node]['parents']}
        context_size = len(list(global_context.values())[0])

        if method == 'mc':
            torch.manual_seed(mc_seed) if mc_seed is not None else None
            sample_shape = (mc_size, 1) if len(parents_context) != 0 and self.is_batched else (mc_size, context_size)
            sampled_value = self.parents_conditional_sample(node, parents_context, sample_shape)
            normalisation_constant = self.children_log_prob(node, sampled_value, global_context).exp().mean(0)

        elif method == 'quad':
            normalisation_constant = []
            for cont_ind in tqdm(range(context_size)):
                global_context_slice = {node: value[cont_ind:cont_ind + 1] for (node, value) in global_context.items()}
                parents_context_slice = {par: global_context_slice[par] for par in self.model[node]['parents']}

                def integrand(value):
                    value = torch.tensor([value])
                    parents_cond_dist = self.parents_conditional_distribution(node, parents_context_slice)
                    unnorm_log_prob = parents_cond_dist.log_prob(value) + \
                        self.children_log_prob(node, value, global_context_slice)
                    return unnorm_log_prob.exp().item()

                normalisation_constant.append(quad_vec(integrand, *self.support_bounds, epsabs=quad_epsabs)[0])

            normalisation_constant = torch.tensor(normalisation_constant)

        else:
            raise NotImplementedError()

        def cond_log_prob(value):
            value = value.reshape(-1, context_size)
            result = self.parents_conditional_distribution(node, parents_context).log_prob(value)

            # Considering only inside-support values for conditional distributions of children_nodes
            in_support = (~torch.isinf(result))
            for val_ind, val in enumerate(value):
                global_context_tile = {n: v[in_support[val_ind]] for (n, v) in global_context.items()}
                children_lob_prob = self.children_log_prob(node, val[in_support[val_ind]].unsqueeze(0), global_context_tile)
                result[val_ind, in_support[val_ind]] += children_lob_prob.squeeze()

            return result - normalisation_constant.log()

        return cond_log_prob


class LinearGaussianNoiseSEM(StructuralEquationModel):
    """
    Class for modelling Linear SEM with Additive Gaussian noise:
    X_i = w^T par(X_i) + eps, eps ~ N(0, sigma_i)
    """

    name = 'linear_gauss'
    is_batched = True

    def __init__(self, dag: DirectedAcyclicGraph,
                 coeff_dict: Dict[str, Dict[str, float]] = {},
                 noise_std_dict: Dict[str, float] = {},
                 default_coef: Union[float, Tuple[float]] = (0.0, 1.0),
                 default_noise_std_bounds: Tuple[float, float] = (0.0, 1.0),
                 seed=None):
        """
        Args:
            dag: DAG, defining SEM
            coeff_dict: Coefficients of linear combination of parent nodes, if not written - considered as zero
            noise_std_dict: Noise std dict for each variable
            default_noise_std_bounds: Default noise std, if not specified in noise_std_dict.
                Sampled from U(default_noise_std_bounds[0], default_noise_std_bounds[1])
        """
        super(LinearGaussianNoiseSEM, self).__init__(dag)

        np.random.seed(seed) if seed is not None else None

        for node in self.topological_order:
            if node in coeff_dict:
                self.model[node]['coeff'] = coeff_dict[node]
            else:
                if type(default_coef) == float:
                    self.model[node]['coeff'] = {par: default_coef for par in self.model[node]['parents']}
                else:
                    self.model[node]['coeff'] = {par: np.random.uniform(*default_coef) for par in self.model[node]['parents']}
            self.model[node]['noise_std'] = noise_std_dict[node] if node in noise_std_dict \
                else np.random.uniform(*default_noise_std_bounds)

    def conditional_distribution(self, node: str, context: Dict[str, Tensor] = None) -> Distribution:
        node_ind = search_nonsorted(self.dag.var_names, [node])
        if context is None or len(context) == 0:  # Unconditional distribution
            return Normal(self.joint_mean[node_ind].item(), torch.sqrt(self.joint_cov[node_ind, node_ind]).item())
        else:  # Conditional distribution
            context_ind = search_nonsorted(self.dag.var_names, list(context.keys()))
            cond_dist = GaussianConditionalEstimator()
            cond_dist.fit_mean_cov(self.joint_mean.numpy(), self.joint_cov.numpy(), inp_ind=node_ind, cont_ind=context_ind)
            context_sorted = [context[par_node] for par_node in context.keys()]
            context_sorted = np.stack(context_sorted).T if len(context_sorted) > 0 else None
            return cond_dist.conditional_distribution(context_sorted)

    def parents_conditional_distribution(self, node: str, parents_context: Dict[str, Tensor] = None) -> Distribution:
        # Only conditioning on parent nodes is possible for now
        assert set(self.model[node]['parents']) == set(parents_context.keys())
        linear_comb = 0.0
        if len(self.model[node]['parents']) > 0:
            for par in parents_context.keys():
                linear_comb += parents_context[par] * torch.tensor(self.model[node]['coeff'][par])
        return Normal(linear_comb, self.model[node]['noise_std'])

    @property
    def joint_cov(self) -> Tensor:
        """
        Covariance of joint DAG distribution, could be used for analytic RFI calculation

        Returns: torch.Tensor of shape (n_vars, n_vars)
        """
        # AX = eps   =>   X = A^-1 eps  =>  Cov(X) =  A^-1 Cov(eps) A^-1.T
        A = np.zeros((len(self.dag.var_names), len(self.dag.var_names)))
        noise_var = np.empty((len(self.dag.var_names),))
        for i, node in enumerate(self.dag.var_names):
            for j, sub_node in enumerate(self.dag.var_names):
                if node in self.model[sub_node]['parents']:
                    A[j, i] = - self.model[sub_node]['coeff'][node]
                if i == j:
                    A[i, j] = 1.0
            noise_var[i] = self.model[node]['noise_std'] ** 2
        return torch.tensor(np.linalg.inv(A) @ np.diag(noise_var) @ np.linalg.inv(A).T, dtype=torch.float32)

    @property
    def joint_mean(self) -> Tensor:
        """
        Mean of joint DAG distribution, could be used for analytic RFI calculation

        Returns: torch.Tensor of shape (n_vars, )
        """
        # AX = eps  =>   X = A^-1 eps  =>  E(X) = A^-1 E(eps) = 0
        return torch.zeros((len(self.dag.var_names), ), dtype=torch.float32)

    @property
    def joint_distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.MultivariateNormal(self.joint_mean, self.joint_cov)


class RandomGPGaussianNoiseSEM(StructuralEquationModel):
    """
    Class for modelling Non-Linear SEM with Additive Gaussian noise. Random function is sampled from GP with RBF
    X_i = f(par(X_i)) + eps, eps ~ N(0, sigma_i) and f ~ GP(0, K(X, X'))
    """

    name = 'gauss_anm'
    is_batched = True

    def __init__(self, dag: DirectedAcyclicGraph, bandwidth: float = 1.0, noise_std_dict: Dict[str, float] = {},
                 default_noise_std_bounds: Tuple[float, float] = (0.0, 1.0), seed=None, interpolation_switch=300):
        """
        Args:
            bandwidth: Bandwidth of GP kernel
            dag: DAG, defining SEM
            noise_std_dict: Noise std dict for each variable
            default_noise_std_bounds: Default noise std, if not specified in noise_std_dict.
                Sampled from U(default_noise_std_bounds[0], default_noise_std_bounds[1])
        """
        super(RandomGPGaussianNoiseSEM, self).__init__(dag)

        np.random.seed(seed) if seed is not None else None

        for node in self.topological_order:
            self.model[node]['noise_std'] = noise_std_dict[node] if node in noise_std_dict \
                else np.random.uniform(*default_noise_std_bounds)
            self.model[node]['stacked_parent_values'] = None  # Needed for repetitive sampling
            self.model[node]['stacked_random_effects'] = None  # Needed for repetitive sampling
            self.model[node]['interpolator'] = None

        self.bandwidth = bandwidth
        self.interpolation_switch = interpolation_switch

    def _random_function(self, node: str, parent_values: Dict[str, Tensor], seed=None) -> Tensor:

        # Ordering
        parent_values = collections.OrderedDict([(par_node, parent_values[par_node])
                                                 for par_node in self.model[node]['parents']])

        parent_values = list(parent_values.values())
        if len(parent_values) == 0:
            return torch.tensor(0.0)

        old_parent_values = self.model[node]['stacked_parent_values']
        old_random_effects = self.model[node]['stacked_random_effects']
        parent_values = torch.stack(parent_values).T

        gp = GaussianProcessRegressor(kernel=RBF(length_scale=self.bandwidth), optimizer=None)

        if old_parent_values is None and old_random_effects is None:  # Sampling for the first time
            node_values = torch.tensor(gp.sample_y(parent_values, random_state=seed)).reshape(-1)

            self.model[node]['stacked_random_effects'] = node_values
            self.model[node]['stacked_parent_values'] = parent_values
            return node_values

        else:  # Sampling, when random function was already evaluated
            if len(old_random_effects) >= self.interpolation_switch:
                if self.model[node]['interpolator'] is None:
                    logger.warning('Using interpolation instead of Gaussian process!')
                    if old_parent_values.shape[1] == 1:
                        interpolator = interp1d(old_parent_values.squeeze(), old_random_effects, fill_value=0.0,
                                                bounds_error=False)
                    else:
                        interpolator = LinearNDInterpolator(old_parent_values.squeeze(), old_random_effects, fill_value=0.0)
                    self.model[node]['interpolator'] = interpolator

                # else:
                #     interpolator = LinearNDInterpolator(old_parent_values, old_random_effects, fill_value=0.0)
                return torch.tensor(self.model[node]['interpolator'](parent_values.squeeze()))
            else:
                gp.fit(old_parent_values, old_random_effects)
                node_values = torch.tensor(gp.sample_y(parent_values, random_state=seed)).reshape(-1)

                self.model[node]['stacked_random_effects'] = torch.cat([old_random_effects, node_values])
                self.model[node]['stacked_parent_values'] = torch.cat([old_parent_values, parent_values], dim=0)
                return node_values

    def parents_conditional_distribution(self, node: str, parents_context: Dict[str, Tensor] = None) -> Distribution:
        # Only conditioning on parent nodes is possible for now
        assert set(self.model[node]['parents']) == set(parents_context.keys())
        random_effect = self._random_function(node, parents_context)
        cond_dist = torch.distributions.Normal(random_effect, self.model[node]['noise_std'])
        self._support_check_wrapper(cond_dist)
        return cond_dist


class PostNonLinearLaplaceSEM(RandomGPGaussianNoiseSEM):
    """
    Class for modelling Post Non-Linear SEM with Additive Laplacian noise. Random function is sampled from GP with RBF
    X_i = g(f(par(X_i)) + eps), eps ~ Laplace(0, sigma_i) and f ~ GP(0, K(X, X'))
    For possible post non-linearities g, see:
    See http://docs.pyro.ai/en/stable/_modules/pyro/contrib/randomvariable/random_variable.html#RandomVariable.transform
    """

    name = 'post_nonlin'
    is_batched = False

    def __init__(self, dag: DirectedAcyclicGraph, bandwidth: float = 1.0, noise_std_dict: Dict[str, float] = {},
                 default_noise_std_bounds: Tuple[float, float] = (0.0, 1.0), seed=None, invertable_nonlinearity: str = 'sigmoid',
                 interpolation_switch=300):
        """
        Args:
            invertable_nonlinearity: Post noise addition non-linearity
            bandwidth: Bandwidth of GP kernel
            dag: DAG, defining SEM
            noise_std_dict: Noise std dict for each variable
            default_noise_std_bounds: Default noise std
        """
        super(PostNonLinearLaplaceSEM, self).__init__(dag, bandwidth, noise_std_dict, default_noise_std_bounds, seed,
                                                      interpolation_switch)
        self.invertable_nonlinearity = invertable_nonlinearity
        if self.invertable_nonlinearity == 'sigmoid':
            self.parents_conditional_support = constraints.unit_interval

    def parents_conditional_distribution(self, node: str, parents_context: Dict[str, Tensor] = None) -> Distribution:
        # Only conditioning on parent nodes is possible for now
        assert set(self.model[node]['parents']) == set(parents_context.keys())

        # Checking, if context is in the support
        if parents_context is not None:
            for cont in parents_context.values():
                if not all(self.parents_conditional_support.check(cont)):
                    logger.warning('Out of support value in context!')

        random_effect = self._random_function(node, parents_context)

        noise = RandomVariable(torch.distributions.Laplace(0, self.model[node]['noise_std']))
        cond_dist = getattr(random_effect + noise, self.invertable_nonlinearity)().dist
        self._support_check_wrapper(cond_dist)
        return cond_dist


class PostNonLinearMultiplicativeHalfNormalSEM(StructuralEquationModel):

    name = 'post_nonlin_half_norm'
    is_batched = False

    def __init__(self, dag: DirectedAcyclicGraph, noise_std_dict: Dict[str, float] = {},
                 default_noise_std_bounds: Tuple[float, float] = (0.5, 1.0), seed=None):
        """
        Args:
            dag: DAG, defining SEM
            noise_std_dict: Noise std dict for each variable
            default_noise_std_bounds: Default noise std
        """
        super(PostNonLinearMultiplicativeHalfNormalSEM, self).__init__(dag)

        np.random.seed(seed) if seed is not None else None

        for node in self.topological_order:
            self.model[node]['noise_std'] = noise_std_dict[node] if node in noise_std_dict \
                else np.random.uniform(*default_noise_std_bounds)

        self.parents_conditional_support = constraints.greater_than(0.0)

    def parents_conditional_distribution(self, node: str, parents_context: Dict[str, Tensor] = None) -> Distribution:
        # Only conditioning on parent nodes is possible for now
        assert set(self.model[node]['parents']) == set(parents_context.keys())

        parent_values = [parents_context[par_node] for par_node in self.model[node]['parents']]
        parent_values = torch.stack(parent_values).T if len(parent_values) > 0 else torch.ones((1, 1))

        # Checking, if context is in the support
        if not self.parents_conditional_support.check(parent_values).all():
            logger.warning('Out of support value in context!')

        log_sums = torch.log(parent_values.sum(dim=1))

        noise = RandomVariable(torch.distributions.HalfNormal(self.model[node]['noise_std']))
        cond_dist = (log_sums + noise).exp().dist
        self._support_check_wrapper(cond_dist)
        return cond_dist
