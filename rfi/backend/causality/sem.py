import networkx as nx
from typing import Dict, Tuple
import collections
import numpy as np
import torch
from torch import Tensor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pyro.contrib.randomvariable import RandomVariable

from rfi.backend.causality.dags import DirectedAcyclicGraph


class StructuralEquationModel:
    """
    Semi-abstract class for Structural Equation Model, defined by DAG. Specific assignments should be redefined in subclasses
    """

    def __init__(self, dag: DirectedAcyclicGraph):
        """
        Args:
            dag: DAG, defining SEM
        """
        # Model init
        self.dag = dag
        self.model = {}
        self.topological_order = []

        # Model and topological order init
        for node in nx.algorithms.dag.topological_sort(dag.DAG):
            parents = dag.DAG.predecessors(node)
            node, parents = dag.var_names[node], [dag.var_names[par] for par in parents]

            self.model[node] = {
                'parents': parents,
                'value': None
            }
            self.topological_order.append(node)

    def sample(self, size: int, seed: int = None) -> Tensor:
        """
        Returns a sample from SEM, columns are ordered as var input_var_names in DAG
        Args:
            seed: Random seed
            size: Sample size

        Returns: torch.Tensor with sampled values of shape (size, n_vars))
        """
        raise NotImplementedError()

    def conditional_pdf(self, node: str, value: Tensor, context: Dict[str, Tensor]) -> Tensor:
        """
        Conditional probability distribution function (density) of node
        Args:
            node: Node
            value: Input values, torch.Tensor of shape (n, )
            context: Conditioning context, as dict of all the other variables (only parent values will be chosen), each value
                should be torch.Tensor of shape (n, )

        Returns: torch.Tensor of shape (n, )
        """
        raise NotImplementedError()


class LinearGaussianNoiseSEM(StructuralEquationModel):
    """
    Class for modelling Linear SEM with Additive Gaussian noise:
    X_i = w^T par(X_i) + eps, eps ~ N(0, sigma_i)
    """

    def __init__(self, dag: DirectedAcyclicGraph, coeff_dict: Dict[str, Dict[str, float]] = {},
                 noise_std_dict: Dict[str, float] = {}, default_noise_std_bounds: Tuple[float, float] = (0.0, 1.0), seed=None):
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
            self.model[node]['coeff'] = \
                coeff_dict[node] if node in coeff_dict else {par: 0.0 for par in self.model[node]['parents']}
            self.model[node]['noise_std'] = noise_std_dict[node] if node in noise_std_dict \
                else np.random.uniform(*default_noise_std_bounds)

    @staticmethod
    def _linear_comb(coeffs_dict, parent_values_dict) -> Tensor:
        result = 0.0
        for par in coeffs_dict.keys():
            result += torch.tensor(parent_values_dict[par], dtype=torch.float32) * \
                      torch.tensor(coeffs_dict[par], dtype=torch.float32)
        return torch.tensor(result)


    def sample(self, size=1, seed=None) -> Tensor:
        torch.manual_seed(seed) if seed is not None else None
        for node in self.topological_order:
            # Sampling noise
            noise = torch.distributions.Normal(0, self.model[node]['noise_std']).rsample(sample_shape=(size,))

            # Linear combination of parent values
            parent_values = {par_node: self.model[par_node]['value'] for par_node in self.model[node]['parents']}
            linear_effect = self._linear_comb(coeffs_dict=self.model[node]['coeff'], parent_values_dict=parent_values)

            self.model[node]['value'] = linear_effect + noise

        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)

    def conditional_pdf(self, node: str, value: Tensor, context: Dict[str, Tensor] = {}) -> Tensor:
        result = torch.zeros((len(value), ))
        for val_ind, val in enumerate(value):
            parent_values = {par_node: context[par_node][val_ind] for par_node in self.model[node]['parents']}
            linear_comb = self._linear_comb(coeffs_dict=self.model[node]['coeff'], parent_values_dict=parent_values)
            cond_dist = torch.distributions.Normal(linear_comb, self.model[node]['noise_std'])
            result[val_ind] = cond_dist.log_prob(val).exp()
        return result

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

    def __init__(self, dag: DirectedAcyclicGraph, bandwidth: float = 1.0, noise_std_dict: Dict[str, float] = {},
                 default_noise_std_bounds: Tuple[float, float] = (0.0, 1.0), seed=None):
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

        self.bandwidth = bandwidth

    def _random_function(self, node: str, new_parent_values: collections.OrderedDict, seed=None) -> Tensor:
        new_parent_values = list(new_parent_values.values())
        if len(new_parent_values) == 0:
            return torch.tensor(0.0)

        old_parent_values = self.model[node]['stacked_parent_values']
        old_random_effects = self.model[node]['stacked_random_effects']
        new_parent_values = torch.stack(new_parent_values).T

        gp = GaussianProcessRegressor(kernel=RBF(length_scale=self.bandwidth), optimizer=None)

        if old_parent_values is None and old_random_effects is None:  # Sampling for the first time
            node_values = torch.tensor(gp.sample_y(new_parent_values, random_state=seed)).reshape(-1)

            self.model[node]['stacked_random_effects'] = node_values
            self.model[node]['stacked_parent_values'] = new_parent_values
            return node_values

        else:  # Sampling, when random function was already evaluated
            gp.fit(old_parent_values, old_random_effects)
            node_values = torch.tensor(gp.sample_y(new_parent_values, random_state=seed)).reshape(-1)

            self.model[node]['stacked_random_effects'] = torch.cat([old_random_effects, node_values])
            self.model[node]['stacked_parent_values'] = torch.cat([old_parent_values, new_parent_values], dim=0)
            return node_values


    def sample(self, size, seed=None) -> Tensor:
        torch.manual_seed(seed) if seed is not None else None

        for node in self.topological_order:
            # Sampling noise
            noise = torch.distributions.Normal(0, self.model[node]['noise_std']).rsample(sample_shape=(size,))

            # Random function of parent values
            new_parent_values = collections.OrderedDict([(par_node, self.model[par_node]['value'])
                                                         for par_node in self.model[node]['parents']])
            random_effect = self._random_function(node, new_parent_values, seed=seed)
            self.model[node]['value'] = random_effect + noise

        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)

    def conditional_pdf(self, node: str, value: Tensor, context: Dict[str, Tensor] = {}) -> Tensor:
        result = torch.zeros((len(value), ))
        for val_ind, val in enumerate(value):
            new_parent_values = collections.OrderedDict([(par_node, context[par_node][val_ind].unsqueeze(-1))
                                                         for par_node in self.model[node]['parents']])
            random_effect = self._random_function(node=node, new_parent_values=new_parent_values)
            cond_dist = torch.distributions.Normal(random_effect, self.model[node]['noise_std'])
            result[val_ind] = cond_dist.log_prob(val).exp()
        return result


class PostNonLinearLaplaceSEM(RandomGPGaussianNoiseSEM):
    """
    Class for modelling Post Non-Linear SEM with Additive Laplacian noise. Random function is sampled from GP with RBF
    X_i = g(f(par(X_i)) + eps), eps ~ Laplace(0, sigma_i) and f ~ GP(0, K(X, X'))
    For possible post non-linearities g, see:
    See http://docs.pyro.ai/en/stable/_modules/pyro/contrib/randomvariable/random_variable.html#RandomVariable.transform
    """

    def __init__(self, dag: DirectedAcyclicGraph, bandwidth: float = 1.0, noise_std_dict: Dict[str, float] = {},
                 default_noise_std_bounds: Tuple[float, float] = (0.0, 1.0), seed=None, invertable_nonlinearity: str = 'sigmoid'):
        """
        Args:
            invertable_nonlinearity: Post noise addition non-linearity
            bandwidth: Bandwidth of GP kernel
            dag: DAG, defining SEM
            noise_std_dict: Noise std dict for each variable
            default_noise_std: Default noise std
        """
        super(PostNonLinearLaplaceSEM, self).__init__(dag, bandwidth, noise_std_dict, default_noise_std_bounds, seed)
        self.invertable_nonlinearity = invertable_nonlinearity

    def sample(self, size, seed=None) -> Tensor:
        torch.manual_seed(seed) if seed is not None else None

        for node in self.topological_order:
            # Sampling noise
            noise = torch.distributions.Laplace(0, self.model[node]['noise_std']).rsample(sample_shape=(size,))

            # Random function of parent values
            new_parent_values = collections.OrderedDict([(par_node, self.model[par_node]['value'])
                                                         for par_node in self.model[node]['parents']])
            random_effect = self._random_function(node, new_parent_values, seed=seed)

            nonlinearity = getattr(torch, self.invertable_nonlinearity)
            self.model[node]['value'] = nonlinearity(random_effect + noise)

        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)

    def conditional_pdf(self, node: str, value: Tensor, context: Dict[str, Tensor] = {}) -> Tensor:
        result = torch.zeros((len(value), ))
        for val_ind, val in enumerate(value):
            new_parent_values = collections.OrderedDict([(par_node, context[par_node][val_ind].unsqueeze(-1))
                                                         for par_node in self.model[node]['parents']])
            random_effect = self._random_function(node=node, new_parent_values=new_parent_values)

            noise = RandomVariable(torch.distributions.Laplace(0, self.model[node]['noise_std']))
            cond_dist = getattr(random_effect + noise, self.invertable_nonlinearity)().dist

            assert cond_dist.support.check(val)  # Check, if evaluated value falls to conditional distribution support

            result[val_ind] = cond_dist.log_prob(val).exp()

        return result


class PostNonLinearMultiplicativeHalfNormalSEM(StructuralEquationModel):

    def __init__(self, dag: DirectedAcyclicGraph, noise_std_dict: Dict[str, float] = {},
                 default_noise_std_bounds: Tuple[float, float] = (0.0, 1.0), seed=None):
        """
        Args:
            dag: DAG, defining SEM
            coeff_dict: Coefficients of linear combination of parent nodes, if not written - considered as zero
            noise_std_dict: Noise std dict for each variable
            default_noise_std: Default noise std
        """
        super(PostNonLinearMultiplicativeHalfNormalSEM, self).__init__(dag)

        np.random.seed(seed) if seed is not None else None

        for node in self.topological_order:
            self.model[node]['noise_std'] = noise_std_dict[node] if node in noise_std_dict \
                else np.random.uniform(*default_noise_std_bounds)

    def sample(self, size, seed=None) -> Tensor:
        torch.manual_seed(seed) if seed is not None else None

        for node in self.topological_order:
            # Sampling noise
            noise = torch.distributions.HalfNormal(self.model[node]['noise_std']).rsample(sample_shape=(size,))
            parent_values = [self.model[par_node]['value'] for par_node in self.model[node]['parents']]
            parent_values = torch.stack(parent_values).T if len(parent_values) > 0 else torch.tensor([[1.0]])

            self.model[node]['value'] = torch.exp(torch.log(parent_values.sum(dim=1)) + noise)

        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)

    def conditional_pdf(self, node: str, value: Tensor, context: Dict[str, Tensor] = {}) -> Tensor:
        result = torch.zeros((len(value), ))
        for val_ind, val in enumerate(value):

            parent_values = [context[par_node][val_ind].unsqueeze(-1) for par_node in self.model[node]['parents']]
            parent_values = torch.stack(parent_values).T if len(parent_values) > 0 else torch.tensor([[1.0]])
            noise = RandomVariable(torch.distributions.HalfNormal(self.model[node]['noise_std']))

            cond_dist = (torch.log(parent_values.sum(dim=1)) + noise).exp().dist

            assert cond_dist.support.check(val)  # Check, if evaluated value falls to conditional distribution support

            result[val_ind] = cond_dist.log_prob(val).exp()

        return result
