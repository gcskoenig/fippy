import networkx as nx
from typing import Dict
import collections
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from rfi.backend.causality.dags import DirectedAcyclicGraph

class StructuralEquationModel:
    """
    Semi-abstract class for Structural Equation Model, defined by DAG. Specific assignments should be redefined in subclasses
    """

    def __init__(self, dag: DirectedAcyclicGraph):
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

    def sample(self, n, seed=None) -> torch.Tensor:
        """
        Returns a sample from SEM, columns are ordered as var names in DAG
        Args:
            seed: Random seed
            n: Sample size

        Returns: torch.tensor with sampled values, shape (n, len(self.dag.var_names)))
        """
        raise NotImplementedError()

    def conditional_pdf(self, node: str, value: torch.Tensor, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class LinearGaussianNoiseSEM(StructuralEquationModel):
    """
    Class for modelling Linear SEM with Additive Gaussian noise:
    X_i = w^T par(X_i) + eps, eps ~ N(0, sigma_i)
    """

    def __init__(self, dag: DirectedAcyclicGraph, coeff_dict: Dict[str, Dict[str, float]] = {},
                 noise_std_dict: Dict[str, float] = {}, default_noise_std: float = 0.0):
        """
        Args:
            dag: DAG, defining SEM
            coeff_dict: Coefficients of linear combination of parent nodes, if not written - considered as zero
            noise_std_dict: Noise std
            default_noise_std: Default noise std
        """
        super(LinearGaussianNoiseSEM, self).__init__(dag)

        for node in self.topological_order:
            self.model[node]['coeff'] = \
                coeff_dict[node] if node in coeff_dict else {par: 0.0 for par in self.model[node]['parents']}
            self.model[node]['noise_std'] = noise_std_dict[node] if node in noise_std_dict else default_noise_std

    @staticmethod
    def _linear_comb(coeffs_dict, parent_values_dict) -> torch.Tensor:
        result = 0.0
        for par in coeffs_dict.keys():
            result += torch.tensor(parent_values_dict[par], dtype=torch.float32) * \
                      torch.tensor(coeffs_dict[par], dtype=torch.float32)
        return torch.tensor(result)


    def sample(self, n=1, seed=None) -> torch.Tensor:
        """
        Returns a sample from SEM, columns are ordered as var names in DAG
        Args:
            seed: Random seed
            n: Sample size

        Returns: torch.tensor with sampled values, shape (n, len(self.dag.var_names)))

        """
        torch.manual_seed(seed)
        for node in self.topological_order:
            # Sampling noise
            noise = torch.distributions.Normal(0, self.model[node]['noise_std']).rsample(sample_shape=(n,))

            # Linear combination of parent values
            parent_values = {par_node: self.model[par_node]['value'] for par_node in self.model[node]['parents']}
            linear_effect = self._linear_comb(coeffs_dict=self.model[node]['coeff'], parent_values_dict=parent_values)

            self.model[node]['value'] = linear_effect + noise

        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)

    def conditional_pdf(self, node: str, value: torch.Tensor, context: Dict[str, torch.Tensor] = {}) -> torch.Tensor:
        result = torch.zeros((len(value), ))
        for val_ind, val in enumerate(value):
            parent_values = {par_node: context[par_node][val_ind] for par_node in self.model[node]['parents']}
            linear_comb = self._linear_comb(coeffs_dict=self.model[node]['coeff'], parent_values_dict=parent_values)
            cond_dist = torch.distributions.Normal(linear_comb, self.model[node]['noise_std'])
            result[val_ind] = cond_dist.log_prob(val).exp()
        return result

    @property
    def joint_cov(self) -> torch.Tensor:
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
    def joint_mean(self) -> torch.Tensor:
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

    def __init__(self, dag: DirectedAcyclicGraph, bandwidth: float = 1.0,
                 noise_std_dict: Dict[str, float] = {}, default_noise_std: float = 0.0):
        """
        Args:
            dag: DAG, defining SEM
            coeff_dict: Coefficients of linear combination of parent nodes, if not written - considered as zero
            noise_std_dict: Noise std
            default_noise_std: Default noise std
        """
        super(RandomGPGaussianNoiseSEM, self).__init__(dag)

        for node in self.topological_order:
            self.model[node]['noise_std'] = noise_std_dict[node] if node in noise_std_dict else default_noise_std
            self.model[node]['stacked_parent_values'] = None  # Needed for repetitive sampling
            self.model[node]['stacked_random_effects'] = None  # Needed for repetitive sampling

        self.bandwidth = bandwidth

    def _random_function(self, node: str, new_parent_values: collections.OrderedDict, seed=None) -> torch.Tensor:
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


    def sample(self, n, seed=None) -> torch.Tensor:
        torch.manual_seed(seed)
        for node in self.topological_order:
            # Sampling noise
            noise = torch.distributions.Normal(0, self.model[node]['noise_std']).rsample(sample_shape=(n,))

            # Random function of parent values
            new_parent_values = collections.OrderedDict([(par_node, self.model[par_node]['value'])
                                                         for par_node in self.model[node]['parents']])
            random_effect = self._random_function(node, new_parent_values, seed=seed)
            self.model[node]['value'] = random_effect + noise

        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)

    def conditional_pdf(self, node: str, value: torch.Tensor, context: Dict[str, torch.Tensor] = {}) -> torch.Tensor:
        result = torch.zeros((len(value), ))
        for val_ind, val in enumerate(value):
            new_parent_values = collections.OrderedDict([(par_node, context[par_node][val_ind].unsqueeze(-1))
                                                         for par_node in self.model[node]['parents']])
            random_effect = self._random_function(node=node, new_parent_values=new_parent_values)
            cond_dist = torch.distributions.Normal(random_effect, self.model[node]['noise_std'])
            result[val_ind] = cond_dist.log_prob(val).exp()
        return result
