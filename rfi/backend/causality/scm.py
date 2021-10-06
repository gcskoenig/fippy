import networkx as nx
import torch
from torch import Tensor
from torch.distributions import Normal, Distribution
import numpy as np

from rfi.backend.causality.dags import DirectedAcyclicGraph

class StructuralCausalModel:
    """
    Semi-abstract class for SCM, defined by a DAG.
    Implementation based on Pyro, implements do-operator and
    tools for the computation of counterfactuals.
    """

    def __init__(self, dag: DirectedAcyclicGraph):
        """
        Args:
            dag: DAG, defining SCM
        """
        # Model init
        self.dag = dag
        self.model = {}
        self.topological_order = []

        for node in nx.algorithms.dag.topological_sort(dag.DAG):
            parents = dag.DAG.predecessors(node)
            children = dag.DAG.successors(node)
            node, parents, children = dag.var_names[node], [dag.var_names[node] for node in parents], \
                                      [dag.var_names[node] for noe in children]

            self.model[node] = {
                'parents': tuple(parents),
                'children': tuple(children),
                'values': None,
                'noise_values': None,
                'noise': None
            }
            self.topological_order.append(node)

    def get_sample_size(self):
        """
        """
        context_0 = self.model[self.topological_order[0]]['context_value']
        if context_0 is None:
            return 0
        else:
            return context_0.shape[0]

    def get_markov_blanket(self, node: str) -> set:
        return self.dag.get_markov_blanket(node)

    def sample_context(self, size: int, seed=None):
        """
        Sampling the context variables
        """
        raise NotImplementedError('not implemented in semi-abstract class')

    def abduct_node(self, node):
        """
        Abduction
        """
        # TODO assert that parents and node are stored in self.model
        raise NotImplementedError('not implemented in semi-abstract class')

    def abduct_node_mb(self):
        """
        Abduction when the node is not observed based on markov blanket variables
        """
        # TODO assert that markov blanket values stored in self.model
        raise NotImplementedError('not implemented in semi-abstract class')

    def compute_node(self, node):
        """
        sampling using structural equations
        """
        raise NotImplementedError('not implemented in semi-abstract class')

    def compute(self, do={}):
        """
        Returns a sample from SEM (observational distribution), columns are ordered as var self.dag.var_names in DAG
        Requires that context variables are set/sampled.
        Args:
            do: dictionary with {'node': value}

        Returns: torch.Tensor with sampled value of shape (size, n_vars))
        """
        assert self.get_sample_size() > 0
        for node in self.topological_order:
            if node in do.keys():
                self.model[node]['values'] = torch.tensor(do[node]).repeat(self.get_sample_size())
            else:
                self.model[node]['values'] = self.compute_node(node).reshape(-1)
            assert (~self.model[node]['values'].isnan()).all()
        return torch.stack([self.model[node]['value'] for node in self.dag.var_names], dim=1)


class LinearGaussianNoiseSCM(StructuralCausalModel):

    def __init__(self, dag: DirectedAcyclicGraph,
                 coeff_dict={},
                 noise_std_dict={},
                 default_coeff=(0.0, 1.0),
                 default_noise_std_bounds=(0.0, 1.0),
                 seed=None):
        """
        Args:
            dag: DAG, defining SEM
            coeff_dict: Coefficients of linear combination of parent nodes, if not written - considered as zero
            noise_std_dict: Noise std dict for each variable
            default_noise_std_bounds: Default noise std, if not specified in noise_std_dict.
                Sampled from U(default_noise_std_bounds[0], default_noise_std_bounds[1])
        """
        super(LinearGaussianNoiseSCM, self).__init__(dag)

        np.random.seed(seed) if seed is not None else None

        for node in self.topological_order:
            self.model[node]['noise'] = Normal(0, noise_std_dict[node]) if node in noise_std_dict \
                else Normal(0, np.random.uniform(*default_noise_std_bounds))

            coeff = None

            if node in coeff_dict:
                coeff = coeff_dict[node]
            else:
                if type(default_coeff) == float:
                    coeff = {par: default_coeff for par in self.model[node]['parents']}
                else:
                    coeff = {par: np.random.uniform(*default_coeff) for par in self.model[node]['parents']}

            self.model[node]['coeff'] = coeff

    def sample_context(self, size: int, seed=None):
        """
        Use the noise in self.model to generate noise.values in the model.
        """
        for node in self.topological_order:
            if isinstance(self.model[node]['noise'], Distribution):
                self.model[node]['noise_values'] = self.model[node]['noise'].sample(size)
            elif isinstance(self.model[node]['noise'], Tensor):
                self.model[node]['noise_values'] = self.model[node]['noise'].repeat(size)
            else:
                raise NotImplementedError('The noise is neither a torch.distributions.Distribution nor torch.Tensor')

    def compute_node(self, node):
        """
        sampling using structural equations
        """
        linear_comb = 0.0
        if len(self.model[node]['parents']) > 0:
            for par in self.model[node]['parents']:
                linear_comb += self.model[par]['values'] * torch.tensor(self.model[node]['coeff'][par])
        linear_comb += self.model[node]['noise_values']
        self.model[node]['values'] = linear_comb
        return linear_comb