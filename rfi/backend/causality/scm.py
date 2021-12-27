import copy

import networkx as nx
import pandas as pd
import torch
from torch import Tensor
from torch.distributions import Normal, Distribution
import numpy as np

from rfi.backend.causality.dags import DirectedAcyclicGraph
from rfi.backend.gaussian.gaussian_estimator import GaussianConditionalEstimator

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

import logging

logger = logging.getLogger(__name__)

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
        self.topological_order = []
        self.INVERTIBLE = False

        # the model dictionary is an internal representation of the SCM. The keys depend on the specific implementation.
        self.model = {}

        for node in nx.algorithms.dag.topological_sort(dag.DAG):
            parents = dag.DAG.predecessors(node)
            children = dag.DAG.successors(node)
            node, parents, children = dag.var_names[node], [dag.var_names[node] for node in parents], \
                                      [dag.var_names[node] for node in children]

            self.model[node] = {
                'parents': tuple(parents),  # parent variable names
                'children': tuple(children),  # children variable names
                'values': None,  # values for the variable
                'noise_values': None,  # values for the noise
                'noise_distribution': None  # distribution for the noise, may be torch.tensor if point-mass probability
                # 'noise_abducted': None  # deprecated functionality
            }
            self.topological_order.append(node)

    def clear_values(self):
        # remove all sampled values
        for node in self.dag.var_names:
            self.model[node]['noise_values'] = None
            self.model[node]['values'] = None

    def remove_parents(self, node):
        """removes endogenous parents for node"""
        # remove child from parents
        for parent in self.model[node]['parents']:
            children = set(self.model[parent]['children'])
            self.model[parent]['children'] = children.difference_update({node})
        # remove parents from node
        self.model[node]['parents'] = ([])

    def copy(self):
        """
        Creates a deepcopy of the SCM.
        """
        return copy.deepcopy(self)

    def get_sample_size(self):
        """
        Returns the current sample size as determined by the noise_values.
        """
        context_0 = self.model[self.topological_order[0]]['noise_values']
        if context_0 is None:
            return 0
        else:
            return context_0.shape[0]

    def get_values(self):
        """
        Returns the current state values in a pandas.DataFrame (endogenous variables only).
        """
        arr = torch.stack([self.model[node]['values'] for node in self.dag.var_names], dim=1).numpy()
        df = pd.DataFrame(arr, columns=self.dag.var_names)
        return df

    def get_noise_values(self):
        """
        Returns the current state values in a pandas.DataFrame (exogenous variables only).
        The noise variables are named u_[var_name].
        """
        arr = torch.stack([self.model[node]['noise_values'] for node in self.dag.var_names], dim=1).numpy()
        df = pd.DataFrame(arr, columns=['u_' + var_name for var_name in self.dag.var_names])
        return df

    def set_noise_values(self, dict):
        """
        Set the noise values from a dictionary like objects (such as pandas.DataFrame).
        Naming convention: noise variables are named u_[var_name]
        """
        for unode in dict.keys():
            node = unode[2:]  # remove u_ from the name
            self.model[node]['noise_distribution'] = torch.tensor(dict[unode])
        return self.get_noise_values()

    def get_markov_blanket(self, node: str) -> set:
        """
        Get the markov blanket variables for a node as set of variable names.
        """
        return self.dag.get_markov_blanket(node)

    def sample_context(self, size: int, seed=None):
        """
        Use the noise in self.model to generate noise.values in the model.
        Either a torch.Distribution object, a torch.tensor for point-mass distributions or a callable function.
        """
        for node in self.topological_order:
            if isinstance(self.model[node]['noise_distribution'], Distribution):
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'].sample((size,)).flatten()
            elif isinstance(self.model[node]['noise_distribution'], Tensor):
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'].repeat(size)
            elif callable(self.model[node]['noise_distribution']):  # TODO: document this case better.
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'](self)
            else:
                raise NotImplementedError('The noise is neither a torch.distributions.Distribution nor torch.Tensor')
        return self.get_noise_values()

    def do(self, intervention_dict):
        """Intervention

        :param intervention_dict: dictionary of interventions of the form 'variable-name' : value
        :return: copy of the structural causal model with the performend interventions
        """
        scm_itv = self.copy()
        logging.info('Intervening on nodes: {}'.format(intervention_dict.keys()))
        # update structural equations
        for node in intervention_dict.keys():
            scm_itv.remove_parents(node)
            scm_itv.model[node]['noise_distribution'] = torch.tensor(intervention_dict[node])
        scm_itv.clear_values()
        return scm_itv

    def abduct_node(self, node, obs, scm_partially_abducted=None, **kwargs):
        """Abduction

        Args:
            node: name of the node
            obs: observation tensor?
            scm_partially_abducted: all parents of the node must have been abducted already.

        Returns:
            object to be stored as "noise" for the node in the model dictionary.
        """

        logger.info('Abducting noise for: {}'.format(node))
        # check whether node was observed, then only parents necessary for deterministic reconstruction
        if node in obs.index and self.INVERTIBLE:
            # if parents observed then reconstruct deterministically
            if set(self.model[node]['parents']).issubset(set(obs.index)):
                logger.info('\t...by inverting the structural equation given the parents and node.')
                return self._abduct_node_par(node, obs, **kwargs)
            # if not all parents observed then the noise is dependent on the unobserved parent
            else:
                logger.info('\t...as function of the parents noise.')
                return self._abduct_node_par_unobs(node, obs, scm_partially_abducted, **kwargs)
        elif node not in obs.index:
            logger.info('\t...using analytical formula and MC integration.')
            return self._abduct_node_obs(node, obs, **kwargs)
        else:
            raise NotImplementedError('No solution for variable observed but not invertible developed yet.')

    def abduct(self, obs, **kwargs):
        """ Abduct all variables from observation
        Assumes a topological ordering in the DAG.
        returns a separate SCM where the abduction was performed.
        """
        scm_abd = self.copy()
        for node in self.topological_order:
            scm_abd.model[node]['noise_distribution'] = self.abduct_node(node, obs, scm_partially_abducted=scm_abd, **kwargs)
            scm_abd.model[node]['noise_values'] = None
            scm_abd.model[node]['values'] = None
        return scm_abd

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
        assert set(do.keys()).issubset(set(self.dag.var_names))
        assert self.get_sample_size() > 0
        for node in self.topological_order:
            if node in do.keys():
                self.model[node]['values'] = torch.tensor(do[node]).repeat(self.get_sample_size())
            else:
                self.model[node]['values'] = self.compute_node(node).reshape(-1)
            assert (~self.model[node]['values'].isnan()).all()
        return self.get_values()


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

        self.INVERTIBLE = True

        np.random.seed(seed) if seed is not None else None

        for node in self.topological_order:
            self.model[node]['noise_distribution'] = Normal(0, noise_std_dict[node]) if node in noise_std_dict \
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

    def _linear_comb_parents(self, node, obs=None):
        linear_comb = 0.0
        if len(self.model[node]['parents']) > 0:
            for par in self.model[node]['parents']:
                if obs is None:
                    linear_comb += self.model[par]['values'] * torch.tensor(self.model[node]['coeff'][par])
                else:
                    linear_comb += torch.tensor(obs[par]) * torch.tensor(self.model[node]['coeff'][par])
        return linear_comb

    def compute_node(self, node):
        """
        sampling using structural equations
        """
        linear_comb = self._linear_comb_parents(node)
        linear_comb += self.model[node]['noise_values']
        self.model[node]['values'] = linear_comb
        return linear_comb

    def _abduct_node_par(self, node, obs, **kwargs):
        linear_comb = self._linear_comb_parents(node, obs=obs)
        noise = torch.tensor(obs[node]) - linear_comb
        return noise

    def _abduct_node_par_unobs(self, node, obs, scm_abd, **kwargs):
        """
        Abduction when one parent is not fully observed, whose parents were fully observed.
        Meaning we transform the distribution using the relationship between parents and noise
        given the invertability of the function

        p(eps=x) = p(y=g(x)) where g(x) maps x to eps given a certain observed state for the node.
        For linear model that is x_unobs = (sum beta_i parent_i - x_j + eps_j) / beta_unobs
        and eps_j = (x_j - (sum beta_i parent_i + beta_unobs x_unobs))
        """
        assert not (scm_abd is None)
        noisy_pars = [par for par in self.model[node]['parents'] if par not in obs.index]
        if len(noisy_pars) != 1:
            raise NotImplementedError('not implemented for more or less than one parent')
        else:
            raise NotImplementedError('implementation must be validated.')
            lc = 0
            for par in self.model[node]['parents']:
                if par not in noisy_pars:
                    lc += self.model[par]['coeff'] * obs[par]
            noisy_par = noisy_pars[0]
            # Note: we assume that for the scm the parents were already abducted from obs
            lc_noisy_par =  self._linear_comb_parents(noisy_par, obs=obs)
            coeff_noisy_par = self.model[node]['coeff'][noisy_par]

            def sample(scm):
                eps_j = obs[node] - (lc + coeff_noisy_par * (lc_noisy_par + scm.model[noisy_par]['noise_values']))
                return eps_j

            return sample

    def _abduct_node_obs(self, node, obs, samplesize=10**7):
        scm_sampler = self.copy()
        scm_sampler.sample_context(samplesize)
        scm_sampler.compute()
        X = scm_sampler.get_values()
        U = scm_sampler.get_noise_values()

        #mb = scm_sampler.get_markov_blanket(node)
        noise_var = 'u_' + node
        obs_vars = sorted(list(obs.index))

        gaussian_estimator = GaussianConditionalEstimator()

        train_inputs = U[noise_var].to_numpy()
        train_context = X[obs_vars].to_numpy()

        gaussian_estimator.fit(train_inputs=train_inputs,
                               train_context=train_context)

        d = gaussian_estimator.conditional_distribution_univariate(obs[obs_vars].to_numpy())
        return d

class BinomialBinarySCM(StructuralCausalModel):

    def __init__(self, dag, p_dict={}):
        super(BinomialBinarySCM, self).__init__(dag)

        self.INVERTIBLE = True

        for node in self.topological_order:
            if node not in p_dict:
                p_dict[node] = torch.tensor(np.random.uniform(0, 1))
            else:
                p_dict[node] = torch.tensor(p_dict[node])
            self.model[node]['noise_distribution'] = dist.Binomial(probs=p_dict[node])

        self.p_dict = p_dict

    def _get_pyro_model(self, target_node):
        """
        Returns pyro model where the target node is modeled as deterministic function of
        a probabilistic noise term, whereas all other nodes are directly modeled as
        probabilistic variables (such that other variables can be observed and the
        noise term can be inferred).
        """
        def pyro_model():
            var_dict = {}
            for node in self.topological_order:
                input = torch.tensor(0.0)
                for par in self.model[node]['parents']:
                    input = input + var_dict[par]
                input = torch.remainder(input, 2)
                if node != target_node:
                    prob = (1.0 - input) * self.p_dict[node] + input * (1 - self.p_dict[node])
                    var = pyro.sample(node, dist.Binomial(probs=prob))
                    var_dict[node] = var.flatten()
                else:
                    noise = pyro.sample('u_'+node, dist.Binomial(probs=self.p_dict[node]))
                    var = torch.remainder(input + noise, 2)
                    var_dict[node] = var.flatten()
            return var_dict

        return pyro_model

    def _linear_comb_parents(self, node, obs=None):
        linear_comb = 0.0
        if len(self.model[node]['parents']) > 0:
            for par in self.model[node]['parents']:
                if obs is None:
                    linear_comb += self.model[par]['values']
                else:
                    linear_comb += torch.tensor(obs[par])
        return linear_comb


    def compute_node(self, node):
        """
        sampling using structural equations
        """
        linear_comb = self._linear_comb_parents(node)
        linear_comb += self.model[node]['noise_values']
        self.model[node]['values'] = linear_comb
        return torch.remainder(linear_comb, 2)

    def _abduct_node_par(self, node, obs, **kwargs):
        linear_comb = self._linear_comb_parents(node, obs=obs)
        noise = torch.tensor(obs[node]) - linear_comb
        return torch.remainder(noise, 2)

    def _abduct_node_par_unobs(self, node, obs, scm_abd, **kwargs):
        """
        Abduction when one parent is not fully observed, whose parents were fully observed.
        Meaning we transform the distribution using the relationship between parents and noise
        given the invertability of the function

        p(eps=x) = p(y=g(x)) where g(x) maps x to eps given a certain observed state for the node.
        For the binary model
        eps_j = (x_j - (sum parent_i + x_unobs)) % 2
         = (x_j - (sum_parent_i + sum_parent_unobs + epsilon_unobs)) % 2
        -> noise flipped if (x_j - (sum parent_i + sum_parent_unobs)) is 1
        """
        assert not (scm_abd is None)
        noisy_pars = [par for par in self.model[node]['parents'] if par not in obs.index]
        if len(noisy_pars) != 1:
            raise NotImplementedError('not implemented for more or less than one parent')
        else:
            noisy_par = noisy_pars[0]
            # Note: we assume that for the scm the parents were already abducted from obs
            # compute whether the eps_noisy_par must be flipped or is identical to eps_node
            linear_comb = 0
            for par in self.model[node]['parents']:
                if par not in noisy_pars:
                    linear_comb += obs[par]
            linear_comb_noisy_par = self._linear_comb_parents(noisy_par, obs=obs)
            flip = torch.remainder(obs[node] - linear_comb - linear_comb_noisy_par, 2)
            # transform noise to the variable distribution (assuming the respective parents were observed)
            def sample(scm):
                if scm.model[noisy_par]['noise_values'] is None:
                    raise RuntimeError('Noise values for {} must be sampled first'.format(noisy_par))
                noisy_par_values = scm.model[noisy_par]['noise_values']
                values = flip * (1-noisy_par_values) + (1-flip) * noisy_par_values
                return values
            return sample

    def _cond_prob_pars(self, node, obs):
        obs_dict = obs.to_dict()
        assert set(self.model[node]['parents']).issubset(obs_dict.keys())
        assert node in obs_dict.keys()
        input = 0
        for par in self.model[node]['parents']:
            input += obs[par]
        u_node = torch.tensor((obs[node] - input) % 2, dtype=torch.float)
        p = self.model[node]['noise_distribution'].probs
        p_new = u_node * p + (1 - u_node) * (1 - p)
        return dist.Binomial(probs=torch.tensor(p_new))

    def _abduct_node_obs(self, node, obs, n_samples=10**3, **kwargs):
        """
        Using formula to analytically compute the distribution
        """
        def helper(node, obs, multiply_node=True):
            p = 1
            if multiply_node:
                p *= self._cond_prob_pars(node, obs).probs
            for par in self.model[node]['children']:
                p *= self._cond_prob_pars(par, obs).probs
            return p

        obs_nom = obs.copy()
        obs_nom[node] = 1
        nominator = helper(node, obs_nom)

        # get conditional distribution of node=1 | parents
        #cond_dist = self._cond_prob_pars(node, obs_nom)
        #sample = cond_dist.sample((n_samples,)).flatten()
        denominator = 0
        for ii in range(2):
            obs_ii = obs.copy()
            obs_ii[node] = ii
            denominator += helper(node, obs_ii, multiply_node=True)

        p = nominator/denominator

        # determine whether the p needs to be flipped or not to be appropriate for eps and not just y
        linear_comb = torch.remainder(self._linear_comb_parents(node, obs), 2)
        p = (1 - linear_comb) * p + linear_comb * (1 - p)

        # handle cases where slightly larger or smaller than bounds
        if p < 0.0:
            p = 0.0
            logger.debug("probability {} was abducted for node {} and obs {}".format(p, node, obs))
        elif p > 1.0:
            p = 1.0
            logger.debug("probability {} was abducted for node {} and obs {}".format(p, node, obs))

        return dist.Binomial(probs=p)

    # def _abduct_node_obs(self, node, obs, num_steps=5*10**4, **kwargs):
    #
    #     obs_dict = obs.to_dict()
    #     for key in obs_dict.keys():
    #         obs_dict[key] = torch.tensor(obs_dict[key])
    #
    #     pyro_model = self._get_pyro_model(node)
    #     model_cond = pyro.condition(pyro_model, data=obs_dict)
    #
    #     par = 'p_' + node
    #     noise_name = 'u_' + node
    #
    #     def guide():
    #         p = pyro.param(par, torch.tensor(0.5))
    #         nd = pyro.sample(noise_name, dist.Binomial(probs=p))
    #         return nd
    #
    #     pyro.clear_param_store()
    #     svi = pyro.infer.SVI(model=model_cond,
    #                          guide=guide,
    #                          optim=pyro.optim.Adam({"lr": 0.0001}),
    #                          loss=pyro.infer.Trace_ELBO())
    #
    #     losses, p = [], []
    #     N = num_steps
    #
    #     for jj in range(N):
    #         losses.append(svi.step())
    #         p.append(pyro.param(par).item())
    #
    #     d = dist.Binomial(probs=p[-1])
    #     return d
