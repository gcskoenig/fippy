from itertools import combinations
import pandas as pd
import networkx as nx
import random
import scipy.special as sp
import numpy as np
import os
from scipy.stats import bernoulli
import warnings


def powerset(items):
    """computes power set of iterable object items

    Args:
        items: iterables object, e.g. list, array

    Returns:
         power set of items
    """

    combo = []
    for r in range(len(items) + 1):
        # use a list to coerce an actual list from the combinations generator
        combo.append(list(combinations(items, r)))
    return combo


# function for all d separations wrt target:
def d_separation(g, y, g2=None, mc=None, random_state=None):
    """Test d-separation of each single node and y given every possible conditioning set in graph g

    Args:
        g : nx.DiGraph
        y : target node with respect to which d-separation is tested
        g2: potential second graph to be tested, that must contain the same nodes as g (typically an estimate of g)
        mc : if int given, mc sampling a subset of d-separations to be tested, recommended for large graphs
        random_state : seed for random and np.random when mc is not None

    Returns:
         pandas dataframe of Boolean values for d-separation for every node except y (if mc is None)
         boolean array of d-separation, it cannot be traced back which d-separations were tested explicitly
         (if mc is not None)
    """
    # list of nodes (strings)
    predictors = list(g.nodes)
    # sort list to get consistent results across different graphs learned on same features (when using mc)
    predictors.sort()
    # remove the target from list of predictors
    predictors.remove(y)
    n = len(predictors)

    # TODO remove everything that is 'too large'
    # number of possible d-separations between one feature and target, i.e. number of potential conditioning sets
    if mc is None:
        # number of potential d-separations per node
        no_d_seps = (2 ** (n-1))
        # warn user if number of d-separation tests is large
        if no_d_seps > 1000000:
            warnings.warn("Warning: No. of d-separation tests per node > 1M, can lead to large runtime")
        # initiate dataframe for all d-separations
        d_separations = pd.DataFrame(index=predictors, columns=range(no_d_seps))

    if mc is not None:
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            rng = np.random.default_rng(seed=random_state)
        else:
            rng = np.random.default_rng()
        # initiate vector to store True/False for d-separations (cannot track nodes and conditioning sets)
        d_seps_bool = []
        if g2 is not None:
            d_seps_bool_2 = []
        # get a vector of probabilities to draw the size of the conditioning set; note that for n-1 potential
        # deconfounders there are n different sizes of conditioning sets because of the empty set
        probs = []
        for jj in range(n):
            probs.append((sp.comb(n-1, jj)) / (2**(n-1)))
        k = 0
        while k < mc:
            # draw index for feature of interest
            ind = random.randint(0, n-1)
            # retrieve feature of interest
            node = predictors[ind]
            # list of all features but feature of interest
            deconfounders = list(g.nodes)
            deconfounders.remove(y)
            deconfounders.remove(node)
            # sample a conditioning set from deconfounders
            # draw a cardinality
            card = np.random.choice(np.arange(n), p=probs)
            if card == 0:
                # test d-separation with empty set as conditioning set
                cond_set = set()
                d_seps_bool.append(nx.d_separated(g, {node}, {y}, cond_set))
                if g2 is not None:
                    d_seps_bool_2.append(nx.d_separated(g2, {node}, {y}, cond_set))

            else:
                # draw as many as 'card' numbers from range(n-1) as indices for conditioning set
                indices = rng.choice(n-1, size=card, replace=False)
                cond_set = set()
                for ii in range(len(indices)):
                    # index for first
                    index = indices[ii]
                    cond_set.add(deconfounders[index])
                d_seps_bool.append(nx.d_separated(g, {node}, {y}, cond_set))
                if g2 is not None:
                    d_seps_bool_2.append(nx.d_separated(g2, {node}, {y}, cond_set))
            k += 1
        if g2 is not None:
            return d_seps_bool, d_seps_bool_2
        else:
            return d_seps_bool  # vector of booleans
    else:
        if g2 is not None:
            print("Note: g2 only for Monte Carlo inference, will not be regarded here")

        for i in range(n):
            # test d-separation w.r.t. target using all conditional sets possible
            # for current predictor at i-th position in predictors
            node = predictors[i]
            deconfounders = list(g.nodes)
            deconfounders.remove(y)
            deconfounders.remove(node)
            power_set = powerset(deconfounders)
            j = 0
            while j < no_d_seps:
                for k in range(len(power_set)):
                    # k == 0 refers to the empty set in power_set
                    if k == 0:
                        cond_set = set()
                        d_separations.iloc[i, j] = nx.d_separated(g, {node}, {y}, cond_set)
                        j += 1
                    else:
                        for jj in range(len(power_set[k])):
                            cond_set = {power_set[k][jj][0]}
                            for m in range(len(power_set[k][jj]) - 1):
                                cond_set.add(power_set[k][jj][m + 1])
                            d_separations.iloc[i, j] = nx.d_separated(
                                g, {node}, {y}, cond_set
                            )
                            j += 1
        return d_separations    # df with d-separation Booleans for every predictor


# compute the number of d-separation statements from n

def dsep_mb(n, mb):
    """computes a lower bound for the number of d-separation statements between a target node
    and every other node in the graph given the size of the target node's Markov blanket

    Args:
        n: number of nodes in graph
        mb: size of Markov blanket of target

    Returns:
        Lower bound for number of d-separations
    """

    # number of nodes other than y and Markov blanket: n-1-mb
    # number of d-separations for such nodes 2**(n-2-mb)
    return (n - 1 - mb) * 2 ** (n - 2 - mb)


def dsep_degree(n, max_degree, sink=False):
    """computes a lower bound for the number of d-separation statements between a target node
        and every other node in the graph given the max degree of a node

        Args:
            n : number of nodes in graph
            max_degree : maximal degree of a node in the graph (max. number of edges associated to a node
            sink : Bool, whether target is sink node or not
        Returns:
            Lower bound for number of d-separations
        """
    if sink is False:
        # maximal size of Markov blanket
        max_mb = max_degree + max_degree ** 2
        return dsep_mb(n, max_mb)
    else:
        max_mb = max_degree
        return dsep_mb(n, max_mb)


def potential_dseps(n):
    """For a graph of size n, return the maximal number of d-separations between each node and a potentially
    dedicated target node

    Args:
        n: number of nodes in graph

    Return:
        Number of potential d-separation statements (float)
    """
    return (n - 1) * (2 ** (n - 2))


def create_folder(directory):
    """Creates directory as specified in function argument if not already existing

    Args:
        directory: string specifying the directory
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory: " + directory)


def create_amat(n, p):
    """Create a random adjacency matrix of size n x n

    Args:
        n: number of nodes
        p: probability of existence of an edge for each pair of nodes

    Returns:
        Adjacency matrix as pd.DataFrame

    """
    # create col_names
    variables = []
    for i in range(n):
        variables.append(str(i+1))

    # create df for amat
    amat = pd.DataFrame(columns=variables, index=variables)

    # TODO avoid double loop (use zip)
    for j in range(n):
        for k in range(n):
            amat.iloc[j, k] = bernoulli(p)


def exact(g_true, g_est, y):
    # the two dataframes of d-separations
    true_dseps = d_separation(g_true, y)
    est_dseps = d_separation(g_est, y)
    # now compare every entry
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(true_dseps.shape[0]):
        for j in range(true_dseps.shape[1]):
            if true_dseps.iloc[i, j] == est_dseps.iloc[i, j]:
                if true_dseps.iloc[i, j] is True:
                    tp += 1
                else:
                    tn += 1
            else:
                if true_dseps.iloc[i, j] is True:
                    fn += 1
                else:
                    fp += 1
    # total number of d-separations in true graph
    d_separated_total = tp + fn
    d_connected_total = tn + fp
    return tp, tn, fp, fn, d_separated_total, d_connected_total


def approx(g_true, g_est, y, mc=None, rand_state=None):
    true_dseps, est_dseps = d_separation(g_true, y, g2=g_est, mc=mc, random_state=rand_state)
    # now compare every entry
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(true_dseps)):
        if true_dseps[i] == est_dseps[i]:
            if true_dseps[i] is True:
                tp += 1
            else:
                tn += 1
        else:
            if true_dseps[i] is True:
                fn += 1
            else:
                fp += 1
    # total number of d-separation among tested nodes (make a node if d-separations were approximated via mc)
    d_separated_total = tp + fn
    d_connected_total = tn + fp
    return tp, tn, fp, fn, d_separated_total, d_connected_total


def convert_amat(df, col_names=False):
    """Convert adjacency matrix of Bools to 0-1 for use in networkx package
    Args:
        df: adjacency matrix
        col_names: toggle overriding of column names with strings starting from "1"
    """

    if col_names:
        col_names_str = []
        for k in range(len(df.columns)):
            col_names_str.append(str(k+1))
        df.columns = col_names_str

    mapping_rf = {False: 0, True: 1}
    col_names = df.columns
    for j in col_names:
        df[j] = df.replace({j: mapping_rf})[j]  # TODO (cl) [j] required in the end?

    # modify adjacency matrix for use in networkx package
    df = df.set_axis(df.columns, axis=0)
    # store modified adjacency matrix
    return df
