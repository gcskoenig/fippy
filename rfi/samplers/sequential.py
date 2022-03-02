"""Simple Conditional Sampling from observed data
Only recommended for categorical data with very few states and
all states being observed
"""
import numpy as np
import pandas as pd
from rfi.samplers.sampler import Sampler
from rfi.samplers.simple import SimpleSampler
from rfi.samplers.gaussian import GaussianSampler
import networkx as nx

idx = pd.IndexSlice

class SequentialSampler(Sampler):
    """
    Simple sampling from observed data
    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, adj_mat, categorical_fs, **kwargs):
        """Initialize Sampler with X_train (and mask)."""
        super().__init__(X_train, **kwargs)
        self.adj_mat = adj_mat
        assert type(adj_mat) == pd.core.frame.DataFrame
        self.g = nx.from_pandas_adjacency(adj_mat, create_using=nx.DiGraph)
        self.categorical_fs = categorical_fs

    def train(self, J, G, verbose=True):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            J: features of interest
            G: arbitrary set of variables (conditioning set)
            verbose: printing
        """

        cat_sampler = SimpleSampler(self.X_train)
        cont_sampler = GaussianSampler(self.X_train)

        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        JuG = list(set(J).union(G))

        if not self._train_J_degenerate(J, G, verbose=verbose):
            # order J topologically
            ordering = nx.topological_sort(self.g)
            J_ord = [j for j in ordering if j in J]

            # iterate over the js and train the respective samplers
            for ii in range(len(J_ord)):
                jj = J_ord[ii]
                G_cond_jj = J_ord[ii+1:]
                G_jj = list(set(G_cond_jj).union(G))

                # TODO: check whether this solution is too hacky? can we use the samplefunc in any other class?
                if jj in self.categorical_fs:
                    cat_sampler.train([jj], G_jj, verbose=verbose)
                    self._store_samplefunc([jj], G_jj, cat_sampler._get_samplefunc([jj], G_jj))
                else:
                    # TODO fit other sampler
                    cont_sampler.train([jj], G_jj, verbose=verbose)
                    self._store_samplefunc([jj], G_jj, cont_sampler._get_samplefunc([jj], G_jj))

            def samplefunc(eval_context, num_samples=1, **kwargs):
                X_eval_sub = pd.DataFrame(eval_context, columns=Sampler._order_fset(G))
                X_eval_sub.index = X_eval_sub.index.rename('i')

                ixss = pd.MultiIndex.from_product([X_eval_sub.index, list(range(num_samples))], names=['i', 'sample'])
                X_eval_sub_ixd = pd.DataFrame([], columns=X_eval_sub.columns, index=ixss)

                for ii in range(num_samples):
                    X_eval_sub_ii = X_eval_sub.reset_index()
                    X_eval_sub_ii['sample'] = ii
                    X_eval_sub_ii = X_eval_sub_ii.set_index(['i', 'sample'])
                    X_eval_sub_ixd.loc[idx[ii, :], idx[:]] = X_eval_sub_ii

                # TODO go the other way around
                for ii in range(len(J_ord) - 1, -1, -1):
                    # only for the first variable that is sampled apply num_samples
                    # for all others only one sample per row
                    num_samples_it = 1
                    if ii == len(J_ord) - 1:
                        num_samples_it = num_samples

                    jj = J_ord[ii]
                    G_cond_jj = J_ord[ii + 1:]
                    G_jj = list(set(G_cond_jj).union(G))

                    # eval_context should now be a pandas dataframe?

                    df_row = self.sample(X_eval_sub, [jj], G_jj, num_samples=num_samples_it)
                    X_eval_sub_ixd = X_eval_sub.merge(df_row, left_index=True, right_index=True)

                    # append to existing columns
                    # TODO


                # sort the columns accordingly

                # convert the whole thing to a numpy array

                # return

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)
            return samplefunc
