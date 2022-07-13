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

    def __init__(self, X_train, categorical_fs, adj_mat=None, cat_sampler=None, cont_sampler=None, **kwargs):
        """Initialize Sampler with X_train (and mask)."""
        super().__init__(X_train, **kwargs)
        self.adj_mat = adj_mat
        if not self.adj_mat is None:
            assert type(self.adj_mat) == pd.core.frame.DataFrame
            self.g = nx.from_pandas_adjacency(adj_mat, create_using=nx.DiGraph)
        else:
            self.g = None
        self.categorical_fs = categorical_fs
        self.cat_sampler = cat_sampler
        self.cont_sampler = cont_sampler
        if cat_sampler is None:
            self.cat_sampler = SimpleSampler(X_train)
        if cont_sampler is None:
            self.cont_sampler = GaussianSampler(X_train)


    def train(self, J, G, verbose=True):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            J: features of interest
            G: arbitrary set of variables (conditioning set)
            verbose: printing
        """
        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        JuG = list(set(J).union(G))

        if not self._train_J_degenerate(J, G, verbose=verbose):
            if self.g is None:
                ordering = self.X_train.columns
            else:
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
                    self.cat_sampler.train([jj], G_jj, verbose=verbose)
                    self._store_samplefunc([jj], G_jj, self.cat_sampler._get_samplefunc([jj], G_jj))
                else:
                    # TODO fit other sampler
                    self.cont_sampler.train([jj], G_jj, verbose=verbose)
                    self._store_samplefunc([jj], G_jj, self.cont_sampler._get_samplefunc([jj], G_jj))

            def samplefunc(eval_context, num_samples=1, **kwargs):
                X_eval_sub = pd.DataFrame(eval_context, columns=Sampler._order_fset(G))
                X_eval_sub.index = X_eval_sub.index.rename('i')

                X_res_dfs = []

                for ii in range(num_samples):
                    X_eval_sub_ii = X_eval_sub.reset_index()
                    X_eval_sub_ii['sample'] = ii
                    X_eval_sub_ii = X_eval_sub_ii.set_index(['i', 'sample'])
                    X_res_dfs.append(X_eval_sub_ii)

                X_res = pd.concat(X_res_dfs)
                # TODO go the other way around
                for ii in range(len(J_ord) - 1, -1, -1):
                    jj = J_ord[ii]
                    G_cond_jj = J_ord[ii + 1:]
                    G_jj = list(set(G_cond_jj).union(G))

                    # num_samples was already incorporated earlier
                    df_row = self.sample(X_res, [jj], G_jj, num_samples=1)
                    X_res[df_row.columns] = np.array(df_row)


                # sort the columns accordingly and convert to numpy
                X_res = X_res[Sampler._order_fset(J)]
                X_res_np = X_res.to_numpy()
                X_res_np_rs = X_res_np.reshape(-1).reshape(num_samples, eval_context.shape[0], eval_context.shape[1])
                X_res_np_rs = X_res_np_rs.swapaxes(0, 1)
                # return
                return X_res_np_rs

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)
