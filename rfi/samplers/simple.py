"""Simple Conditional Sampling from observed data
Only recommended for categorical data with very few states and
all states being observed
"""
import numpy as np
import pandas as pd
from rfi.samplers.sampler import Sampler
from rfi.utils import create_multiindex


class Simple(Sampler):
    """
    Simple sampling from observed data
    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, **kwargs):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train, **kwargs)

    def sample(self, X_test, J, G, num_samples=1):
        """Simple sampling approach given the values for set G in X_test
        sampling from the distribution implied by X_train"""

        X_test = X_test.reset_index(drop=True)

        snrs = np.arange(num_samples)
        obs = np.arange(X_test.shape[0])
        vss = [snrs, obs]
        ns = ['sample', 'i']
        index = create_multiindex(ns, vss)

        if len(J) == 0:
            df = pd.DataFrame([], index=index)
            return df
        else:
            x = self.X_train
            # initiate df
            df = pd.DataFrame([], index=index, columns=J)
            # For every i in snrs and every j in obs sample vector of features in J
            for i in snrs:
                for j in obs:
                    x_ = x.copy()
                    # drop every row that is not compatible with the condition (G=X_test[G][j])
                    for k in G:
                        feature_value = X_test.loc[j, k]
                        x_ = x_[x_[k] == feature_value]
                    # reduce dataframe to features of interest
                    x_ = x_[J]
                    # sample features J for observation j (simultaneously to avoid off-manifold data)
                    J_j = x_.sample()
                    ind = (i, j)
                    df.loc[ind, ] = J_j.values
            return df