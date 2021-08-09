from rfi.decorrelators.decorrelator import Decorrelator
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimpleDecorrelator(Decorrelator):
    """
    Naive implementation that just independently resamples
    the variables to be dropped.
    """
    def __init(self, X_train, X_val=None, **kwargs):
        super().__init__(X_train, X_val=True)

    def decorrelate(self, X_test, K, J, C, num_samples=1):
        """Simple decorrelating from observed data for
        categorical data with few states only
         Note: may create off-manifold data"""

        if num_samples != 1:
            raise NotImplementedError('only num_samples 1 implemented.')

        X_test = X_test.reset_index(drop=True)
        obs = np.arange(X_test.shape[0])

        if len(K) == 0:
            # make this a valid option for num_samples > 1
            return X_test
        else:
            x = self.X_train
            # initiate df
            df = X_test.copy()
            # For every j in obs sample vector of features in K
            for j in obs:
                x_ = x.copy()
                # drop every row that is not compatible with the condition (C=X_test[C][j])
                for c in C:
                    feature_value = X_test.loc[j, c]
                    x_ = x_[x_[c] == feature_value]
                # reduce dataframe to features of interest
                x_ = x_[K]
                # sample features K for observation j (simultaneously but off-manifold data possible - for K, J)
                K_j = x_.sample()
                K_j = K_j.reset_index()
                for k in K:
                    df.loc[j, k] = K_j.loc[0, k]
            return df
