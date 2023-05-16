import numpy as np
import pandas as pd
from rfi.utils import deprecated


def sample_id(J_ixs):
    """ Simple sampler that returns sample function that simply
    copies the original data

    Args:
        J_ixs: ixs for columns to be "sampled" within
            context numpy array
    """

    def sample(X_context, num_samples=1):
        """ Sample function that returns a copy of
        X_test[:, J] of shape (#obs, #num_samples, #J_ixs)
        """
        X_J = X_context[:, J_ixs]
        X_J_rep = np.stack([X_J]*num_samples, axis=2)
        X_J_rep = X_J_rep.swapaxes(1, 2)
        # index = create_multiindex(['sample', 'i'],
        #                           [np.arange(num_samples), list(X_J.index)])
        # X_J_rep.index = index
        # res = np.zeros((X_context.shape[0], num_samples, len(J_ixs)))
        # for kk in range(num_samples):
        #     res[:, kk, :] = X_context[:, J_ixs]
        return X_J_rep

    return sample
  
def sample_perm(J_ixs, X_train):
    """ Simple sampler that permutes the value

    Args:
        J_ixs: ixs for columns to be "sampled"
    """
    n_train = X_train.shape[0]
    X_train = np.array(X_train)
    def sample(X_test, num_samples=1):
        """
        Sample function that returns a permutation of
        X_test[:, ix] of shape (#obs, #num_samples, #ix)
        """
        # res = np.zeros((X_test.shape[0], num_samples, len(J_ixs)))
        X_J = X_train[:, J_ixs]
        X_J_perms = []
        for kk in range(num_samples):
            # xs = np.array(X_test[:, J_ixs])  # copy into new array
            # np.random.shuffle(xs)  # shuffle in-place
            # res[:, kk, :] = xs
            X_J_perm = X_J[np.random.permutation(range(X_J.shape[0])), :]
            X_J_perms.append(X_J_perm)
        res = np.stack(X_J_perms, axis=2)
        res = res.swapaxes(1, 2)
        return res
    return sample
