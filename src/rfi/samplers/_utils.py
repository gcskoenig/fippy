import numpy as np
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
        # res = np.zeros((X_context.shape[0], num_samples, len(J_ixs)))
        arrs = []
        for kk in range(num_samples):
            arr = X_context[:, J_ixs].reshape(X_context.shape[0], 1, len(J_ixs))
            arrs.append(arr)
        res = np.concatenate(arrs, axis=1)
        return res

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
        n_test = X_test.shape[0]
        # res = np.zeros((X_test.shape[0], num_samples, len(J_ixs)))
        xss = []
        for kk in range(num_samples):
            ixs = np.random.choice(n_train, n_test, replace=True)
            xs = np.array(X_train[np.ix_(ixs, J_ixs)]).reshape(n_test, 1, len(J_ixs))
            xss.append(xs)
        res = np.concatenate(xss, axis=1)
        return res
    return sample
