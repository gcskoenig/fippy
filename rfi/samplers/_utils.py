import numpy as np
from rfi.utils import deprecated


def sample_id(J_ixs):
    ''' Simple sampler that returns sample function that simply
    copies the original data

    Args:
        J_ixs: ixs for columns to be "sampled" within
            context numpy array
    '''

    def sample(X_context, num_samples=1):
        """ Sample function that returs a copy of
        X_test[:, J] of shape (#obs, #num_samples, #J_ixs)
        """
        res = np.zeros((X_context.shape[0], num_samples, len(J_ixs)))
        for kk in range(num_samples):
            res[:, kk, :] = X_context[:, J_ixs]
        return res

    return sample

@deprecated
def sample_perm(J_ixs):
    ''' Simple sampler that permutes the value

    Args:
        X_test: data
        ixs: ixs for columns to be "sampled"
    '''

    def sample(X_test, num_samples=1):
        """
        Sample function that returs a permutation of
        X_test[:, ix] of shape (#obs, #num_samples, #ix)
        """
        res = np.zeros((X_test.shape[0], num_samples, len(J_ixs)))
        for kk in range(num_samples):
            xs = np.array(X_test[:, J_ixs])  # copy into new array
            np.random.shuffle(xs)  # shuffle in-place
            res[:, kk, :] = xs
        return res
    return sample
