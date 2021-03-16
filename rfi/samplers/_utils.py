import numpy as np


def sample_id(J):
    ''' Simple sampler that returns sample function that simply
    copies the original data

    Args:
        X_test: data
        ixs: ixs for columns to be "sampled" within
            context numpy array
    '''

    def sample(X_context, num_samples=1):
        """ Sample function that returs a copy of
        X_test[:, J] of shape (#obs, #num_samples, #J)
        """
        res = np.zeros((X_context.shape[0], num_samples, len(J)))
        for kk in range(num_samples):
            res[:, kk, :] = np.array(X_context[J].to_numpy())
        return res

    return sample


def sample_perm(J):
    ''' Simple sampler that permutes the value

    Args:
        X_test: data
        ixs: ixs for columns to be "sampled"
    '''

    def sample(X_context, num_samples=1):
        """
        Sample function that returs a permutation of
        X_test[:, J] of shape (#obs, #num_samples, #J)
        """
        res = np.zeros((len(X_context), num_samples, len(J)))
        for kk in range(num_samples):
            xs = np.array(X_context[J].to_numpy())  # copy into new array
            np.random.shuffle(xs)  # shuffle in-place
            res[:, kk, :] = xs
        return res

    return sample
