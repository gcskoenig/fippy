"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to model the
conditional distribution.
"""
import rfi.samplers.Sampler as Sampler
import DeepKnockoffs


class GaussianSampler(Sampler):
    """
    Second order Gaussian Sampler.

    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, fsoi):
        """Initialize Sampler with X_train and mask."""

        super().__init__(self, X_train, fsoi)

    def train(self, G):
        """Trains sampler using the training dataset to resample
        relative to any variable set G.

        Args:
            G: arbitrary set of variables.

        Returns:
            Nothing. Now the sample function can be used
            to resample on seen or unseen data.
        """
        super().train(G)  # updates "is_trained" functionality
        

        # TODO(gcsk): train one model per fsoi
        # TODO(gcsk): Print progress
        pass

    def sample(self, X_test, G):
        """Sample features of interest using trained resampler.

        Args:
            X_test: Data for which sampling shall be performed.

        Returns:
            Resampled data for the features of interest.
            np.array with shape (X_test.shape[0], # features of interest)
        """
        super().sample(X_test, G)  # asserts that it was trained

        # TODO(gcsk): resample for every fsoi
        # TODO(gcsk): print progress
        pass



# def knockoff_caller(X_train, X_test):
#     '''
#     knockoff code as given in examples for deepknockoff paper

#     '''
#     SigmaHat = np.cov(X_train, rowvar=False)
#     second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train,0))#, method="sdp")
#     knockoffs = second_order.generate(X_test)
#     return knockoffs


# def create_2nd_order_knockoff(j, G, X_train, X_test):
#     '''
#     j: index of variable of interest
#     G: list of conditioning indexes
#     '''
#     G = np.array(G) # conditioning set
#     S = np.zeros(np.prod(G.shape)+1, dtype=np.int16)
#     S[:-1] = G
#     S[-1] = j # variable to be replaced
#     knockoffs = knockoff_caller(X_train[:, S], X_test[:, S]) # creates knockoffs
#     knockoff_j = knockoffs[:, -1]
#     return knockoff_j # knockoff of j computed from G