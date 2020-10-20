"""Model-X Knockoff sampler based on arXiv:1610.02351.

Second-order Gaussian models are used to model the
conditional distribution.
"""
import rfi.samplers.Sampler as Sampler
import DeepKnockoffs


class GaussianSampler(Sampler):
    """
    Second order Gaussian Sampler

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
        
        # TODO(gcsk): Do the actual training.
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

        # TODO(gcsk): Do the actual sampling
        # TODO(gcsk): print progress
        pass
