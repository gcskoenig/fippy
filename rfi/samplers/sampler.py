"""Sampler can be used sample perturbed versions of a variable
conditional on a set G.

More details can be found in the class description
"""


class Sampler():
    """Can be used to resample perturbed versions of a variable conditional on
    any set of variables G.

    # TODO(gcsk): potentially allow storing training on multiple sets G.

    Attributes:
        X_train: reference to training data.
        mask: features of interest.
        __trained: private variable indicating whether the sampler was trained
        __G: private variable indicating regarding which set G
          the sampler was trained
    """

    def __init__(self, X_train, fsoi):
        """Initialize Sampler with X_train and mask."""
        self.X_train = X_train
        self.fsoi = fsoi
        self.__trained = False
        self.__G = None

    def is_trained(self, G):
        """Indicates whether the Sampler has been trained
        on a specific RFI set G.
        
        Args:
            G: RFI set G to be checked

        Returns:
            Whether the sampler was trained with respect to
            a set G.

        """
        # TODO(gcsk): validate whether trained
        # TODO(gcsk): validate whether G and __G coincide
        pass

    def train(self, G):
        """Trains sampler using the training dataset to resample
        relative to any variable set G.

        Args:
            G: arbitrary set of variables.

        Returns:
            Nothing. Now the sample function can be used
            to resample on seen or unseen data.
        """
        self.__trained = True
        self.__G = G
        pass

    def sample(self, X_eval):
        """Sample features of interest using trained resampler.

        Args:
            X_eval: Data for which sampling shall be performed.

        Returns:
            Resampled data for the features of interest.
            np.array with shape (X_eval.shape[0], # features of interest)
        """
        if self.is_trained():
            # TODO(gcsk): assert that it was trained on the correct set
            pass
        else:
            pass
