"""Sampler can be used sample perturbed versions of a variable
conditional on a set G.

More details can be found in the class description
"""

import rfi.utils as utils


class Sampler():
    """Can be used to resample perturbed versions of a variable conditional on
    any set of variables G.

    # TODO(gcsk): potentially allow storing training on multiple sets G.

    Attributes:
        X_train: reference to training data.
        fsoi: features of interest.
        _trainedGs: dictionary with (fsoi, G) as key and callable sampler as value
    """

    def __init__(self, X_train, fsoi):
        """Initialize Sampler with X_train and mask."""
        self.X_train = X_train
        self.fsoi = fsoi
        self._trainedGs = {}

    def is_trained(self, J, G):
        """Indicates whether the Sampler has been trained
        on a specific RFI set G.

        Args:
            G: RFI set G to be checked

        Returns:
            Whether the sampler was trained with respect to
            a set G.

        """
        G_key = utils.to_key(G) # transform into hashable form
        for jj in J:
            jj_key = utils.to_key([jj])
            if (jj_key, G_key) not in self._trainedGs:
                return False 

        return (J_key, G_key) in self._trainedGs # check whether key is in dictionary

    def train(self, G):
        """Trains sampler using the training dataset to resample
        relative to any variable set G.

        Args:
            G: arbitrary set of variables.

        Returns:
            Nothing. Now the sample function can be used
            to resample on seen or unseen data.
        """
        pass

    def sample(self, X_test, G):
        """Sample features of interest using trained resampler.

        Args:
            X_test: Data for which sampling shall be performed.

        Returns:
            Resampled data for the features of interest.
            np.array with shape (X_test.shape[0], # features of interest)
        """
        pass
