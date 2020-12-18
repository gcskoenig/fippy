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

    def __init__(self, X_train):
        """Initialize Sampler with X_train and mask."""
        self.X_train = X_train
        self._trainedGs = {}


    @staticmethod
    def _to_key(S):
        '''
        Converts array to key for trainedGs Dict
        '''
        return utils.to_key(S)

    @staticmethod
    def _to_array(S):
        """Coverts to numpy array
        """
        return np.array(S, dtype=np.int16).reshape(-1)

    def is_trained(self, J, G):
        """Indicates whether the Sampler has been trained
        on a specific RFI set G for features J.

        Args:
            G: RFI set G to be checked

        Returns:
            Whether the sampler was trained with respect to
            a set G.

        """
        G_key, J_key = Sampler._to_key(G), Sampler._to_key(J) # transform into hashable form
        trained = (G_key, J_key) in self._trainedGs()
        return trained


    def _train_J_degenerate(self, J, G, verbose=True):
        """Training function that takes care of degenerate cases
        where either j is in G or G is empty.
        
        Args:
            J: features of interest
            G: relative feature set

        Returns:
            Whether a degenerate case was present.
        """        
        G_key, J_key = utils.to_key(G), utils.to_key(J)

        degenerate = True

        if J in G: # TODO update for subset numpy array
            self._trainedGs[(j_key, G_key)] = sample_id(j)
        elif G.size == 0:
            self._trainedGs[(j_key, G_key)] = sample_perm(j)
        else:
            degenerate = False

        return degenerate

    def _store_samplefunc(self, samplefunc, J, G, verbose=True):
        """Storing a trained sample function

        Args:
            samplefunc: function
            J: features of interest
            G: relative feature set
            verbose: printing or not
        """
        G_key, J_key = Sampler._to_key(G), Sampler.to_key(J)
        self._trainedGs[(J_key, G_key)] = samplefunc
        if verbose:
            print('Training ended. Sampler saved.')


    def train(self, J, G, verbose=True):
        """Trains sampler using the training dataset to resample
        relative to any variable set G.

        Args:
            J: set of features to train on
            G: arbitrary set of variables.

        Returns:
            Nothing. Now the sample function can be used
            to resample on seen or unseen data.
        """
        print('Training Sampler for: {} | {}'.format(J, G))


    def sample(self, X_test, J, G):
        """Sample features of interest using trained resampler.

        Args:
            J: Set of features to sample
            G: relative feature set
            X_test: Data for which sampling shall be performed.

        Returns:
            Resampled data for the features of interest.
            np.array with shape (X_test.shape[0], # features of interest)
        """
                # initialize numpy matrix
        J = np.array(J, dtype=np.int16)
        sampled_data = np.zeros((X_test.shape[0], J.shape[0]))

        # sample
        G_key, J_key = Sampler._to_key(G), Sampler._to_key(J)

        if not super().is_trained(J, G):
            # TODO(gcsk): raise exception
        else:
            sample_func = self._trainedGs[(J_key, G_key)]
            sampled_data = sample_func(X_test)
            return sampled_data
