"""Sampler can be used sample perturbed versions of a variable
conditional on a set G.

More details can be found in the class description
"""

import rfi.utils as utils
import numpy as np
from rfi.samplers._utils import sample_id, sample_perm
import logging


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
        trained = (J_key, G_key) in self._trainedGs
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
        degenerate = True
        
        # are we conditioning on zero elements?
        if G.size == 0:
            logging.debug('Degenerate Training: Empty G')
            self._store_samplefunc(J, G, sample_perm(J))
        # are all elements in G being conditioned upon?
        elif np.sum(1 - np.isin(J, G)) == 0:
            logging.debug('Degenerate Training: J subseteq G')
            self._store_samplefunc(J, G, sample_id(J))
        else:
            logging.debug('Training not degenerate.')
            degenerate = False

        return degenerate

    def _store_samplefunc(self, J, G, samplefunc, verbose=True):
        """Storing a trained sample function

        Args:
            samplefunc: function
            J: features of interest
            G: relative feature set
            verbose: printing or not
        """
        G_key, J_key = Sampler._to_key(G), Sampler._to_key(J)
        self._trainedGs[(J_key, G_key)] = samplefunc
        logging.info('Training ended. Sampler saved.')

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
        logging.info('Training Sampler for: {} | {}'.format(J, G))

    def sample(self, X_test, J, G, num_samples=1):
        """Sample features of interest using trained resampler.

        Args:
            J: Set of features to sample
            G: relative feature set
            X_test: Data for which sampling shall be performed. (format as self.X_train)
            num_samples: number of resamples without retraining shall be computed

        Returns:
            Resampled data for the features of interest.
            np.array with shape (X_test.shape[0], #num_samples, # features of interest)
        """
        # initialize numpy matrix
        #sampled_data = np.zeros((X_test.shape[0], num_samples, J.shape[0]))

        # sample
        G_key, J_key = Sampler._to_key(G), Sampler._to_key(J)

        if not self.is_trained(J, G):
            raise RuntimeError("Sampler not trained on {} | {}".format(J, G))
        else:
            sample_func = self._trainedGs[(J_key, G_key)]
            sampled_data = sample_func(X_test, num_samples=num_samples)
            return sampled_data
