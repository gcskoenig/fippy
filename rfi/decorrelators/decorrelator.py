"""Decorrelator can be used sample perturbed versions of a variable
conditional on a set C.

More details can be found in the class description
"""

import rfi.utils as utils
import numpy as np
import logging
from typing import Union, Callable
from rfi.samplers.sampler import Sampler

logger = logging.getLogger(__name__)


class Decorrelator:
    """Can be used to decorrelate samples with a variable of interest
    while preserving desirable properties of the distribution.

    Attributes:
        X_train: reference to the training data to be decorrelated
        _trained_decorrelation_funcs: dictionary with (fsoi, C)
            as key and callable sampler as value
    """

    def __init__(self, X_train, X_val=None):
        """Initialize Decorrelator with X_train"""
        self.X_train = X_train
        self.X_val = X_val
        self._trained_decorrelation_funcs = {}
        self._trained_estimators = {}

    @staticmethod
    def _to_key(S):
        """
        Converts array to key for trainedCs Dict
        """
        return Sampler._to_key(S)

    @staticmethod
    def _to_array(S):
        """Coverts to numpy array
        """
        return Sampler._to_array(S)

    @staticmethod
    def _order_fset(S):
        return Sampler._order_fset(S)

    @staticmethod
    def _pd_to_np(df):
        return Sampler._pd_to_np(df)

    def is_trained(self, K, J, C):
        """Indicates whether the Decorrelator has been trained
        on a specific RFI set C for features J.

        Args:
            C: RFI set C to be checked

        Returns:
            Whether the sampler was trained with respect to
            a set C.
        """
        K_key = Decorrelator._to_key(K)
        C_key, J_key = Decorrelator._to_key(C), Decorrelator._to_key(J)
        trained = (K_key, J_key, C_key) in self._trained_decorrelation_funcs
        return trained

    # def _train_J_degenerate(self, J, C, verbose=True):
    #     """Training function that takes care of degenerate cases
    #     where either j is in C or C is empty.
    #     Args:
    #         J: features of interest
    #         C: relative feature set

    #     Returns:
    #         Whether a degenerate case was present.
    #     """
    #     degenerate = True

    #     # are we conditioning on zero elements?
    #     if C.size == 0:
    #         logger.debug('Degenerate Training: Empty C')
    #         self._store_samplefunc(J, C, sample_perm(J))
    #     # are all elements in C being conditioned upon?
    #     elif np.sum(1 - np.isin(J, C)) == 0:
    #         logger.debug('Degenerate Training: J subseteq C')
    #         self._store_samplefunc(J, C, sample_id(J))
    #     else:
    #         logger.debug('Training not degenerate.')
    #         degenerate = False

    #     return degenerate

    def _store_decorrelationfunc(self, K, J, C,
                                 samplefunc,
                                 verbose=True):
        """Storing a trained sample function

        Args:
            samplefunc: function
            K: features of interest
            J: set of features to be removed
            C: relative feature set
            verbose: printing or not
        """
        K_key, C_key, J_key = Decorrelator._to_key(
            K), Decorrelator._to_key(C), Decorrelator._to_key(J)
        self._trained_decorrelation_funcs[(K_key, J_key, C_key)] = samplefunc
        logger.info('Training ended. Decorrelator saved.')

    def train(self, K, J, C, verbose=True):
        """Trains decorrelator such that X_K idp X_J | X_C

        Args:
            K: variable set
            J: variable set
            C: variable set

        Returns:
            Nothing. Now the sample function can be used
            to resample on seen or unseen data.
        """
        logger.info('Training Decorrelator for: {} ⟂ {} | {}'.format(K, J, C))
        if (np.intersect1d(J, C).shape[0] > 0):
            raise RuntimeError(
                'J and C share indices, but should be dinjoint.')

    def decorrelate(self, X_test, K, J,
                    C, num_samples=1):
        """Sample features of interest using trained resampler.

        Args:
            J: Set of features to sample
            C: relative feature set
            X_test: Data for which sampling shall be
                performed. (format as self.X_train)
            num_samples: number of resamples without
                retraining shall be computed

        Returns:
            Resampled data for the features of interest.
            np.array with shape
            (X_test.shape[0], #num_samples, # features of interest)
        """
        # initialize numpy matrix
        # sampled_data = np.zeros((X_test.shape[0], num_samples, J.shape[0]))

        # sample
        K_key, C_key, J_key = Decorrelator._to_key(
            K), Decorrelator._to_key(C), Decorrelator._to_key(J)

        if num_samples != 1:
            raise NotImplementedError('only num_samples 1 implemented.')

        if not self.is_trained(K, J, C):
            raise RuntimeError(
                "Decorrelator not trained for {} ⟂ {} | {}".format(K, J, C))
        else:
            decorrf = self._trained_decorrelation_funcs[(K_key, J_key, C_key)]
            #X_test_np = Decorrelator._pd_to_np(X_test)  # ordering columns
            decorrelated_data = decorrf(X_test)
            return decorrelated_data
