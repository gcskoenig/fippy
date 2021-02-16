from rfi.decorrelators.decorrelator import Decorrelator
from rfi.backend.gaussian import GaussianConditionalEstimator
from rfi.samplers.gaussian import GaussianSampler
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GaussianDecorrelator(Decorrelator):
    """
    Second order Gaussian Decorrelator.

    Attributes:
        see rfi.decorellators.decorrelator
    """

    def __init__(self, X_train, X_val=None, **kwargs):
        """Initialize Decorrelator with X_train and mask."""
        super().__init__(X_train, X_val=X_val)

    def train(self, K, J, C, verbose=True):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            J: features of interest
            G: arbitrary set of variables
            verbose: printing
        """
        K = Decorrelator._to_array(K)
        J = Decorrelator._to_array(J)
        C = Decorrelator._to_array(C)
        super().train(K, J, C, verbose=verbose)

        #if not self._train_J_degenerate(K, J, C, verbose=verbose):
        if not False:
            K_intersect_J = np.intersect1d(K, J)
            K_intersect_C = np.intersect1d(K, C)
            K_leftover = np.setdiff1d(np.setdiff1d(K, J), C)

            intersect_sampler = None

            if K_intersect_J.shape[0] > 0:
                intersectsampler = GaussianSampler(self.X_train)
                intersectsampler.train(K_intersect_J, C)
                #values_train[:, K_intersect] = np.transpose(sampler.sample(X_train, K_intersect, C), (0, 1, 2)).reshape(values_train.shape[0], -1)
                
            estimators = [] # will be filled with tuples (cdf, icdf)

            for jj in np.arange(K_leftover.shape[0]):
                logger.debug('k: {}, J: {}, C: {}'.format(K_leftover[jj], J, C))    

                estimator_cdf = GaussianConditionalEstimator()
                estimator_icdf = GaussianConditionalEstimator()

                ixs_perturbed_cdf = np.setdiff1d(K_leftover[:jj], np.union1d(J, C))
                context_cdf_fit = np.concatenate((self.X_train[:, np.union1d(J, C)], self.X_train[:, ixs_perturbed_cdf]), axis=1) # for fit
                estimator_cdf.fit(self.X_train[:, K_leftover[jj]], context_cdf_fit)

                logger.debug('estimator cdf, Sigma: {}, RegrCoef: {}'.format(estimator_cdf.Sigma, estimator_cdf.RegrCoeff))

                ixs_perturbed_icdf = np.setdiff1d(K_leftover[:jj], C)
                context_icdf_fit = np.concatenate((self.X_train[:, C], self.X_train[:, ixs_perturbed_icdf]), axis=1) # for fit
                estimator_icdf.fit(self.X_train[:, K_leftover[jj]], context_icdf_fit)
                
                estimators.append((estimator_cdf, estimator_icdf))
                logger.debug('estimator cdf, Sigma: {}, RegrCoef: {}'.format(estimator_icdf.Sigma, estimator_icdf.RegrCoeff))



            gaussian_estimator = GaussianConditionalEstimator()
            gaussian_estimator.fit(train_inputs=self.X_train[:, J], train_context=self.X_train[:, C])

            def decorrelationfunc(X_test):
                values_test = np.zeros(X_test.shape)
                
                if K_intersect_J.shape[0] > 0:
                    values_test[:, K_intersect_J] = np.transpose(intersectsampler.sample(X_test, K_intersect_J, C), (0, 1, 2)).reshape(values_test.shape[0], -1)

                if K_intersect_C.shape[0] > 0:
                    values_test[:, K_intersect_C] = X_test[:, K_intersect_C]

                for jj in np.arange(K_leftover.shape[0]):
                    estimator_cdf, estimator_icdf = estimators[jj]

                    ixs_perturbed_cdf = np.setdiff1d(K_leftover[:jj], np.union1d(J, C))
                    context_cdf_test = np.concatenate((X_test[:, np.union1d(J, C)], values_test[:, ixs_perturbed_cdf]), axis=1) # for cdf        
                    quantiles_test = estimator_cdf.cdf(X_test[:, K_leftover[jj]], context_cdf_test)

                    ixs_perturbed_icdf = np.setdiff1d(K_leftover[:jj], C)
                    context_icdf_test = np.concatenate((X_test[:, C], values_test[:, ixs_perturbed_icdf]), axis=1) # for cdf
                    values_test[:, K_leftover[jj]] = estimator_icdf.icdf(quantiles_test, context_icdf_test)

                return values_test

            self._store_decorrelationfunc(K, J, C, decorrelationfunc, verbose=verbose)

            return gaussian_estimator
        else:
            return None