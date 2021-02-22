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

        if not False:
            K_intersect_J = np.intersect1d(K, J)
            K_leftover = np.setdiff1d(np.setdiff1d(K, J), C)

            if K_intersect_J.shape[0] > 0:
                intersectsampler = GaussianSampler(self.X_train)
                intersectsampler.train(K_intersect_J, C)

            estimators = []  # will be filled with tuples (cdf, icdf)

            for jj in np.arange(K_leftover.shape[0]):
                logger.debug('{}(k) idp {} | {}'.format(K_leftover[jj], J, C))

                estimator_cdf = GaussianConditionalEstimator()
                estimator_icdf = GaussianConditionalEstimator()

                ixs_perturbed_cdf = np.setdiff1d(K_leftover[:jj],
                                                 np.union1d(J, C))
                tupl = (self.X_train[:, np.union1d(J, C)],
                        self.X_train[:, ixs_perturbed_cdf])
                context_cdf_fit = np.concatenate(tupl, axis=1)  # for fit
                estimator_cdf.fit(self.X_train[:, K_leftover[jj]],
                                  context_cdf_fit)
                txt = 'estimator cdf, '
                txt = txt + 'Sigma: {}, '.format(estimator_cdf.Sigma)
                txt = txt + 'RegrCoef: {}'.format(estimator_cdf.RegrCoeff)
                logger.debug(txt)

                ixs_perturbed_icdf = np.setdiff1d(K_leftover[:jj], C)
                tpl = (self.X_train[:, C], self.X_train[:, ixs_perturbed_icdf])
                context_icdf_fit = np.concatenate(tpl, axis=1)  # for fit
                estimator_icdf.fit(self.X_train[:, K_leftover[jj]],
                                   context_icdf_fit)
                estimators.append((estimator_cdf, estimator_icdf))
                txt = 'estimator icdf, '
                txt = txt + 'Sigma: {}, '.format(estimator_icdf.Sigma)
                txt = txt + 'RegrCoef: {}'.format(estimator_icdf.RegrCoeff)
                logger.debug(txt)

            def decorrelationfunc(X_test):
                values_test = np.array(X_test)

                if K_intersect_J.shape[0] > 0:
                    smpl = intersectsampler.sample(X_test, K_intersect_J, C)
                    smpl = np.transpose(smpl, (0, 1, 2))
                    smpl = smpl.reshape(values_test.shape[0],
                                        K_intersect_J.shape[0])
                    values_test[:, K_intersect_J] = smpl

                for jj in np.arange(K_leftover.shape[0]):
                    estimator_cdf, estimator_icdf = estimators[jj]

                    ixs_perturbed_cdf = np.setdiff1d(K_leftover[:jj],
                                                     np.union1d(J, C))
                    tpl = (X_test[:, np.union1d(J, C)],
                           values_test[:, ixs_perturbed_cdf])
                    context_cdf_test = np.concatenate(tpl, axis=1)  # for cdf
                    qs_test = estimator_cdf.cdf(X_test[:, K_leftover[jj]],
                                                context_cdf_test)

                    ixs_perturbed_icdf = np.setdiff1d(K_leftover[:jj], C)
                    tpl = (X_test[:, C], values_test[:, ixs_perturbed_icdf])
                    context_icdf_test = np.concatenate(tpl, axis=1)  # for cdf
                    tmp = estimator_icdf.icdf(qs_test, context_icdf_test)
                    values_test[:, K_leftover[jj]] = tmp

                return values_test

            self._store_decorrelationfunc(K, J, C,
                                          decorrelationfunc,
                                          verbose=verbose)

            return None  # TODO implement returning/saving estimators
        else:
            return None
