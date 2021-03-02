from rfi.decorrelators.decorrelator import Decorrelator
from rfi.backend.gaussian import GaussianConditionalEstimator
from rfi.samplers.gaussian import GaussianSampler
import rfi.utils as utils
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
            K: features to be knocked out from J cup C to C
            J: features on top of baseline
            C: baseline conditioning set
            verbose: printing
        """
        K = Decorrelator._to_array(Decorrelator._order_fset(K))
        J = Decorrelator._to_array(Decorrelator._order_fset(J))
        C = Decorrelator._to_array(Decorrelator._order_fset(C))
        super().train(K, J, C, verbose=verbose)

        K_intersect_J = np.intersect1d(K, J)
        K_leftover = np.setdiff1d(np.setdiff1d(K, J), C)
        K_intersect_J = Decorrelator._order_fset(K_intersect_J)
        K_leftover = Decorrelator._order_fset(K_leftover)

        if len(K_intersect_J) > 0:
            intersectsampler = GaussianSampler(self.X_train)
            intersectsampler.train(K_intersect_J, C)

        estimators = []  # will be filled with tuples (cdf, icdf)

        for jj in range(len(K_leftover)):
            logger.debug('{}(k) idp {} | {}'.format(K_leftover[jj], J, C))

            estimator_cdf = GaussianConditionalEstimator()
            estimator_icdf = GaussianConditionalEstimator()

            # build context and target, train
            # here: K_leftover[jj] | J cup C cup already_perturbed
            J_and_C = sorted(set(J).union(C))
            Ptbd_cdf = sorted(set(K_leftover[:jj]).union(J_and_C))

            tupl = (self.X_train.loc[:, J_and_C].to_numpy(),
                    self.X_train.loc[:, Ptbd_cdf].to_numpy())
            context_cdf_fit = np.concatenate(tupl, axis=1)  # for fit
            estimator_cdf.fit(self.X_train.loc[:, K_leftover[jj]].to_numpy(),
                              context_cdf_fit)
            txt = 'estimator cdf, '
            txt = txt + 'Sigma: {}, '.format(estimator_cdf.Sigma)
            txt = txt + 'RegrCoef: {}'.format(estimator_cdf.RegrCoeff)
            logger.debug(txt)

            # build context and target, train
            # here: K_leftover[jj] | C cup already_perturbed
            Ptbd_icdf = sorted(set(K_leftover[:jj]).difference(C))
            tpl = (self.X_train.loc[:, C].to_numpy(),
                   self.X_train.loc[:, Ptbd_icdf].to_numpy())
            context_icdf_fit = np.concatenate(tpl, axis=1)  # for fit
            estimator_icdf.fit(self.X_train.loc[:, K_leftover[jj]].to_numpy(),
                               context_icdf_fit)
            estimators.append((estimator_cdf, estimator_icdf))
            txt = 'estimator icdf, '
            txt = txt + 'Sigma: {}, '.format(estimator_icdf.Sigma)
            txt = txt + 'RegrCoef: {}'.format(estimator_icdf.RegrCoeff)
            logger.debug(txt)

        def decorrelationfunc(X_test):
            values_test = X_test.copy()

            # index in K that are in J can directly be resampled using C
            if len(K_intersect_J) > 0:
                smpl = intersectsampler.sample(X_test, K_intersect_J, C)
                values_test.loc[:, K_intersect_J] = smpl.to_numpy()

            # the remaining features in K have to be decorrelated
            for jj in range(len(K_leftover)):
                estimator_cdf, estimator_icdf = estimators[jj]

                # build context and target, sample
                # here: quantiles of K_leftover[jj] | J and C and perturbed
                J_and_C = sorted(set(J).union(C))
                Ptbd_cdf = sorted(set(K_leftover[:jj]).union(J_and_C))
                tpl = (X_test.loc[:, J_and_C].to_numpy(),
                       values_test.loc[:, Ptbd_cdf].to_numpy())
                context_cdf_test = np.concatenate(tpl, axis=1)  # for cdf
                inputs_cdf_test = X_test.loc[:, K_leftover[jj]].to_numpy()
                qs_test = estimator_cdf.cdf(inputs_cdf_test,
                                            context_cdf_test)

                # build context and target, sample
                # here: values of K_leftover[jj] | quantiles of C and perturbed
                Ptbd_icdf = sorted(set(K_leftover[:jj]).difference(C))
                tpl = (X_test.loc[:, C].to_numpy(),
                       values_test.loc[:, Ptbd_icdf].to_numpy())
                context_icdf_test = np.concatenate(tpl, axis=1)  # for cdf
                tmp = estimator_icdf.icdf(qs_test, context_icdf_test)
                values_test.loc[:, K_leftover[jj]] = tmp

            return values_test  # pandas dataframe

        self._store_decorrelationfunc(K, J, C,
                                      decorrelationfunc,
                                      verbose=verbose)

        return None  # TODO implement returning/saving estimators
