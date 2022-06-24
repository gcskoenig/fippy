from src.rfi.decorrelators.decorrelator import Decorrelator
from src.rfi.backend.gaussian import GaussianConditionalEstimator
from src.rfi.samplers.gaussian import GaussianSampler
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NaiveGaussianDecorrelator(Decorrelator):
    """
    Naive implementation that just independently resamples
    the variables to be dropped.
    """
    def __init__(self, X_train, X_val=None, **kwargs):
        super().__init__(X_train, X_val=True)

    def train(self, K, J, C):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            K: features to be knocked out from J cup C to C
            J: features on top of baseline
            C: baseline conditioning set
        """
        K = Decorrelator._to_array(Decorrelator._order_fset(K))
        J = Decorrelator._to_array(Decorrelator._order_fset(J))
        C = Decorrelator._to_array(Decorrelator._order_fset(C))
        super().train(K, J, C)

        if len(K) > 0:
            drop_sampler = GaussianSampler(self.X_train)
            drop_sampler.train(K, C)

        def decorrelationfunc(X_test):
            values_test = X_test.copy()

            # index in K that are in J can directly be resampled using C
            if len(K) > 0:
                smpl = drop_sampler.sample(X_test, K, C)
                values_test.loc[:, K] = smpl.to_numpy()

            return values_test  # pandas dataframe

        self._store_decorrelationfunc(K, J, C,
                                      decorrelationfunc)

        return None  # TODO implement returning/saving estimators


class GaussianDecorrelator(Decorrelator):
    """
    Second order Gaussian Decorrelator.

    Attributes:
        see rfi.decorellators.decorrelator
    """

    def __init__(self, X_train, X_val=None, **kwargs):
        """Initialize Decorrelator with X_train and mask."""
        super().__init__(X_train, X_val=X_val)

    def train(self, K, J, C):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            K: features to be knocked out from J cup C to C
            J: features on top of baseline
            C: baseline conditioning set
        """
        K = Decorrelator._to_array(Decorrelator._order_fset(K))
        J = Decorrelator._to_array(Decorrelator._order_fset(J))
        C = Decorrelator._to_array(Decorrelator._order_fset(C))
        super().train(K, J, C)

        K_intersect_J = np.intersect1d(K, J)  # resample from X^C directly
        K_leftover = np.setdiff1d(np.setdiff1d(K, J), C)  # actually decorrelate
        K_intersect_J = Decorrelator._order_fset(K_intersect_J)
        K_leftover = Decorrelator._order_fset(K_leftover)

        if len(K_intersect_J) > 0:
            intersectsampler = GaussianSampler(self.X_train)
            intersectsampler.train(K_intersect_J, C)

        estimators = []  # will be filled with tuples (cdf, icdf)

        # sort K_leftover by decreasing partial correlation
        # TODO(gcsk) if partial correlation 1 or 0 replace with the respective originals
        ordering = None
        if len(K_leftover) > 0:
            if len(C) > 0:
                corrs = []
                for kk in range(len(K_leftover)):
                    abs_sum = 0
                    for jj in range(len(J)):
                        r = self.X_train.partial_corr(x=K_leftover[kk],
                                                      y=J[kk], covar=C)
                        abs_sum = abs_sum + abs(r)
                    corrs.append(abs_sum)
                ordering = np.argsort(corrs)
            else:
                corr_coef = self.X_train.corr().abs()
                ordering = np.array(corr_coef.loc[J, K_leftover].sum().argsort())

        K_leftover = np.array(K_leftover)[ordering].flatten()

        for jj in range(len(K_leftover)):
            logger.debug('{}(k) idp {} | {}'.format(K_leftover[jj], J, C))

            estimator_cdf = GaussianConditionalEstimator()
            estimator_icdf = GaussianConditionalEstimator()

            # build context and target, train
            # here: K_leftover[jj] | J cup C cup already_perturbed
            J_and_C = sorted(set(J).union(C))
            Ptbd_cdf = sorted(set(K_leftover[:jj]).union(J_and_C))  # UNION is a mistake?

            tupl = (self.X_train.loc[:, J_and_C].to_numpy(),  # ERROR remove this line?
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
                Ptbd_cdf = sorted(set(K_leftover[:jj]).union(J_and_C))  # ERROR ?
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
                #                          make_uniform=True)
                values_test.loc[:, K_leftover[jj]] = tmp

            return values_test  # pandas dataframe

        self._store_decorrelationfunc(K, J, C,
                                      decorrelationfunc)

        return None  # TODO implement returning/saving estimators
