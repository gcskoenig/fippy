"""
Sampler based on conditional normalizing flows. Using affine and invertable radial transformations.
"""
import logging

from rfi.samplers.sampler import Sampler
from rfi.backend.cnf import NormalisingFlowEstimator

logger = logging.getLogger(__name__)


class CNFSampler(Sampler):
    def __init__(self, X_train, fit_method='fit_by_cv', fit_params={'time_budget_s': None}, **kwargs):
        super().__init__(X_train)
        self.fit_method = fit_method
        self.fit_params = fit_params if fit_params is not None else {}

    def train(self, J, G, verbose=True):

        J = Sampler._to_array(J)
        G = Sampler._to_array(G)
        super().train(J, G, verbose=verbose)

        if not self._train_J_degenerate(J, G, verbose=verbose):
            logger.info(f'Fitting sampler for features {J}. Fitting method: {self.fit_method}. '
                        f'Fitting parameters: {self.fit_params}')

            cnf = NormalisingFlowEstimator(inputs_size=len(J), context_size=len(G), **self.fit_params)
            getattr(cnf, self.fit_method)(train_inputs=self.X_train[:, J], train_context=self.X_train[:, G],
                                          **self.fit_params)

            def samplefunc(X_test, **kwargs):
                return cnf.sample(X_test[:, G], **kwargs)

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)

            return cnf

        else:
            return None
