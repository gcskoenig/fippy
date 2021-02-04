"""
Sampler based on conditional normalizing flows. Using affine and invertable radial transformations.
"""
import numpy as np
import logging

from rfi.samplers.sampler import Sampler
from rfi import utils
from rfi.backend.cnf import NormalisingFlowEstimator
from rfi.samplers._utils import sample_id, sample_perm

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

            if len(J) > 1:
                raise NotImplementedError()

            cnf = NormalisingFlowEstimator(inputs_size=len(J), context_size=len(G))
            getattr(cnf, self.fit_method)(train_inputs=self.X_train[:, J], train_context=self.X_train[:, G],
                                          **self.fit_params)

            def samplefunc(X_test, **kwargs):
                return cnf.sample(X_test[:, G], **kwargs)

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)

            return cnf

        else:
            return None
        #
        #     if self.X_val is not None:
        #         val_log_prob = cnf.log_prob(inputs=self.X_val[:, j], context=self.X_val[:, G]).mean()
        #         val_log_probs.append(val_log_prob)
        #
        # elif self.X_val is not None:
        #     val_log_probs.append(None)
        #
        # if len(J) > 1:
        #
        #     def samplefunc(X_test, **kwargs):
        #         sampled_data = []
        #         for j in J:
        #             j = Sampler._to_array([j])
        #             G_key, j_key = Sampler._to_key(G), Sampler._to_key(j)
        #             sampled_data.append(np.squeeze(self._trained_sampling_funcs[(j_key, G_key)](X_test, **kwargs)))
        #         return np.stack(sampled_data, axis=-1)
        #
        #     self._store_samplefunc(J, G, samplefunc, verbose=verbose)
        #
        # if self.X_val is not None:
        #     return val_log_probs
