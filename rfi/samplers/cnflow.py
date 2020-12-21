"""
Sampler based on conditional normalizing flows. Using affine and invertable radial transformations.
"""
import numpy as np
import logging

from rfi.samplers.sampler import Sampler
from rfi import utils
from rfi.backend.cnf import ConditionalNormalisingFlowEstimator
from rfi.samplers._utils import sample_id, sample_perm

logger = logging.getLogger(__name__)


class CNFSampler(Sampler):
    def __init__(self, X_train):
        super().__init__(X_train)
        self.time_budget_s = 120

    def train(self, J, G, verbose=True):

        J = Sampler._to_array(J)
        G = Sampler._to_array(G)
        super().train(J, G, verbose=verbose)

        if not super()._train_J_degenerate(J, G, verbose=verbose):
            logger.info(f'Fitting sampler for feature {j}. Time budget for CV search: {self.time_budget_s} sec')
            cnf = ConditionalNormalisingFlowEstimator(context_size=len(G))
            cnf.fit_by_cv(train_inputs=self.X_train[:, J], train_context=self.X_train[:, G], 
                          time_budget_s=self.time_budget_s)
            samplefunc = lambda X_test: cnf.sample(X_test[:, G]).reshape(-1)
            super()._store_samplefunc(J, G, samplefunc, verbose=verbose)