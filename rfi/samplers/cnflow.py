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

    def _train_j(self, j, G):

        G_key, j_key = utils.to_key(G), utils.to_key([j])

        if j in G:
            self._trainedGs[(j_key, G_key)] = sample_id(j)
        elif G.size == 0:
            self._trainedGs[(j_key, G_key)] = sample_perm(j)
        else:
            logger.info(f'Fitting sampler for feature {j}. Time budget for CV search: {self.time_budget_s} sec')
            cnf = ConditionalNormalisingFlowEstimator(context_size=len(G))
            cnf.fit_by_cv(train_inputs=self.X_train[:, j], train_context=self.X_train[:, G], time_budget_s=self.time_budget_s)
            self._trainedGs[(j_key, G_key)] = lambda X_test: cnf.sample(X_test[:, G]).reshape(-1)

    def train(self, J, G):
        J = np.array(J, dtype=np.int16).reshape(-1)
        G = np.array(G, dtype=np.int16)
        for j in J:
            self._train_j(j, G)

    def sample(self, X_test, J, G):
        J = np.array(J, dtype=np.int16).reshape(-1)
        G = np.array(G, dtype=np.int16)
        sampled_data = np.zeros((X_test.shape[0], J.shape[0]))
        G_key = utils.to_key(G)

        for j_ind, j in enumerate(J):
            j_key = utils.to_key([j])
            # TODO(gcsk): check if trained
            sampled_data[:, j_ind] = self._trainedGs[(j_key, G_key)](X_test)

        return sampled_data
