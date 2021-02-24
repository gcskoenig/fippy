"""
Sampler based on conditional normalizing flows. Using affine and invertable radial transformations.
"""
import logging

from rfi.samplers.sampler import Sampler
from rfi.backend.cnf import NormalisingFlowEstimator
from rfi.backend.categorical import CategoricalEstimator
from rfi.utils import search_nonsorted

logger = logging.getLogger(__name__)


class CNFSampler(Sampler):
    def __init__(self, X_train, fit_method='fit_by_cv', fit_params={'time_budget_s': None}, **kwargs):
        super().__init__(X_train, **kwargs)
        self.fit_method = fit_method
        self.fit_params = fit_params if fit_params is not None else {}

    def train(self, J, G, verbose=True):

        J = Sampler._to_array(J)
        G = Sampler._to_array(G)
        super().train(J, G, verbose=verbose)

        if not self._train_J_degenerate(J, G, verbose=verbose):

            if not set(J).isdisjoint(self.cat_inputs) and len(J) > 1:
                raise NotImplementedError('Multiple categorical or mixed variables sampling is not supported.')

            train_inputs = self.X_train[:, J]
            train_context = self.X_train[:, G]

            # Categorical variables in context
            cat_context = None
            if not set(G).isdisjoint(self.cat_inputs):
                G_cat = Sampler._to_array(list(set(G).intersection(self.cat_inputs)))
                cat_context = search_nonsorted(G, G_cat)
                logger.info(f'One hot encoding following context features: {G_cat}')

            # Categorical variable as input
            if not set(J).isdisjoint(self.cat_inputs):
                logger.info(f'One hot encoding following inputs features: {J}')
                logger.info(f'Fitting categorical sampler for features {J}. Fitting method: {self.fit_method}. '
                            f'Fitting parameters: {self.fit_params}')
                model = CategoricalEstimator(context_size=train_context.shape[1], cat_context=cat_context, **self.fit_params)
            # Continuous variable as input
            else:
                logger.info(f'Fitting continuous sampler for features {J}. Fitting method: {self.fit_method}. '
                            f'Fitting parameters: {self.fit_params}')
                model = NormalisingFlowEstimator(inputs_size=len(J), context_size=train_context.shape[1], cat_context=cat_context,
                                                 **self.fit_params)

            # Fitting a sampler
            getattr(model, self.fit_method)(train_inputs=train_inputs, train_context=train_context, **self.fit_params)

            def samplefunc(X_test, **kwargs):
                return model.sample(X_test[:, G], **kwargs)

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)

            return model

        else:
            return None
