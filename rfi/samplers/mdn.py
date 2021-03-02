"""
Sampler based on conditional gaussian mixture networks
"""
import logging

from rfi.samplers.sampler import Sampler
from rfi.utils import search_nonsorted
from rfi.backend.mdn import MixtureDensityNetworkEstimator
from rfi.backend.categorical import CategoricalEstimator

logger = logging.getLogger(__name__)


class MDNSampler(Sampler):
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
                raise NotImplementedError('Multiple categorical or mixed '
                                          'variables sampling is not '
                                          'supported.')

            train_inputs = self.X_train[sorted(set(J))].to_numpy()
            train_context = self.X_train[sorted(set(G))].to_numpy()

            # Categorical variables in context
            cat_context = None
            if not set(G).isdisjoint(self.cat_inputs):
                G_cat = list(set(G).intersection(self.cat_inputs))
                cat_ixs = utils.fset_to_ix(sorted(G), sorted(G_cat))
                # cat_context = search_nonsorted(G, G_cat)
                logger.info('One hot encoding following context '
                            'features: {}'.format(G_cat))

            # Categorical variable as input
            if not set(J).isdisjoint(self.cat_inputs):
                logger.info('One hot encoding following inputs '
                            'features: {}'.format(J))
                logger.info('Fitting categorical sampler for '
                            'features {}. '.format(J)
                            'Fitting method: '
                            '{}. '.format(self.fit_method)
                            'Fitting parameters: '
                            '{}'.format(self.fit_params))
                context_size = train_context.shape[1]
                model = CategoricalEstimator(context_size=context_size,
                                             cat_context=cat_ixs,
                                             **self.fit_params)
            # Continuous variable as input
            else:
                logger.info(f'Fitting continuous sampler for '
                            f'features {J}. Fitting method: '
                            f'{self.fit_method}. '
                            f'Fitting parameters: {self.fit_params}')
                cntxt_sz = train_context.shape[1]
                model = MixtureDensityNetworkEstimator(inputs_size=len(J),
                                                       context_size=cntxt_sz,
                                                       cat_context=cat_context,
                                                       **self.fit_params)

            # Fitting a sampler
            getattr(model, self.fit_method)(train_inputs=train_inputs,
                                            train_context=train_context,
                                            **self.fit_params)

            def samplefunc(eval_context, **kwargs):
                return model.sample(eval_context, **kwargs)

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)

            return model

        else:
            return None
