"""TypeDispatchSampler: routes to continuous or categorical sub-sampler."""
import pandas as pd

from fippy.samplers.base import Sampler


class TypeDispatchSampler(Sampler):
    """Routes to continuous or categorical sub-sampler based on target dtype.

    Univariate only (len(J) = 1). For multivariate use, wrap in SequentialSampler.

    Example:
        sampler = TypeDispatchSampler(
            X_train,
            continuous_sampler=RFResidualSampler(X_train),
            categorical_sampler=RFClassificationSampler(X_train),
        )
    """

    multivariate = False
    supports_categorical_target = True

    def __init__(
        self,
        X_train: pd.DataFrame,
        continuous_sampler: Sampler,
        categorical_sampler: Sampler,
    ):
        super().__init__(X_train)
        self._continuous = continuous_sampler
        self._categorical = categorical_sampler

    @property
    def supports_categorical_context(self):
        return (self._continuous.supports_categorical_context
                and self._categorical.supports_categorical_context)

    def _select_sampler(self, J):
        assert len(J) == 1
        if J[0] in self._categorical_cols:
            return self._categorical
        return self._continuous

    def _fit(self, J, S):
        self._select_sampler(J)._fit(J, S)

    def _sample(self, X, J, S, n_samples):
        return self._select_sampler(J)._sample(X, J, S, n_samples)
