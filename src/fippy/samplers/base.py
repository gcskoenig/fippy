"""Sampler ABC: base class for all conditional distribution samplers."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Sampler(ABC):
    """Base class for all conditional distribution samplers.

    Subclasses declare three capability flags as class attributes:
        multivariate: Can sample len(J) > 1 natively.
        supports_categorical_target: Can produce samples for categorical features in J.
        supports_categorical_context: Can condition on categorical features in S.

    Categorical columns are detected from DataFrame dtypes (CategoricalDtype, object).
    """

    multivariate: bool = False
    supports_categorical_target: bool = False
    supports_categorical_context: bool = False

    def __init__(self, X_train: pd.DataFrame):
        self.X_train = X_train
        self.feature_names = list(X_train.columns)
        self._categorical_cols = self._detect_categorical(X_train)

    @staticmethod
    def _detect_categorical(X: pd.DataFrame) -> set[str]:
        """Detect categorical columns from dtype."""
        return {
            col
            for col in X.columns
            if isinstance(X[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(X[col])
        }

    def check_compatibility(self, requires_multivariate: bool = False):
        """Pre-flight validation: check sampler compatibility with the dataset.

        Called by the Explainer at the start of loo()/shapley(), before the
        feature loop. The Explainer determines what properties are required
        and passes them in.

        Args:
            requires_multivariate: Whether the computation requires sampling
                len(J) > 1 (e.g., Shapley always, LOO with multi-column groups).

        Raises:
            ValueError: Comprehensive error listing all incompatibilities.
        """
        errors = []

        cat_cols = self._categorical_cols
        if cat_cols:
            if not self.supports_categorical_target:
                errors.append(
                    f"Categorical columns {cat_cols} will appear as targets, "
                    f"but {type(self).__name__} does not support categorical "
                    f"targets. Use a sampler that supports categorical targets "
                    f"(e.g., PermutationSampler, ARFSampler) or wrap with "
                    f"TypeDispatchSampler."
                )
            if not self.supports_categorical_context:
                errors.append(
                    f"Categorical columns {cat_cols} will appear in conditioning "
                    f"sets, but {type(self).__name__} does not support categorical "
                    f"context features."
                )

        if requires_multivariate and not self.multivariate:
            errors.append(
                f"{type(self).__name__} is univariate (multivariate=False) "
                f"and cannot sample multiple features jointly. "
                f"Wrap it in SequentialSampler."
            )

        if errors:
            raise ValueError(
                f"Sampler {type(self).__name__} is incompatible with the "
                f"requested computation:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def fit(self, J, S):
        """Fit P(X_J | X_S)."""
        self._fit(J, S)

    def sample(self, X, J, S, n_samples=1):
        """Sample from P(X_J | X_S). Shape: (n_obs, n_samples, len(J))."""
        return self._sample(X, J, S, n_samples)

    @abstractmethod
    def _fit(self, J, S): ...

    @abstractmethod
    def _sample(self, X, J, S, n_samples) -> np.ndarray: ...
