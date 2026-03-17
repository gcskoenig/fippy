"""Forest-based univariate conditional samplers."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from fippy.samplers.base import Sampler

_RF_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}


class _ForestSamplerBase(Sampler):
    """Shared logic for RF-based univariate samplers."""

    def __init__(self, X_train: pd.DataFrame, *, tune: bool = True,
                 n_iter: int = 20, cv: int = 5, random_state=None):
        super().__init__(X_train)
        self.tune = tune
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self._cache: dict[tuple, tuple] = {}
        # Build encoding maps for categorical context features.
        self._cat_maps: dict[str, dict] = {
            col: {v: i for i, v in enumerate(X_train[col].unique())}
            for col in self._categorical_cols
        }

    def _context_array(self, X, S):
        """Convert context columns to numeric numpy array for sklearn."""
        if not S:
            return np.zeros((len(X), 0))
        cols = sorted(S)
        arrays = []
        for col in cols:
            if col in self._categorical_cols:
                mapping = self._cat_maps[col]
                codes = np.array([mapping.get(v, -1) for v in X[col]], dtype=float)
                arrays.append(codes)
            else:
                arrays.append(X[col].values.astype(float))
        return np.column_stack(arrays)

    def _tune_and_fit(self, estimator, X_context, y, scoring=None):
        """Fit estimator, optionally with hyperparameter tuning."""
        if self.tune and len(y) >= self.cv:
            search = RandomizedSearchCV(
                estimator,
                _RF_PARAM_GRID,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=-1,
            )
            search.fit(X_context, y)
            return search.best_estimator_
        estimator.fit(X_context, y)
        return estimator

    @staticmethod
    def _key(J, S):
        return (tuple(sorted(J)), tuple(sorted(S)))


class RFResidualSampler(_ForestSamplerBase):
    """Univariate conditional sampler for continuous targets.

    Fits a Random Forest regressor to estimate E[X_j | X_S] and stores training
    residuals. At sample time, predicts the conditional expectation and adds a
    randomly resampled training residual.

    Assumes homoscedastic residuals: the residual distribution does not depend
    on X_S.
    """

    multivariate = False
    supports_categorical_target = False
    supports_categorical_context = True

    def _fit(self, J, S):
        key = self._key(J, S)
        if key in self._cache:
            return

        assert len(J) == 1
        y_train = self.X_train[J[0]].values.astype(float)

        if not S:
            mean = y_train.mean()
            self._cache[key] = ("marginal", mean, y_train - mean)
            return

        X_ctx = self._context_array(self.X_train, S)
        model = self._tune_and_fit(
            RandomForestRegressor(random_state=self.random_state),
            X_ctx, y_train,
        )
        residuals = y_train - model.predict(X_ctx)
        self._cache[key] = ("fitted", model, residuals)

    def _sample(self, X, J, S, n_samples):
        key = self._key(J, S)
        if key not in self._cache:
            self._fit(J, S)

        entry = self._cache[key]
        n_obs = len(X)
        result = np.empty((n_obs, n_samples, 1))

        if entry[0] == "marginal":
            mean, residuals = entry[1], entry[2]
            for k in range(n_samples):
                idx = np.random.randint(0, len(residuals), size=n_obs)
                result[:, k, 0] = mean + residuals[idx]
        else:
            model, residuals = entry[1], entry[2]
            preds = model.predict(self._context_array(X, S))
            for k in range(n_samples):
                idx = np.random.randint(0, len(residuals), size=n_obs)
                result[:, k, 0] = preds + residuals[idx]

        return result


class RFClassificationSampler(_ForestSamplerBase):
    """Univariate conditional sampler for categorical targets.

    Fits a Random Forest classifier to estimate P(X_j | X_S). At sample time,
    predicts class probabilities and samples from them.
    """

    multivariate = False
    supports_categorical_target = True
    supports_categorical_context = True

    def _fit(self, J, S):
        key = self._key(J, S)
        if key in self._cache:
            return

        assert len(J) == 1
        y_train = self.X_train[J[0]].values

        if not S:
            classes, counts = np.unique(y_train, return_counts=True)
            self._cache[key] = ("marginal", classes, counts / counts.sum())
            return

        X_ctx = self._context_array(self.X_train, S)
        model = self._tune_and_fit(
            RandomForestClassifier(random_state=self.random_state),
            X_ctx, y_train,
            scoring="neg_log_loss",
        )
        self._cache[key] = ("fitted", model)

    def _sample(self, X, J, S, n_samples):
        key = self._key(J, S)
        if key not in self._cache:
            self._fit(J, S)

        entry = self._cache[key]
        n_obs = len(X)
        result = np.empty((n_obs, n_samples, 1), dtype=object)

        if entry[0] == "marginal":
            classes, probs = entry[1], entry[2]
            for k in range(n_samples):
                idx = np.random.choice(len(classes), size=n_obs, p=probs)
                result[:, k, 0] = classes[idx]
        else:
            model = entry[1]
            probs = model.predict_proba(self._context_array(X, S))
            classes = model.classes_
            cumprobs = np.cumsum(probs, axis=1)
            for k in range(n_samples):
                u = np.random.rand(n_obs, 1)
                idx = (u < cumprobs).argmax(axis=1)
                result[:, k, 0] = classes[idx]

        return result
