"""GaussianSampler: conditional sampling via Gaussian conditioning formulas."""
import numpy as np
import pandas as pd
from fippy.backend.estimators import GaussianConditionalEstimator


class GaussianSampler:
    """Second-order Gaussian conditional sampler.

    Computes P(X_J | X_S) using standard multivariate normal conditioning:
      mu_{J|S} = mu_J + Sigma_JS @ Sigma_SS^{-1} @ (x_S - mu_S)
      Sigma_{J|S} = Sigma_JJ - Sigma_JS @ Sigma_SS^{-1} @ Sigma_SJ

    Estimators are fitted lazily and cached by (J, S) key.
    """

    multivariate = True

    def __init__(self, X_train: pd.DataFrame):
        self.X_train = X_train
        self.feature_names = list(X_train.columns)
        self._cache: dict[tuple, object] = {}

    def fit(self, J, S):
        """Fit P(X_J | X_S)."""
        key = self._key(J, S)
        if key in self._cache:
            return

        J, S = list(J), list(S)
        J_only = sorted(set(J) - set(S))

        if not J_only:
            self._cache[key] = "identity"
            return

        estimator = GaussianConditionalEstimator()
        estimator.fit(
            train_inputs=self.X_train[J_only].to_numpy(),
            train_context=self.X_train[sorted(S)].to_numpy() if S else np.zeros((len(self.X_train), 0)),
        )
        self._cache[key] = (estimator, J_only)

    def sample(self, X, J, S, n_samples=1):
        """Sample from P(X_J | X_S).

        Returns: np.ndarray of shape (n_obs, n_samples, len(J)).
        """
        J, S = list(J), list(S)
        key = self._key(J, S)
        if key not in self._cache:
            self.fit(J, S)

        entry = self._cache[key]
        n_obs = len(X)

        if entry == "identity":
            vals = X[J].values if isinstance(X, pd.DataFrame) else X
            return np.broadcast_to(vals[:, np.newaxis, :], (n_obs, n_samples, len(J))).copy()

        estimator, J_only = entry
        context = X[sorted(S)].values if S else np.zeros((n_obs, 0))
        # estimator.sample returns (n_obs, n_samples, len(J_only))
        raw = estimator.sample(context, num_samples=n_samples)

        J_in_S = [j for j in J if j in S]
        if not J_in_S:
            # J_only == J (possibly reordered)
            if sorted(J) == J:
                return raw
            # Reorder to match J
            result = np.empty((n_obs, n_samples, len(J)))
            for i, j in enumerate(J):
                src = J_only.index(j)
                result[:, :, i] = raw[:, :, src]
            return result

        # Mix identity (J in S) and sampled (J_only)
        result = np.empty((n_obs, n_samples, len(J)))
        for i, j in enumerate(J):
            if j in J_in_S:
                col_vals = X[j].values if isinstance(X, pd.DataFrame) else X[:, self.feature_names.index(j)]
                result[:, :, i] = col_vals[:, np.newaxis]
            else:
                src = J_only.index(j)
                result[:, :, i] = raw[:, :, src]
        return result

    @staticmethod
    def _key(J, S):
        return (tuple(sorted(J)), tuple(sorted(S)))
