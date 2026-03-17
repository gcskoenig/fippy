"""PermutationSampler: samples from marginal distribution by permuting training data."""
import numpy as np
import pandas as pd

from fippy.samplers.base import Sampler


class PermutationSampler(Sampler):
    """Samples from the marginal distribution P(X_J) by randomly
    drawing rows from training data.

    This is the default sampler for distribution="marginal".
    """

    multivariate = True
    supports_categorical_target = True
    supports_categorical_context = True

    def _fit(self, J, S):
        """No fitting needed for permutation sampling."""
        pass

    def _sample(self, X, J, S, n_samples=1):
        """Sample by drawing from training data (ignores S).

        Args:
            X: Data to generate samples for (determines n_obs).
            J: Feature names to sample.
            S: Conditioning set (ignored — marginal sampling).
            n_samples: Number of samples per observation.

        Returns:
            np.ndarray of shape (n_obs, n_samples, len(J)).
        """
        n_obs = len(X)
        n_j = len(J)
        result = np.empty((n_obs, n_samples, n_j))

        train_values = self.X_train[J].values
        n_train = len(train_values)

        for k in range(n_samples):
            indices = np.random.randint(0, n_train, size=n_obs)
            result[:, k, :] = train_values[indices]

        return result
