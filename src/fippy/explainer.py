"""Explainer: the main interface for computing feature importance."""
import numpy as np
import pandas as pd
import logging
from typing import Callable, Literal
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from fippy.result import ExplanationResult
from fippy.groups import FeatureGroup, as_groups

logger = logging.getLogger(__name__)


class Explainer:
    """Compute feature importance using perturbation or refit methods.

    Args:
        predict: Model prediction function f(X) -> np.ndarray.
        X_train: Training data (used for samplers and marginal distribution).
        loss: Observation-wise loss L(y, yhat) -> np.ndarray.
        sampler: Conditional sampler (needed for distribution="conditional").
            Must implement sample(X, J, S, n_samples) -> (n_obs, n_samples, len(J)).
        features: Default features/groups. None = all columns individually.
    """

    def __init__(
        self,
        predict: Callable,
        X_train: pd.DataFrame,
        loss: Callable,
        sampler=None,
        features=None,
    ):
        self.predict = predict
        self.X_train = X_train
        self.loss = loss
        self.sampler = sampler
        self._default_features = features
        self._columns = list(X_train.columns)

    def set_y_train(self, y_train):
        """Set training labels (needed for restriction='refit')."""
        self._y_train = np.asarray(y_train)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def loo(
        self,
        X: pd.DataFrame,
        y,
        restriction: Literal["resample", "marginalize", "refit"],
        *,
        distribution: Literal["marginal", "conditional"] | None = None,
        G=None,
        n_repeats: int = 1,
        n_samples: int | None = None,
        learner=None,
        features=None,
        n_jobs: int = 1,
    ) -> ExplanationResult:
        """Leave-one-out feature importance: v(D) - v(D \\ {j})."""
        self._validate_args(restriction, distribution, G, n_samples, learner)
        self._validate_X(X)
        y = np.asarray(y)
        groups = self._resolve_features(features)

        n_obs = len(X)
        scores = np.empty((1, n_repeats, n_obs, len(groups)))
        baseline_loss = self.loss(y, self.predict(X))

        for r in tqdm(range(n_repeats), desc="loo repeats", disable=n_repeats <= 1):
            if n_jobs == 1:
                for j, group in enumerate(tqdm(groups, desc="features", leave=False)):
                    scores[0, r, :, j] = self._loo_scores(
                        X, y, baseline_loss, group,
                        restriction, distribution, G, n_samples, learner,
                    )
            else:
                results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(self._loo_scores)(
                        X, y, baseline_loss, group,
                        restriction, distribution, G, n_samples, learner,
                    )
                    for group in groups
                )
                for j, res in enumerate(results):
                    scores[0, r, :, j] = res

        return ExplanationResult(
            feature_names=[g.name for g in groups],
            scores=scores,
            attribution="loo",
            restriction=restriction,
            distribution=distribution,
            baseline_loss=baseline_loss,
        )

    def shapley(
        self,
        X: pd.DataFrame,
        y,
        restriction: Literal["resample", "marginalize", "refit"],
        *,
        distribution: Literal["marginal", "conditional"] | None = None,
        n_repeats: int = 1,
        n_samples: int | None = None,
        n_permutations: int | None = None,
        convergence_threshold: float | None = 0.01,
        learner=None,
        features=None,
        n_jobs: int = 1,
    ) -> ExplanationResult:
        """Shapley-based feature importance: E_S[v(S u {j}) - v(S)]."""
        if n_jobs != 1:
            raise ValueError("n_jobs > 1 is not supported for shapley()")
        self._validate_args(restriction, distribution, None, n_samples, learner)
        self._validate_X(X)
        y = np.asarray(y)
        groups = self._resolve_features(features)

        n_obs = len(X)
        scores = np.empty((1, n_repeats, n_obs, len(groups)))
        baseline_loss = self.loss(y, self.predict(X))

        for r in tqdm(range(n_repeats), desc="shapley repeats", disable=n_repeats <= 1):
            scores[0, r, :, :] = self._shapley_scores(
                X, y, groups, restriction, distribution,
                n_samples, n_permutations, convergence_threshold, learner,
            )

        return ExplanationResult(
            feature_names=[g.name for g in groups],
            scores=scores,
            attribution="shapley",
            restriction=restriction,
            distribution=distribution,
            baseline_loss=baseline_loss,
        )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def pfi(self, X, y, *, n_repeats=1, features=None, n_jobs=1):
        """Permutation Feature Importance (loo + resample + marginal)."""
        return self.loo(X, y, "resample", distribution="marginal",
                        n_repeats=n_repeats, features=features, n_jobs=n_jobs)

    def cfi(self, X, y, *, n_repeats=1, features=None, n_jobs=1):
        """Conditional Feature Importance (loo + resample + conditional)."""
        return self.loo(X, y, "resample", distribution="conditional",
                        n_repeats=n_repeats, features=features, n_jobs=n_jobs)

    def rfi(self, X, y, G, *, n_repeats=1, features=None, n_jobs=1):
        """Relative Feature Importance (loo + resample + conditional + G)."""
        return self.loo(X, y, "resample", distribution="conditional",
                        G=G, n_repeats=n_repeats, features=features,
                        n_jobs=n_jobs)

    def loco(self, X, y, learner, *, n_repeats=1, features=None, n_jobs=1):
        """Leave-One-Covariate-Out (loo + refit)."""
        return self.loo(X, y, "refit", learner=learner,
                        n_repeats=n_repeats, features=features, n_jobs=n_jobs)

    def sage(self, X, y, *, distribution, n_samples, n_repeats=1,
             n_permutations=None, convergence_threshold=0.01, features=None):
        """SAGE (shapley + marginalize)."""
        return self.shapley(X, y, "marginalize", distribution=distribution,
                            n_repeats=n_repeats, n_samples=n_samples,
                            n_permutations=n_permutations,
                            convergence_threshold=convergence_threshold,
                            features=features)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_args(self, restriction, distribution, G, n_samples, learner):
        if restriction not in ("resample", "marginalize", "refit"):
            raise ValueError(f"Invalid restriction: '{restriction}'")

        if restriction in ("resample", "marginalize"):
            if distribution is None:
                raise ValueError(
                    f"distribution must be 'marginal' or 'conditional' "
                    f"when restriction='{restriction}'"
                )
            if distribution not in ("marginal", "conditional"):
                raise ValueError(f"Invalid distribution: '{distribution}'")
        elif restriction == "refit":
            if distribution is not None:
                raise ValueError(
                    "distribution cannot be specified for restriction='refit'"
                )

        if G is not None and distribution != "conditional":
            raise ValueError("G can only be specified when distribution='conditional'")

        if distribution == "conditional" and self.sampler is None:
            raise ValueError(
                "distribution='conditional' requires a sampler. "
                "Pass a sampler to the Explainer constructor."
            )

        if restriction == "marginalize":
            if n_samples is None:
                raise ValueError("n_samples required for restriction='marginalize'")
        elif n_samples is not None:
            raise ValueError(
                f"n_samples cannot be specified for restriction='{restriction}'"
            )

        if restriction == "refit":
            if learner is None:
                raise ValueError("learner required for restriction='refit'")
        elif learner is not None:
            raise ValueError(
                f"learner cannot be specified for restriction='{restriction}'"
            )

    def _validate_X(self, X):
        missing = set(self._columns) - set(X.columns)
        if missing:
            raise ValueError(f"X is missing columns: {missing}")

    def _resolve_features(self, features):
        feat = features if features is not None else self._default_features
        groups = as_groups(feat, all_columns=self._columns)
        all_cols = set(self._columns)
        for g in groups:
            bad = set(g.columns) - all_cols
            if bad:
                raise ValueError(f"Unknown columns in feature '{g.name}': {bad}")
        return groups

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self._columns)

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _get_sampler(self, distribution):
        if distribution == "marginal":
            return self.sampler if self.sampler is not None else _PermutationSampler(self.X_train)
        return self.sampler

    def _sample(self, X, J, S, distribution, n_samples=1):
        """Sample replacement values for features J. Returns (n_obs, n_samples, len(J))."""
        sampler = self._get_sampler(distribution)
        return sampler.sample(X, J=J, S=S, n_samples=n_samples)

    def _perturbed_predict(self, X, J, S, distribution, n_samples):
        """Predict with features J replaced. Returns (n_obs,) averaged over samples."""
        samples = self._sample(X, J, S, distribution, n_samples)
        X_np = X[self._columns].values
        col_idx = {c: i for i, c in enumerate(self._columns)}
        j_idx = [col_idx[c] for c in J]

        preds = np.zeros((len(X), n_samples))
        for k in range(n_samples):
            X_pert = X_np.copy()
            for ii, ci in enumerate(j_idx):
                X_pert[:, ci] = samples[:, k, ii]
            preds[:, k] = self.predict(
                pd.DataFrame(X_pert, columns=self._columns, index=X.index)
            )
        return preds.mean(axis=1)

    def _marginalized_loss(self, X, y, J, S, distribution, n_samples):
        """Loss with J marginalized/resampled. Returns (n_obs,)."""
        avg_preds = self._perturbed_predict(X, J, S, distribution, n_samples)
        return self.loss(y, avg_preds)

    def _loo_scores(self, X, y, baseline_loss, group, restriction,
                    distribution, G, n_samples, learner):
        """Observation-wise LOO scores for one feature/group. Returns (n_obs,)."""
        drop = group.columns
        if restriction == "refit":
            return self._refit_scores(X, y, baseline_loss, drop, learner)

        keep = [c for c in self._columns if c not in drop]
        if distribution == "marginal":
            S = []
        else:
            S = list(G) if G is not None else keep

        ns = 1 if restriction == "resample" else n_samples
        restricted_preds = self._perturbed_predict(X, drop, S, distribution, ns)
        return self.loss(y, restricted_preds) - baseline_loss

    def _refit_scores(self, X, y, baseline_loss, drop_cols, learner):
        from sklearn.base import clone
        keep = [c for c in self._columns if c not in drop_cols]
        reduced = clone(learner)
        reduced.fit(self.X_train[keep], self._y_train)
        preds = reduced.predict(X[keep])
        return self.loss(y, preds) - baseline_loss

    def _shapley_scores(self, X, y, groups, restriction, distribution,
                        n_samples, n_permutations, convergence_threshold, learner):
        """Shapley values via random permutation sampling. Returns (n_obs, n_features)."""
        n_obs = len(X)
        n_feat = len(groups)

        if n_permutations is None:
            n_permutations = 500

        running_mean = np.zeros((n_obs, n_feat))
        running_m2 = np.zeros((n_obs, n_feat))

        pbar = tqdm(range(n_permutations), desc="permutations")
        for t in pbar:
            perm = np.random.permutation(n_feat)
            mc = np.zeros((n_obs, n_feat))

            coalition_cols = []
            prev_loss = self._value_fn(X, y, coalition_cols, restriction,
                                       distribution, n_samples, learner)

            for pos in range(n_feat):
                idx = perm[pos]
                coalition_cols = coalition_cols + groups[idx].columns
                curr_loss = self._value_fn(X, y, coalition_cols, restriction,
                                           distribution, n_samples, learner)
                # Negate: importance = loss without - loss with (positive = helpful)
                mc[:, idx] = prev_loss - curr_loss
                prev_loss = curr_loss

            # Welford's online algorithm
            n = t + 1
            delta = mc - running_mean
            running_mean += delta / n
            running_m2 += delta * (mc - running_mean)

            # Convergence check
            if convergence_threshold is not None and n >= 10:
                var = running_m2 / (n - 1)
                se = np.sqrt(var / n).mean(axis=0)
                abs_mean = np.abs(running_mean.mean(axis=0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_se = np.where(abs_mean > 1e-10, se / abs_mean, 0.0)
                if np.all(rel_se < convergence_threshold):
                    logger.info(f"Shapley converged after {n} permutations")
                    pbar.close()
                    break

        return running_mean

    def _value_fn(self, X, y, coalition_cols, restriction, distribution,
                  n_samples, learner):
        """Value function v(S). Returns observation-wise loss (n_obs,)."""
        drop = [c for c in self._columns if c not in coalition_cols]
        if not drop:
            return self.loss(y, self.predict(X))

        if restriction == "refit":
            if not coalition_cols:
                return self.loss(y, np.full(len(X), y.mean()))
            from sklearn.base import clone
            reduced = clone(learner)
            reduced.fit(self.X_train[coalition_cols], self._y_train)
            return self.loss(y, reduced.predict(X[coalition_cols]))

        ns = n_samples if n_samples is not None else 1
        if distribution == "marginal":
            S = []
        else:
            S = coalition_cols
        return self._marginalized_loss(X, y, drop, S, distribution, ns)


class _PermutationSampler:
    """Built-in marginal sampler: draws from training data ignoring S."""

    def __init__(self, X_train):
        self._values = {c: X_train[c].values for c in X_train.columns}
        self._n = len(X_train)

    def sample(self, X, J, S, n_samples=1):
        n_obs = len(X)
        result = np.empty((n_obs, n_samples, len(J)))
        for k in range(n_samples):
            for i, col in enumerate(J):
                idx = np.random.randint(0, self._n, size=n_obs)
                result[:, k, i] = self._values[col][idx]
        return result
