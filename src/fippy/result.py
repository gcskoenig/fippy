"""ExplanationResult: stores feature importance scores and provides inference."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ExplanationResult:
    """Stores feature importance scores from Explainer.

    Attributes:
        feature_names: Names of features or feature groups.
        scores: 4D array of shape (n_folds, n_repeats, n_obs, n_features).
        attribution: "loo" or "shapley".
        restriction: "resample", "marginalize", or "refit".
        distribution: "marginal", "conditional", or None (for refit).
        description: Human-readable description of the method.
    """
    feature_names: list[str]
    scores: np.ndarray  # (n_folds, n_repeats, n_obs, n_features)
    attribution: str
    restriction: str
    distribution: str | None = None
    description: str = ""

    @property
    def method(self) -> str:
        """Short string describing the method."""
        parts = [self.attribution, self.restriction]
        if self.distribution is not None:
            parts.append(self.distribution)
        return "_".join(parts)

    def importance(self, aggregate_over=None) -> pd.DataFrame:
        """Mean importance per feature.

        Args:
            aggregate_over: Which axis provides the std.
                "repeats" -- std over repeats
                "folds" -- std over folds (requires n_folds > 1)
                None (default) -- auto-selects "repeats" when n_folds == 1.
                    Raises ValueError when n_folds > 1.
        """
        if aggregate_over is None:
            if self.scores.shape[0] > 1:
                raise ValueError(
                    "n_folds > 1: you must choose aggregate_over='repeats' or "
                    "'folds' explicitly."
                )
            aggregate_over = "repeats"

        # Mean over observations first
        per_repeat = np.nanmean(self.scores, axis=2)  # (n_folds, n_repeats, n_features)

        if aggregate_over == "repeats":
            means = per_repeat.mean(axis=(0, 1))
            if self.scores.shape[0] > 1:
                # Average over folds first, then std over repeats
                stds = per_repeat.mean(axis=0).std(axis=0)
            else:
                stds = per_repeat[0].std(axis=0)
        elif aggregate_over == "folds":
            if self.scores.shape[0] <= 1:
                raise ValueError("aggregate_over='folds' requires n_folds > 1")
            per_fold = per_repeat.mean(axis=1)  # (n_folds, n_features)
            means = per_fold.mean(axis=0)
            stds = per_fold.std(axis=0)
        else:
            raise ValueError(
                f"aggregate_over must be 'repeats' or 'folds', got '{aggregate_over}'"
            )

        return pd.DataFrame({
            "importance": means,
            "std": stds,
        }, index=pd.Index(self.feature_names, name="feature"))

    def obs_importance(self) -> np.ndarray:
        """Observation-wise importance, shape (n_obs, n_features).

        Averages over repeats. Only supported for n_folds == 1.
        """
        if self.scores.shape[0] > 1:
            raise NotImplementedError(
                "obs_importance() is not yet supported for multi-fold results."
            )
        return np.nanmean(self.scores[0], axis=0)  # (n_obs, n_features)

    def ci(self, alpha=0.05, method="t", aggregate_over=None) -> pd.DataFrame:
        """Confidence intervals.

        Args:
            method: "t" or "quantile".
            aggregate_over: "repeats", "folds", or None (auto).
            alpha: Significance level.

        Returns:
            DataFrame with [importance, lower, upper] indexed by feature.
        """
        if aggregate_over is None:
            if self.scores.shape[0] > 1:
                raise ValueError(
                    "n_folds > 1: you must choose aggregate_over='repeats' or "
                    "'folds' explicitly."
                )
            aggregate_over = "repeats"

        # Mean over observations
        per_repeat = np.nanmean(self.scores, axis=2)  # (n_folds, n_repeats, n_features)

        if aggregate_over == "repeats":
            if self.scores.shape[0] > 1:
                values = per_repeat.mean(axis=0)  # avg over folds -> (n_repeats, n_features)
            else:
                values = per_repeat[0]  # (n_repeats, n_features)
            n = values.shape[0]
        elif aggregate_over == "folds":
            if self.scores.shape[0] <= 1:
                raise ValueError("aggregate_over='folds' requires n_folds > 1")
            values = per_repeat.mean(axis=1)  # avg over repeats -> (n_folds, n_features)
            n = values.shape[0]
        else:
            raise ValueError(
                f"aggregate_over must be 'repeats' or 'folds', got '{aggregate_over}'"
            )

        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=1)

        if method == "t":
            from scipy.stats import t as t_dist
            t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
            se = stds / np.sqrt(n)
            lower = means - t_crit * se
            upper = means + t_crit * se
        elif method == "quantile":
            lower = np.quantile(values, alpha / 2, axis=0)
            upper = np.quantile(values, 1 - alpha / 2, axis=0)
        else:
            raise ValueError(f"Unknown CI method: {method}")

        return pd.DataFrame({
            "importance": means,
            "lower": lower,
            "upper": upper,
        }, index=pd.Index(self.feature_names, name="feature"))

    def test(self, method="t", alternative="greater", p_adjust=None) -> pd.DataFrame:
        """Hypothesis test H0: importance_j <= 0.

        Tests on observation-wise scores averaged over repeats.
        Only supported for n_folds == 1.

        Args:
            method: "t" or "wilcoxon".
            alternative: "greater", "less", or "two-sided".
            p_adjust: None, "bonferroni", "holm", or "bh".

        Returns:
            DataFrame with [importance, statistic, p_value] indexed by feature.
        """
        if self.scores.shape[0] > 1:
            raise NotImplementedError(
                "test() is not yet supported for multi-fold results."
            )

        # Average over repeats -> (n_obs, n_features)
        obs_scores = np.nanmean(self.scores[0], axis=0)
        n_features = obs_scores.shape[1]

        results = []
        for j in range(n_features):
            vals = obs_scores[:, j]
            vals = vals[~np.isnan(vals)]

            if np.std(vals) == 0:
                raise ValueError(
                    f"Cannot run test for feature '{self.feature_names[j]}': "
                    f"observation-wise scores have zero variance."
                )

            if method == "t":
                from scipy.stats import ttest_1samp
                stat, p = ttest_1samp(vals, 0, alternative=alternative)
            elif method == "wilcoxon":
                from scipy.stats import wilcoxon
                stat, p = wilcoxon(vals, alternative=alternative)
            else:
                raise ValueError(f"Unknown test method: {method}")

            results.append({
                "feature": self.feature_names[j],
                "importance": vals.mean(),
                "statistic": stat,
                "p_value": p,
            })

        df = pd.DataFrame(results).set_index("feature")

        if p_adjust is not None:
            df["p_value"] = _adjust_pvalues(df["p_value"].values, p_adjust)

        return df

    def to_csv(self, path):
        """Save result to CSV."""
        # Flatten scores and save with metadata
        n_folds, n_repeats, n_obs, n_features = self.scores.shape
        flat = self.scores.reshape(-1, n_features)
        df = pd.DataFrame(flat, columns=self.feature_names)
        df.attrs["n_folds"] = n_folds
        df.attrs["n_repeats"] = n_repeats
        df.attrs["n_obs"] = n_obs

        # Write metadata as comment lines, then data
        import json
        meta = {
            "feature_names": self.feature_names,
            "shape": list(self.scores.shape),
            "attribution": self.attribution,
            "restriction": self.restriction,
            "distribution": self.distribution,
            "description": self.description,
        }
        with open(path, "w") as f:
            f.write(f"# {json.dumps(meta)}\n")
            df.to_csv(f, index=False)

    @classmethod
    def from_csv(cls, path):
        """Load result from CSV."""
        import json
        with open(path, "r") as f:
            meta_line = f.readline().strip()
            meta = json.loads(meta_line[2:])  # strip "# "
            df = pd.read_csv(f)

        scores = df.values.reshape(meta["shape"])
        return cls(
            feature_names=meta["feature_names"],
            scores=scores,
            attribution=meta["attribution"],
            restriction=meta["restriction"],
            distribution=meta.get("distribution"),
            description=meta.get("description", ""),
        )


def _adjust_pvalues(pvalues, method):
    """Multiple testing correction."""
    n = len(pvalues)
    if method == "bonferroni":
        return np.minimum(pvalues * n, 1.0)
    elif method == "holm":
        order = np.argsort(pvalues)
        adjusted = np.empty(n)
        for i, idx in enumerate(order):
            adjusted[idx] = min(pvalues[idx] * (n - i), 1.0)
        # Enforce monotonicity
        cummax = 0.0
        for i in range(n):
            idx = order[i]
            adjusted[idx] = max(adjusted[idx], cummax)
            cummax = adjusted[idx]
        return adjusted
    elif method == "bh":
        order = np.argsort(pvalues)[::-1]
        adjusted = np.empty(n)
        cummin = 1.0
        for i, idx in enumerate(order):
            rank = n - i
            adjusted[idx] = min(pvalues[idx] * n / rank, cummin)
            cummin = adjusted[idx]
        return np.minimum(adjusted, 1.0)
    else:
        raise ValueError(f"Unknown p_adjust method: {method}")
