"""Tests for the refactored Explainer API.

Uses controlled synthetic data where ground-truth feature importance
is known analytically, so we can verify computed values make sense.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from fippy import Explainer, ExplanationResult
from fippy.samplers import GaussianSampler, PermutationSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def squared_error(y_true, y_pred):
    """Observation-wise squared error."""
    return (y_true - y_pred) ** 2


def make_independent_linear(n=2000, seed=42):
    """y = 3*x1 + 0*x2 + noise.  x1, x2 independent standard normal."""
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = 3 * x1 + rng.randn(n) * 0.1
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y, name="y")


def make_correlated_linear(n=2000, rho=0.9, seed=42):
    """y = 3*x1 + 0*x2 + noise.  x1, x2 correlated (correlation=rho)."""
    rng = np.random.RandomState(seed)
    z1 = rng.randn(n)
    z2 = rng.randn(n)
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    y = 3 * x1 + rng.randn(n) * 0.1
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y, name="y")


def make_multifeature(n=2000, seed=42):
    """y = 2*x1 + 1*x2 + 0*x3 + 0*x4 + noise.  All independent."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "x3": rng.randn(n),
        "x4": rng.randn(n),
    })
    y = 2 * X["x1"] + 1 * X["x2"] + rng.randn(n) * 0.1
    return X, pd.Series(y, name="y")


def make_grouped_features(n=2000, seed=42):
    """y = 2*(x1a + x1b) + 0*x2a + 0*x2b + noise."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "x1a": rng.randn(n),
        "x1b": rng.randn(n),
        "x2a": rng.randn(n),
        "x2b": rng.randn(n),
    })
    y = 2 * (X["x1a"] + X["x1b"]) + rng.randn(n) * 0.1
    return X, pd.Series(y, name="y")


def make_interaction(n=2000, seed=42):
    """y = 5 * x1 * x2 + noise.  No main effects."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "x3": rng.randn(n),
    })
    y = 5 * X["x1"] * X["x2"] + rng.randn(n) * 0.1
    return X, pd.Series(y, name="y")


def fit_linear(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict


def fit_rf(X, y, seed=42):
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(X, y)
    return model.predict


# ---------------------------------------------------------------------------
# ExplanationResult Tests
# ---------------------------------------------------------------------------

class TestExplanationResult:

    def test_scores_shape(self):
        """scores has shape (1, n_repeats, n_obs, n_features)."""
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=3)

        assert result.scores.ndim == 4
        assert result.scores.shape[0] == 1       # n_folds
        assert result.scores.shape[1] == 3       # n_repeats
        assert result.scores.shape[2] == len(X)  # n_obs
        assert result.scores.shape[3] == 2       # n_features

    def test_importance_returns_dataframe(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=3)

        imp = result.importance()
        assert isinstance(imp, pd.DataFrame)
        assert "importance" in imp.columns
        assert "std" in imp.columns
        assert list(imp.index) == result.feature_names

    def test_obs_importance_shape(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=5)

        obs_imp = result.obs_importance()
        assert obs_imp.shape == (len(X), 2)

    def test_feature_names_match(self):
        X, y = make_multifeature()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, y)

        assert set(result.feature_names) == set(X.columns)

    def test_serialization_roundtrip(self, tmp_path):
        X, y = make_independent_linear(n=100)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=2)

        path = str(tmp_path / "result.csv")
        result.to_csv(path)
        loaded = ExplanationResult.from_csv(path)

        assert loaded.feature_names == result.feature_names
        assert loaded.method == result.method
        np.testing.assert_allclose(loaded.scores, result.scores)


# ---------------------------------------------------------------------------
# PFI Tests
# ---------------------------------------------------------------------------

class TestPFI:

    def test_irrelevant_feature_has_zero_importance(self):
        X, y = make_independent_linear(n=5000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["x2", "importance"]) < 0.5

    def test_relevant_feature_has_positive_importance(self):
        X, y = make_independent_linear(n=5000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()
        assert imp.loc["x1", "importance"] > 1.0

    def test_ordering_matches_true_importance(self):
        X, y = make_multifeature(n=5000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()

        assert imp.loc["x1", "importance"] > imp.loc["x2", "importance"]
        assert imp.loc["x2", "importance"] > imp.loc["x3", "importance"] + 0.1
        assert abs(imp.loc["x3", "importance"]) < 0.5
        assert abs(imp.loc["x4", "importance"]) < 0.5

    def test_pfi_detects_interaction_features(self):
        X, y = make_interaction(n=5000)
        predict = fit_rf(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()

        assert imp.loc["x1", "importance"] > 1.0
        assert imp.loc["x2", "importance"] > 1.0
        assert abs(imp.loc["x3", "importance"]) < 1.0

    def test_pfi_correlated_features_false_attribution(self):
        """PFI with a tree model attributes some importance to correlated x2."""
        X, y = make_correlated_linear(n=5000, rho=0.9)
        predict = fit_rf(X, y)  # RF splits on x2 due to correlation
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()
        # x2 gets nonzero PFI since permuting it breaks correlation structure
        assert imp.loc["x2", "importance"] > 0.001

    def test_independent_features_pfi_equals_cfi(self):
        X, y = make_independent_linear(n=5000)
        predict = fit_linear(X, y)
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        pfi_result = explainer.pfi(X, y, n_repeats=10)
        cfi_result = explainer.cfi(X, y, n_repeats=10)

        pfi_imp = pfi_result.importance()
        cfi_imp = cfi_result.importance()

        for feat in ["x1", "x2"]:
            np.testing.assert_allclose(
                pfi_imp.loc[feat, "importance"],
                cfi_imp.loc[feat, "importance"],
                atol=1.0,
            )

    def test_n_repeats_reduces_standard_error(self):
        """More repeats should reduce the standard error of the mean."""
        X, y = make_independent_linear(n=3000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        n_few, n_many = 5, 100
        result_few = explainer.pfi(X, y, n_repeats=n_few)
        result_many = explainer.pfi(X, y, n_repeats=n_many)

        # SE = std / sqrt(n_repeats)
        se_few = result_few.importance().loc["x1", "std"] / np.sqrt(n_few)
        se_many = result_many.importance().loc["x1", "std"] / np.sqrt(n_many)
        assert se_many < se_few

    def test_pfi_subset_of_features(self):
        X, y = make_multifeature(n=2000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, features=["x1", "x3"])
        assert set(result.feature_names) == {"x1", "x3"}
        assert result.scores.shape[3] == 2


# ---------------------------------------------------------------------------
# CFI Tests
# ---------------------------------------------------------------------------

class TestCFI:

    def test_cfi_irrelevant_feature_near_zero(self):
        X, y = make_correlated_linear(n=5000, rho=0.9)
        predict = fit_linear(X, y)
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        result = explainer.cfi(X, y, n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["x2", "importance"]) < 1.0

    def test_cfi_relevant_feature_positive(self):
        X, y = make_correlated_linear(n=5000, rho=0.9)
        predict = fit_linear(X, y)
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        result = explainer.cfi(X, y, n_repeats=10)
        imp = result.importance()
        assert imp.loc["x1", "importance"] > 0.5

    def test_cfi_resolves_pfi_false_attribution(self):
        X, y = make_correlated_linear(n=5000, rho=0.9)
        predict = fit_rf(X, y)  # RF splits on x2 due to correlation
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        pfi_result = explainer.pfi(X, y, n_repeats=10)
        cfi_result = explainer.cfi(X, y, n_repeats=10)
        assert cfi_result.importance().loc["x2", "importance"] < pfi_result.importance().loc["x2", "importance"]

    def test_cfi_requires_sampler(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)  # no sampler

        with pytest.raises(ValueError, match="requires a sampler"):
            explainer.cfi(X, y)

    def test_cfi_more_repeats_reduces_standard_error(self):
        X, y = make_independent_linear(n=2000)
        predict = fit_linear(X, y)
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        n_few, n_many = 5, 50
        result_few = explainer.cfi(X, y, n_repeats=n_few)
        result_many = explainer.cfi(X, y, n_repeats=n_many)

        se_few = result_few.importance().loc["x1", "std"] / np.sqrt(n_few)
        se_many = result_many.importance().loc["x1", "std"] / np.sqrt(n_many)
        assert se_many < se_few


# ---------------------------------------------------------------------------
# LOCO Tests
# ---------------------------------------------------------------------------

class TestLOCO:

    def _make_explainer_with_y_train(self, X, y):
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)
        explainer.set_y_train(y)
        return explainer

    def test_loco_irrelevant_feature_near_zero(self):
        X, y = make_independent_linear(n=2000)
        explainer = self._make_explainer_with_y_train(X, y)

        result = explainer.loco(X, y, learner=LinearRegression(), n_repeats=1)
        imp = result.importance()
        assert abs(imp.loc["x2", "importance"]) < 0.5

    def test_loco_relevant_feature_positive(self):
        X, y = make_independent_linear(n=2000)
        explainer = self._make_explainer_with_y_train(X, y)

        result = explainer.loco(X, y, learner=LinearRegression(), n_repeats=1)
        imp = result.importance()
        assert imp.loc["x1", "importance"] > 1.0

    def test_loco_ordering(self):
        X, y = make_multifeature(n=2000)
        explainer = self._make_explainer_with_y_train(X, y)

        result = explainer.loco(X, y, learner=LinearRegression(), n_repeats=1)
        imp = result.importance()

        assert imp.loc["x1", "importance"] > imp.loc["x2", "importance"]
        assert imp.loc["x2", "importance"] > imp.loc["x3", "importance"] + 0.1

    def test_loco_correlated_features(self):
        X, y = make_correlated_linear(n=2000, rho=0.9)
        explainer = self._make_explainer_with_y_train(X, y)

        result = explainer.loco(X, y, learner=LinearRegression(), n_repeats=1)
        imp = result.importance()
        assert imp.loc["x1", "importance"] > imp.loc["x2", "importance"]


# ---------------------------------------------------------------------------
# Feature Group Tests
# ---------------------------------------------------------------------------

class TestFeatureGroups:

    def test_pfi_with_groups(self):
        X, y = make_grouped_features(n=5000)
        predict = fit_linear(X, y)
        groups = {"group1": ["x1a", "x1b"], "group2": ["x2a", "x2b"]}
        explainer = Explainer(predict, X, loss=squared_error, features=groups)

        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()

        assert set(result.feature_names) == {"group1", "group2"}
        assert imp.loc["group1", "importance"] > 1.0
        assert abs(imp.loc["group2", "importance"]) < 0.5

    def test_cfi_with_groups(self):
        X, y = make_grouped_features(n=5000)
        predict = fit_linear(X, y)
        sampler = GaussianSampler(X)
        groups = {"group1": ["x1a", "x1b"], "group2": ["x2a", "x2b"]}
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler, features=groups)

        result = explainer.cfi(X, y, n_repeats=5)
        imp = result.importance()

        assert imp.loc["group1", "importance"] > 1.0
        assert abs(imp.loc["group2", "importance"]) < 0.5

    def test_loco_with_groups(self):
        X, y = make_grouped_features(n=2000)
        predict = fit_linear(X, y)
        groups = {"group1": ["x1a", "x1b"], "group2": ["x2a", "x2b"]}
        explainer = Explainer(predict, X, loss=squared_error, features=groups)
        explainer.set_y_train(y)

        result = explainer.loco(X, y, learner=LinearRegression(), n_repeats=1)
        imp = result.importance()

        assert set(result.feature_names) == {"group1", "group2"}
        assert imp.loc["group1", "importance"] > imp.loc["group2", "importance"]

    def test_sage_with_groups(self):
        X, y = make_grouped_features(n=2000)
        predict = fit_linear(X, y)
        groups = {"group1": ["x1a", "x1b"], "group2": ["x2a", "x2b"]}
        explainer = Explainer(predict, X, loss=squared_error, features=groups)

        result = explainer.sage(X, y, distribution="marginal",
                                n_samples=20, n_permutations=20, n_repeats=1)
        imp = result.importance()

        assert set(result.feature_names) == {"group1", "group2"}
        assert imp.loc["group1", "importance"] > imp.loc["group2", "importance"]

    def test_groups_override_at_call_site(self):
        X, y = make_multifeature(n=2000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        groups = {"ab": ["x1", "x2"], "cd": ["x3", "x4"]}
        result = explainer.pfi(X, y, features=groups, n_repeats=5)

        assert set(result.feature_names) == {"ab", "cd"}
        assert result.scores.shape[3] == 2


# ---------------------------------------------------------------------------
# SAGE Tests
# ---------------------------------------------------------------------------

class TestSAGE:

    def test_sage_values_sum_to_total(self):
        X, y = make_independent_linear(n=2000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.sage(X, y, distribution="marginal",
                                n_permutations=50, n_samples=50, n_repeats=1)
        imp = result.importance()
        sage_sum = imp["importance"].sum()

        y_arr = y.values
        loss_empty = explainer._value_fn(X, y_arr, [], "marginalize", "marginal", 50, None)
        loss_full = explainer._value_fn(X, y_arr, list(X.columns), "marginalize", "marginal", 50, None)
        # SAGE sums to total loss reduction: loss(empty) - loss(full)
        total_reduction = loss_empty.mean() - loss_full.mean()

        np.testing.assert_allclose(sage_sum, total_reduction, atol=1.0)

    def test_sage_ordering_correct(self):
        X, y = make_multifeature(n=3000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.sage(X, y, distribution="marginal",
                                n_permutations=100, n_samples=20, n_repeats=1)
        imp = result.importance()

        assert abs(imp.loc["x1", "importance"]) > abs(imp.loc["x2", "importance"])
        assert abs(imp.loc["x3", "importance"]) < 0.5
        assert abs(imp.loc["x4", "importance"]) < 0.5

    def test_sage_marginal_vs_conditional_independent(self):
        X, y = make_independent_linear(n=3000)
        predict = fit_linear(X, y)
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        result_m = explainer.sage(X, y, distribution="marginal",
                                  n_permutations=50, n_samples=20, n_repeats=1)
        result_c = explainer.sage(X, y, distribution="conditional",
                                  n_permutations=50, n_samples=20, n_repeats=1)

        for feat in ["x1", "x2"]:
            np.testing.assert_allclose(
                result_m.importance().loc[feat, "importance"],
                result_c.importance().loc[feat, "importance"],
                atol=1.5,
            )

    def test_sage_conditional_requires_sampler(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="requires a sampler"):
            explainer.sage(X, y, distribution="conditional",
                           n_samples=10, n_permutations=10)

    def test_sage_convergence(self):
        X, y = make_independent_linear(n=1000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.sage(X, y, distribution="marginal",
                                n_permutations=500, n_samples=10, n_repeats=1,
                                convergence_threshold=0.05)
        imp = result.importance()
        assert imp.loc["x1", "importance"] != 0  # converged to something

    def test_sage_symmetry(self):
        rng = np.random.RandomState(42)
        n = 5000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = 3 * X["x1"] + 3 * X["x2"] + rng.randn(n) * 0.1

        predict = fit_linear(X, pd.Series(y))
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.sage(X, pd.Series(y), distribution="marginal",
                                n_permutations=100, n_samples=20, n_repeats=1)
        imp = result.importance()
        np.testing.assert_allclose(
            imp.loc["x1", "importance"],
            imp.loc["x2", "importance"],
            atol=1.0,
        )


# ---------------------------------------------------------------------------
# Sampler Tests
# ---------------------------------------------------------------------------

class TestPermutationSampler:

    def test_output_shape(self):
        X, _ = make_independent_linear(n=200)
        sampler = PermutationSampler(X)

        result = sampler.sample(X, J=["x1"], S=["x2"], n_samples=5)
        assert result.shape == (200, 5, 1)

    def test_marginal_ignores_conditioning(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(1000), "x2": rng.randn(1000)})
        sampler = PermutationSampler(X)

        X_low = pd.DataFrame({"x1": np.zeros(100), "x2": np.full(100, -10.0)})
        X_high = pd.DataFrame({"x1": np.zeros(100), "x2": np.full(100, 10.0)})

        np.random.seed(0)
        samples_low = sampler.sample(X_low, J=["x1"], S=["x2"], n_samples=100)
        np.random.seed(0)
        samples_high = sampler.sample(X_high, J=["x1"], S=["x2"], n_samples=100)

        np.testing.assert_array_equal(samples_low, samples_high)

    def test_marginal_distribution_matches_training(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(10000) * 3 + 5, "x2": rng.randn(10000)})
        sampler = PermutationSampler(X)

        samples = sampler.sample(X.head(1), J=["x1"], S=["x2"], n_samples=10000)
        sampled_x1 = samples[0, :, 0]

        np.testing.assert_allclose(sampled_x1.mean(), 5.0, atol=0.2)
        np.testing.assert_allclose(sampled_x1.std(), 3.0, atol=0.2)


class TestGaussianSampler:

    def test_output_shape(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(500), "x2": rng.randn(500)})
        sampler = GaussianSampler(X)

        result = sampler.sample(X, J=["x1"], S=["x2"], n_samples=5)
        assert result.shape == (500, 5, 1)

    def test_conditional_mean_correct(self):
        rng = np.random.RandomState(42)
        n = 50000
        rho = 0.8
        z1 = rng.randn(n)
        z2 = rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho**2) * z2

        X = pd.DataFrame({"x1": x1, "x2": x2})
        sampler = GaussianSampler(X)

        X_cond = pd.DataFrame({"x1": [0.0], "x2": [2.0]})
        samples = sampler.sample(X_cond, J=["x1"], S=["x2"], n_samples=10000)
        sampled_mean = samples[0, :, 0].mean()

        expected_mean = rho * 2.0
        np.testing.assert_allclose(sampled_mean, expected_mean, atol=0.1)

    def test_conditional_variance_correct(self):
        rng = np.random.RandomState(42)
        n = 50000
        rho = 0.8
        z1 = rng.randn(n)
        z2 = rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho**2) * z2

        X = pd.DataFrame({"x1": x1, "x2": x2})
        sampler = GaussianSampler(X)

        X_cond = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
        samples = sampler.sample(X_cond, J=["x1"], S=["x2"], n_samples=10000)
        sampled_var = samples[0, :, 0].var()

        expected_var = 1 - rho**2
        np.testing.assert_allclose(sampled_var, expected_var, atol=0.05)

    def test_multivariate_conditional(self):
        rng = np.random.RandomState(42)
        n = 10000
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": rng.randn(n),
        })
        X["x1"] = X["x1"] + 0.5 * X["x3"]
        X["x2"] = X["x2"] + 0.3 * X["x3"]

        sampler = GaussianSampler(X)
        result = sampler.sample(X, J=["x1", "x2"], S=["x3"], n_samples=3)
        assert result.shape == (n, 3, 2)

    def test_empty_conditioning_set(self):
        rng = np.random.RandomState(42)
        n = 10000
        X = pd.DataFrame({"x1": rng.randn(n) * 2 + 1, "x2": rng.randn(n)})
        sampler = GaussianSampler(X)

        X_dummy = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
        samples = sampler.sample(X_dummy, J=["x1"], S=[], n_samples=10000)
        sampled_x1 = samples[0, :, 0]

        np.testing.assert_allclose(sampled_x1.mean(), 1.0, atol=0.1)
        np.testing.assert_allclose(sampled_x1.std(), 2.0, atol=0.1)


# ---------------------------------------------------------------------------
# Inference Tests
# ---------------------------------------------------------------------------

class TestInference:

    def test_ci_contains_point_estimate(self):
        X, y = make_independent_linear(n=2000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=20)
        ci = result.ci(alpha=0.05, method="t")
        imp = result.importance()

        for feat in result.feature_names:
            point = imp.loc[feat, "importance"]
            assert ci.loc[feat, "lower"] <= point <= ci.loc[feat, "upper"]

    def test_ci_irrelevant_feature_near_zero(self):
        X, y = make_independent_linear(n=3000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=30)
        ci = result.ci(alpha=0.05, method="t")
        # CI for irrelevant feature should be near zero
        assert abs(ci.loc["x2", "importance"]) < 0.5
        assert ci.loc["x2", "upper"] - ci.loc["x2", "lower"] < 2.0

    def test_ci_relevant_feature_excludes_zero(self):
        X, y = make_independent_linear(n=3000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=30)
        ci = result.ci(alpha=0.05, method="t")
        assert ci.loc["x1", "lower"] > 0

    def test_ci_wider_at_lower_confidence(self):
        X, y = make_independent_linear(n=2000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=20)
        ci_95 = result.ci(alpha=0.05, method="t")
        ci_99 = result.ci(alpha=0.01, method="t")

        width_95 = ci_95.loc["x1", "upper"] - ci_95.loc["x1", "lower"]
        width_99 = ci_99.loc["x1", "upper"] - ci_99.loc["x1", "lower"]
        assert width_99 > width_95

    def test_hypothesis_test_rejects_relevant(self):
        X, y = make_independent_linear(n=3000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=30)
        test_result = result.test(method="t", alternative="greater")
        assert test_result.loc["x1", "p_value"] < 0.01

    def test_hypothesis_test_does_not_reject_irrelevant(self):
        X, y = make_independent_linear(n=3000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=30)
        test_result = result.test(method="t", alternative="greater")
        assert test_result.loc["x2", "p_value"] > 0.05

    def test_multiple_test_methods_agree_on_direction(self):
        X, y = make_independent_linear(n=3000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=30)
        for method in ["t", "wilcoxon"]:
            test_result = result.test(method=method, alternative="greater")
            assert test_result.loc["x1", "p_value"] < 0.05
            assert test_result.loc["x2", "p_value"] > 0.01

    def test_p_adjust_increases_pvalues(self):
        X, y = make_multifeature(n=2000)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, y, n_repeats=20)
        raw = result.test(method="t", alternative="greater")
        adjusted = result.test(method="t", alternative="greater", p_adjust="bonferroni")

        for feat in result.feature_names:
            assert adjusted.loc[feat, "p_value"] >= raw.loc[feat, "p_value"] - 1e-10


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestValidation:

    def test_distribution_required_for_resample(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="distribution must be"):
            explainer.loo(X, y, "resample", n_repeats=1)

    def test_distribution_forbidden_for_refit(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="distribution cannot be specified"):
            explainer.loo(X, y, "refit", distribution="marginal", learner=LinearRegression())

    def test_G_forbidden_for_marginal(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="G can only be specified"):
            explainer.loo(X, y, "resample", distribution="marginal", G=["x2"])

    def test_n_samples_required_for_marginalize(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="n_samples required"):
            explainer.loo(X, y, "marginalize", distribution="marginal")

    def test_n_samples_forbidden_for_resample(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="n_samples cannot"):
            explainer.loo(X, y, "resample", distribution="marginal", n_samples=10)

    def test_learner_required_for_refit(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="learner required"):
            explainer.loo(X, y, "refit")

    def test_learner_forbidden_for_resample(self):
        X, y = make_independent_linear()
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="learner cannot"):
            explainer.loo(X, y, "resample", distribution="marginal", learner=LinearRegression())

    def test_mismatched_columns_raises(self):
        X, y = make_independent_linear(n=200)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        X_wrong = X.rename(columns={"x1": "z1"})
        with pytest.raises(ValueError, match="missing columns"):
            explainer.pfi(X_wrong, y)

    def test_invalid_feature_name_raises(self):
        X, y = make_independent_linear(n=200)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        with pytest.raises(ValueError, match="Unknown columns"):
            explainer.pfi(X, y, features=["nonexistent_feature"])


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_feature(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(500)})
        y = 2 * X["x1"] + rng.randn(500) * 0.1
        predict = fit_linear(X, pd.Series(y))
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, pd.Series(y), n_repeats=5)
        assert result.scores.shape[3] == 1
        assert result.importance().loc["x1", "importance"] > 0

    def test_constant_feature(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(1000), "x_const": np.ones(1000)})
        y = 2 * X["x1"] + rng.randn(1000) * 0.1
        predict = fit_linear(X, pd.Series(y))
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, pd.Series(y), n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["x_const", "importance"]) < 0.01

    def test_reproducibility_with_seed(self):
        X, y = make_independent_linear(n=500)
        predict = fit_linear(X, y)
        explainer = Explainer(predict, X, loss=squared_error)

        np.random.seed(123)
        result1 = explainer.pfi(X, y, n_repeats=5)

        np.random.seed(123)
        result2 = explainer.pfi(X, y, n_repeats=5)

        np.testing.assert_array_equal(result1.scores, result2.scores)
