"""Test suite 3: Samplers, ExplanationResult array logic, and validation.

Tests:
  3.1 GaussianSampler: ground-truth conditional parameters
  3.2 PermutationSampler: marginal distribution preservation
  3.3 Two-sample distribution smoke tests
  3.4 ExplanationResult: array logic
  3.5 Validation error tests
  3.6 Categorical compatibility tests
  3.7 RFResidualSampler tests
  3.8 RFClassificationSampler tests
  3.9 TypeDispatchSampler + LOO integration
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

from fippy import Explainer, ExplanationResult
from fippy.losses import squared_error
from fippy.samplers import (
    GaussianSampler,
    PermutationSampler,
    RFClassificationSampler,
    RFResidualSampler,
    TypeDispatchSampler,
)


# ===========================================================================
# 3.1 GaussianSampler: ground-truth conditional parameters
# ===========================================================================

class TestGaussianSamplerGroundTruth:
    """Verify learned parameters and conditional samples match theory."""

    @pytest.fixture(scope="class")
    def gaussian_data(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 2.0, 0.7],
            [0.3, 0.7, 1.5],
        ])
        rng = np.random.RandomState(42)
        n = 20000
        L = np.linalg.cholesky(sigma)
        Z = rng.randn(n, 3)
        data = Z @ L.T + mu
        X = pd.DataFrame(data, columns=["x1", "x2", "x3"])
        return X, mu, sigma

    def test_learned_mean(self, gaussian_data):
        X, mu, _ = gaussian_data
        # GaussianSampler learns from training data, we can check via
        # sampling with empty S (marginal)
        sampler = GaussianSampler(X)
        X_test = pd.DataFrame({"x1": [0.0], "x2": [0.0], "x3": [0.0]})
        samples = sampler.sample(X_test, J=["x1"], S=[], n_samples=20000)
        sampled_mean = samples[0, :, 0].mean()
        np.testing.assert_allclose(sampled_mean, mu[0], atol=0.1)

    def test_learned_variance(self, gaussian_data):
        X, _, sigma = gaussian_data
        sampler = GaussianSampler(X)
        X_test = pd.DataFrame({"x1": [0.0], "x2": [0.0], "x3": [0.0]})
        samples = sampler.sample(X_test, J=["x1"], S=[], n_samples=20000)
        sampled_var = samples[0, :, 0].var()
        np.testing.assert_allclose(sampled_var, sigma[0, 0], atol=0.1)

    def test_conditional_mean_x1_given_x2x3(self, gaussian_data):
        """Verify mu_{1|23} = mu_1 + Sigma_{1,(23)} Sigma_{(23),(23)}^-1 (x_{23} - mu_{23})."""
        X, mu, sigma = gaussian_data
        sampler = GaussianSampler(X)

        # Condition on x2=3.0, x3=4.0
        x_s = np.array([3.0, 4.0])
        mu_s = mu[1:]
        sigma_js = sigma[0:1, 1:]  # (1, 2)
        sigma_ss = sigma[1:, 1:]   # (2, 2)
        expected_mean = mu[0] + (sigma_js @ np.linalg.solve(sigma_ss, x_s - mu_s))[0]

        X_test = pd.DataFrame({"x1": [0.0], "x2": [3.0], "x3": [4.0]})
        samples = sampler.sample(X_test, J=["x1"], S=["x2", "x3"], n_samples=20000)
        sampled_mean = samples[0, :, 0].mean()

        np.testing.assert_allclose(sampled_mean, expected_mean, atol=0.1)

    def test_conditional_variance_x1_given_x2x3(self, gaussian_data):
        """Verify Sigma_{1|23} = Sigma_11 - Sigma_{1,(23)} Sigma_{(23),(23)}^-1 Sigma_{(23),1}."""
        X, mu, sigma = gaussian_data
        sampler = GaussianSampler(X)

        sigma_js = sigma[0:1, 1:]
        sigma_ss = sigma[1:, 1:]
        sigma_sj = sigma[1:, 0:1]
        expected_var = sigma[0, 0] - (sigma_js @ np.linalg.solve(sigma_ss, sigma_sj))[0, 0]

        X_test = pd.DataFrame({"x1": [0.0], "x2": [3.0], "x3": [4.0]})
        samples = sampler.sample(X_test, J=["x1"], S=["x2", "x3"], n_samples=20000)
        sampled_var = samples[0, :, 0].var()

        np.testing.assert_allclose(sampled_var, expected_var, atol=0.1)

    def test_output_shape(self, gaussian_data):
        X, _, _ = gaussian_data
        sampler = GaussianSampler(X)
        result = sampler.sample(X.head(10), J=["x1", "x2"], S=["x3"], n_samples=5)
        assert result.shape == (10, 5, 2)

    def test_multivariate_conditional_shape(self, gaussian_data):
        X, _, _ = gaussian_data
        sampler = GaussianSampler(X)
        result = sampler.sample(X.head(50), J=["x1", "x2"], S=["x3"], n_samples=3)
        assert result.shape == (50, 3, 2)


# ===========================================================================
# 3.2 PermutationSampler: marginal distribution preservation
# ===========================================================================

class TestPermutationSamplerMarginal:

    @pytest.fixture(scope="class")
    def perm_data(self):
        rng = np.random.RandomState(42)
        n = 10000
        x1 = rng.randn(n) * 3 + 5
        x2 = 0.8 * x1 + rng.randn(n)  # correlated
        X = pd.DataFrame({"x1": x1, "x2": x2})
        return X

    def test_marginal_mean(self, perm_data):
        X = perm_data
        sampler = PermutationSampler(X)
        X_test = X.head(1)
        samples = sampler.sample(X_test, J=["x1"], S=["x2"], n_samples=10000)
        np.testing.assert_allclose(samples[0, :, 0].mean(), X["x1"].mean(), atol=0.2)

    def test_marginal_variance(self, perm_data):
        X = perm_data
        sampler = PermutationSampler(X)
        X_test = X.head(1)
        samples = sampler.sample(X_test, J=["x1"], S=["x2"], n_samples=10000)
        np.testing.assert_allclose(samples[0, :, 0].std(), X["x1"].std(), atol=0.2)

    def test_marginal_ignores_conditioning(self, perm_data):
        """Samples should not depend on X_S value."""
        X = perm_data
        sampler = PermutationSampler(X)
        X_low = pd.DataFrame({"x1": [0.0], "x2": [-100.0]})
        X_high = pd.DataFrame({"x1": [0.0], "x2": [100.0]})
        np.random.seed(0)
        s_low = sampler.sample(X_low, J=["x1"], S=["x2"], n_samples=100)
        np.random.seed(0)
        s_high = sampler.sample(X_high, J=["x1"], S=["x2"], n_samples=100)
        np.testing.assert_array_equal(s_low, s_high)

    def test_no_correlation_with_other_features(self, perm_data):
        """Permutation-sampled X1 should be uncorrelated with X2."""
        X = perm_data
        sampler = PermutationSampler(X)
        n_test = 1000
        X_test = X.head(n_test)
        # One sample per observation
        samples = sampler.sample(X_test, J=["x1"], S=["x2"], n_samples=1)
        sampled_x1 = samples[:, 0, 0]
        x2_vals = X_test["x2"].values
        corr = np.corrcoef(sampled_x1, x2_vals)[0, 1]
        assert abs(corr) < 0.1


# ===========================================================================
# 3.3 Two-sample distribution smoke tests
# ===========================================================================

class TestSamplerDistributionTests:

    def test_ks_test_permutation_sampler(self):
        """KS test: permutation samples match training marginal."""
        rng = np.random.RandomState(42)
        n = 10000
        X = pd.DataFrame({"x1": rng.randn(n) * 2 + 1, "x2": rng.randn(n)})
        sampler = PermutationSampler(X)
        samples = sampler.sample(X.head(1), J=["x1"], S=["x2"], n_samples=5000)
        sampled = samples[0, :, 0]
        # KS test against training distribution
        ks_stat, p_value = stats.ks_2samp(sampled, X["x1"].values)
        assert p_value > 0.01  # Should not reject that they're from same dist

    def test_ks_test_gaussian_sampler_marginal(self):
        """KS test: Gaussian sampler with S=[] should match marginal."""
        rng = np.random.RandomState(42)
        n = 10000
        X = pd.DataFrame({"x1": rng.randn(n) * 2 + 1, "x2": rng.randn(n)})
        sampler = GaussianSampler(X)
        X_test = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
        samples = sampler.sample(X_test, J=["x1"], S=[], n_samples=5000)
        sampled = samples[0, :, 0]

        # Should be N(mu_x1, sigma_x1^2) ≈ N(1, 4)
        ks_stat, p_value = stats.kstest(
            sampled, "norm", args=(X["x1"].mean(), X["x1"].std())
        )
        assert p_value > 0.01

    def test_skewness_permutation_sampler(self):
        """Skewness of permutation samples should match training."""
        rng = np.random.RandomState(42)
        n = 10000
        # Skewed distribution
        x1 = rng.exponential(2, n)
        X = pd.DataFrame({"x1": x1, "x2": rng.randn(n)})
        sampler = PermutationSampler(X)
        samples = sampler.sample(X.head(1), J=["x1"], S=["x2"], n_samples=10000)
        sampled = samples[0, :, 0]

        train_skew = stats.skew(X["x1"].values)
        sample_skew = stats.skew(sampled)
        np.testing.assert_allclose(sample_skew, train_skew, atol=0.3)

    def test_gaussian_conditional_cross_covariance(self):
        """Cov(tilde_X1, X2) should match theoretical conditional covariance."""
        rng = np.random.RandomState(42)
        n = 10000
        rho = 0.7
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        sampler = GaussianSampler(X)

        # Sample X1 | X2 for many observations
        samples = sampler.sample(X, J=["x1"], S=["x2"], n_samples=1)
        sampled_x1 = samples[:, 0, 0]
        x2_vals = X["x2"].values

        # Cov(X1|X2, X2) = rho (from the regression E[X1|X2] = rho*X2)
        # The sampled X1 = rho*X2 + noise, so corr(sampled_X1, X2) = rho
        corr = np.corrcoef(sampled_x1, x2_vals)[0, 1]
        np.testing.assert_allclose(corr, rho, atol=0.1)


# ===========================================================================
# 3.4 ExplanationResult: array logic
# ===========================================================================

class TestExplanationResultArrayLogic:

    def test_constant_scores_mean_and_std(self):
        """Constant scores → mean = c, std = 0."""
        c = 3.5
        scores = np.full((1, 5, 100, 3), c)
        result = ExplanationResult(
            feature_names=["a", "b", "c"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        imp = result.importance()
        for feat in ["a", "b", "c"]:
            assert imp.loc[feat, "importance"] == pytest.approx(c)
            assert imp.loc[feat, "std"] == pytest.approx(0.0, abs=1e-10)

    def test_known_std_across_repeats(self):
        """Scores where repeat r has all values = r. Std should match."""
        n_repeats = 10
        n_obs = 50
        n_features = 2
        scores = np.zeros((1, n_repeats, n_obs, n_features))
        for r in range(n_repeats):
            scores[0, r, :, :] = float(r)

        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        imp = result.importance()
        expected_mean = np.mean(range(n_repeats))
        expected_std = np.std(range(n_repeats))  # ddof=0 for population std

        for feat in ["a", "b"]:
            np.testing.assert_allclose(imp.loc[feat, "importance"], expected_mean, atol=1e-10)
            np.testing.assert_allclose(imp.loc[feat, "std"], expected_std, atol=1e-10)

    def test_obs_importance_shape_and_averaging(self):
        """obs_importance averages over repeats, returns (n_obs, n_features)."""
        n_repeats = 4
        n_obs = 20
        scores = np.random.RandomState(42).randn(1, n_repeats, n_obs, 3)
        result = ExplanationResult(
            feature_names=["a", "b", "c"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        obs = result.obs_importance()
        assert obs.shape == (n_obs, 3)
        np.testing.assert_allclose(obs, scores[0].mean(axis=0))

    def test_ci_contains_mean(self):
        """CI should contain the point estimate."""
        rng = np.random.RandomState(42)
        scores = rng.randn(1, 30, 100, 2) + 5.0  # mean ≈ 5
        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        ci = result.ci(alpha=0.05, method="t")
        imp = result.importance()
        for feat in ["a", "b"]:
            assert ci.loc[feat, "lower"] <= imp.loc[feat, "importance"]
            assert imp.loc[feat, "importance"] <= ci.loc[feat, "upper"]

    def test_ci_quantile_method(self):
        """Quantile CI should also contain the mean for well-behaved data."""
        rng = np.random.RandomState(42)
        scores = rng.randn(1, 50, 100, 2) + 5.0
        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        ci = result.ci(alpha=0.05, method="quantile")
        imp = result.importance()
        for feat in ["a", "b"]:
            assert ci.loc[feat, "lower"] <= imp.loc[feat, "importance"]
            assert imp.loc[feat, "importance"] <= ci.loc[feat, "upper"]

    def test_multi_fold_requires_explicit_aggregate(self):
        """n_folds > 1 without aggregate_over should raise."""
        scores = np.random.randn(3, 5, 100, 2)
        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        with pytest.raises(ValueError, match="n_folds > 1"):
            result.importance()

    def test_multi_fold_aggregate_folds(self):
        """aggregate_over='folds' computes std over fold-level means."""
        rng = np.random.RandomState(42)
        n_folds, n_repeats, n_obs = 5, 3, 100
        scores = np.zeros((n_folds, n_repeats, n_obs, 2))
        for k in range(n_folds):
            scores[k, :, :, :] = float(k)  # fold k has all values = k

        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        imp = result.importance(aggregate_over="folds")
        expected_mean = np.mean(range(n_folds))
        expected_std = np.std(range(n_folds))

        for feat in ["a", "b"]:
            np.testing.assert_allclose(imp.loc[feat, "importance"], expected_mean, atol=1e-10)
            np.testing.assert_allclose(imp.loc[feat, "std"], expected_std, atol=1e-10)

    def test_serialization_roundtrip(self, tmp_path):
        scores = np.random.RandomState(42).randn(1, 3, 50, 2)
        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="shapley",
            restriction="marginalize",
            distribution="marginal",
        )
        path = str(tmp_path / "result.csv")
        result.to_csv(path)
        loaded = ExplanationResult.from_csv(path)

        assert loaded.feature_names == result.feature_names
        assert loaded.attribution == result.attribution
        assert loaded.restriction == result.restriction
        assert loaded.distribution == result.distribution
        np.testing.assert_allclose(loaded.scores, result.scores)

    def test_method_string(self):
        result = ExplanationResult(
            feature_names=["a"],
            scores=np.zeros((1, 1, 10, 1)),
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        assert result.method == "loo_resample_marginal"

    def test_method_string_no_distribution(self):
        result = ExplanationResult(
            feature_names=["a"],
            scores=np.zeros((1, 1, 10, 1)),
            attribution="loo",
            restriction="refit",
        )
        assert result.method == "loo_refit"


# ===========================================================================
# 3.5 Validation error tests
# ===========================================================================

class TestValidation:

    @pytest.fixture
    def basic_explainer(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(200), "x2": rng.randn(200)})
        y = X["x1"] + rng.randn(200) * 0.1
        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error)
        return explainer, X, pd.Series(y.values, name="y")

    def test_refit_with_distribution_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="distribution cannot be specified"):
            explainer.loo(X, y, "refit", distribution="marginal", learner=LinearRegression())

    def test_resample_without_distribution_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="distribution must be"):
            explainer.loo(X, y, "resample")

    def test_resample_with_learner_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="learner cannot"):
            explainer.loo(X, y, "resample", distribution="marginal",
                          learner=LinearRegression())

    def test_refit_without_learner_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="learner required"):
            explainer.loo(X, y, "refit")

    def test_conditional_without_sampler_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="requires a sampler"):
            explainer.loo(X, y, "resample", distribution="conditional")

    def test_marginalize_without_n_samples_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="n_samples required"):
            explainer.loo(X, y, "marginalize", distribution="marginal")

    def test_n_samples_with_resample_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="n_samples cannot"):
            explainer.loo(X, y, "resample", distribution="marginal", n_samples=10)

    def test_G_with_marginal_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="G can only be specified"):
            explainer.loo(X, y, "resample", distribution="marginal", G=["x2"])

    def test_missing_columns_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        X_wrong = X.rename(columns={"x1": "z1"})
        with pytest.raises(ValueError, match="missing columns"):
            explainer.pfi(X_wrong, y)

    def test_unknown_feature_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="Unknown columns"):
            explainer.pfi(X, y, features=["nonexistent"])

    def test_invalid_restriction_raises(self, basic_explainer):
        explainer, X, y = basic_explainer
        with pytest.raises(ValueError, match="Invalid restriction"):
            explainer.loo(X, y, "invalid_restriction", distribution="marginal")


# ===========================================================================
# 3.6 Categorical compatibility tests
# ===========================================================================

class TestCategoricalCompatibility:
    """Test that samplers fail fast on categorical data when unsupported."""

    @pytest.fixture
    def categorical_data(self):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        })
        y = X["x1"] + rng.randn(n) * 0.1
        return X, y

    def test_gaussian_sampler_rejects_categorical_target(self, categorical_data):
        X, y = categorical_data
        sampler = GaussianSampler(X)
        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        with pytest.raises(ValueError, match="categorical targets"):
            explainer.cfi(X, y)

    def test_gaussian_sampler_rejects_categorical_context(self, categorical_data):
        X, y = categorical_data
        sampler = GaussianSampler(X)
        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        with pytest.raises(ValueError, match="categorical context"):
            explainer.cfi(X, y)

    def test_gaussian_sampler_rejects_object_dtype(self):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.choice(["a", "b"], n),  # object dtype
        })
        y = X["x1"] + rng.randn(n) * 0.1
        sampler = GaussianSampler(X)
        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        with pytest.raises(ValueError, match="categorical targets"):
            explainer.cfi(X, y)

    def test_permutation_sampler_accepts_categorical(self, categorical_data):
        X, _ = categorical_data
        sampler = PermutationSampler(X)
        # PermutationSampler supports categorical in both roles — check passes
        sampler.check_compatibility(requires_multivariate=True)

    def test_check_compatibility_reports_both_errors(self, categorical_data):
        """When both target and context are unsupported, both errors are reported."""
        X, _ = categorical_data
        sampler = GaussianSampler(X)
        with pytest.raises(ValueError, match="(?s)categorical targets.*categorical context"):
            sampler.check_compatibility()

    def test_no_error_on_numeric_only_data(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(100), "x2": rng.randn(100)})
        sampler = GaussianSampler(X)
        # Should not raise
        sampler.check_compatibility()

    def test_integer_columns_not_detected_as_categorical(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(100), "x2": rng.randint(0, 3, 100)})
        sampler = GaussianSampler(X)
        assert sampler._categorical_cols == set()

    def test_astype_category_detected(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(100), "x2": rng.randint(0, 3, 100)})
        X["x2"] = X["x2"].astype("category")
        sampler = GaussianSampler(X)
        assert sampler._categorical_cols == {"x2"}

    def test_gaussian_rejects_categorical_in_shapley(self, categorical_data):
        X, y = categorical_data
        sampler = GaussianSampler(X)
        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        with pytest.raises(ValueError, match="categorical targets"):
            explainer.sage(X, y, distribution="conditional", n_samples=2,
                           n_permutations=2)

    def test_marginal_without_sampler_skips_check(self):
        """PFI with no sampler uses internal _PermutationSampler — no compatibility check."""
        rng = np.random.RandomState(42)
        n = 50
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"].values
        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error)
        # No sampler provided → internal fallback, no check_compatibility call
        result = explainer.pfi(X, y)
        assert result.scores.shape[2] == n


# ===========================================================================
# 3.7 RFResidualSampler: continuous conditional sampling
# ===========================================================================

class TestRFResidualSampler:

    @pytest.fixture(scope="class")
    def linear_data(self):
        """x2 = 2*x1 + noise, so E[x2|x1] ≈ 2*x1."""
        rng = np.random.RandomState(42)
        n = 500
        x1 = rng.randn(n)
        noise = rng.randn(n) * 0.3
        x2 = 2.0 * x1 + noise
        return pd.DataFrame({"x1": x1, "x2": x2})

    def test_output_shape(self, linear_data):
        X = linear_data
        sampler = RFResidualSampler(X, tune=False, random_state=42)
        result = sampler.sample(X.head(10), J=["x2"], S=["x1"], n_samples=5)
        assert result.shape == (10, 5, 1)

    def test_conditional_mean_recovered(self, linear_data):
        """Conditional mean E[x2|x1] ≈ 2*x1 should be recovered."""
        X = linear_data
        sampler = RFResidualSampler(X, tune=False, random_state=42)
        # Condition on x1=1.0 → expect mean ≈ 2.0
        X_test = pd.DataFrame({"x1": [1.0], "x2": [0.0]})
        samples = sampler.sample(X_test, J=["x2"], S=["x1"], n_samples=5000)
        sampled_mean = samples[0, :, 0].mean()
        np.testing.assert_allclose(sampled_mean, 2.0, atol=0.3)

    def test_residual_variance_preserved(self, linear_data):
        """Variance of samples should reflect residual variance ≈ 0.3^2."""
        X = linear_data
        sampler = RFResidualSampler(X, tune=False, random_state=42)
        X_test = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
        samples = sampler.sample(X_test, J=["x2"], S=["x1"], n_samples=5000)
        sampled_std = samples[0, :, 0].std()
        # Residual std ≈ 0.3, but RF training residuals are smaller (overfitting)
        # Just check it's in a reasonable range
        assert 0.05 < sampled_std < 0.6

    def test_empty_S_gives_marginal(self, linear_data):
        """With S=[], samples should match the marginal of x2."""
        X = linear_data
        sampler = RFResidualSampler(X, tune=False, random_state=42)
        X_test = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
        samples = sampler.sample(X_test, J=["x2"], S=[], n_samples=5000)
        sampled_mean = samples[0, :, 0].mean()
        np.testing.assert_allclose(sampled_mean, X["x2"].mean(), atol=0.2)

    def test_categorical_context(self):
        """RF can condition on categorical features."""
        rng = np.random.RandomState(42)
        n = 300
        cat = rng.choice(["a", "b"], n)
        # x1 depends on category: a → mean 0, b → mean 5
        x1 = np.where(cat == "a", rng.randn(n), rng.randn(n) + 5.0)
        X = pd.DataFrame({"x1": x1, "cat": pd.Categorical(cat)})

        sampler = RFResidualSampler(X, tune=False, random_state=42)
        X_a = pd.DataFrame({"x1": [0.0], "cat": pd.Categorical(["a"], categories=["a", "b"])})
        X_b = pd.DataFrame({"x1": [0.0], "cat": pd.Categorical(["b"], categories=["a", "b"])})

        samples_a = sampler.sample(X_a, J=["x1"], S=["cat"], n_samples=3000)
        samples_b = sampler.sample(X_b, J=["x1"], S=["cat"], n_samples=3000)

        mean_a = samples_a[0, :, 0].mean()
        mean_b = samples_b[0, :, 0].mean()
        # Means should be clearly separated
        assert mean_b - mean_a > 2.0

    def test_with_tuning(self, linear_data):
        """Tuned sampler should produce reasonable samples."""
        X = linear_data
        sampler = RFResidualSampler(X, tune=True, n_iter=5, cv=3, random_state=42)
        X_test = pd.DataFrame({"x1": [1.0], "x2": [0.0]})
        samples = sampler.sample(X_test, J=["x2"], S=["x1"], n_samples=3000)
        sampled_mean = samples[0, :, 0].mean()
        np.testing.assert_allclose(sampled_mean, 2.0, atol=0.4)

    def test_caching(self, linear_data):
        """Second call with same (J, S) should use cache."""
        X = linear_data
        sampler = RFResidualSampler(X, tune=False, random_state=42)
        sampler.fit(["x2"], ["x1"])
        assert len(sampler._cache) == 1
        sampler.fit(["x2"], ["x1"])  # should hit cache
        assert len(sampler._cache) == 1
        sampler.fit(["x1"], ["x2"])  # different (J, S)
        assert len(sampler._cache) == 2

    def test_capability_flags(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(50), "x2": rng.randn(50)})
        sampler = RFResidualSampler(X, tune=False)
        assert sampler.multivariate is False
        assert sampler.supports_categorical_target is False
        assert sampler.supports_categorical_context is True

    def test_rejects_categorical_target(self):
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "cat": pd.Categorical(rng.choice(["a", "b"], n)),
        })
        sampler = RFResidualSampler(X, tune=False)
        with pytest.raises(ValueError, match="categorical targets"):
            sampler.check_compatibility()


# ===========================================================================
# 3.8 RFClassificationSampler: categorical conditional sampling
# ===========================================================================

class TestRFClassificationSampler:

    @pytest.fixture(scope="class")
    def categorical_data(self):
        """cat depends on x1: x1 > 0 → mostly 'a', x1 < 0 → mostly 'b'."""
        rng = np.random.RandomState(42)
        n = 500
        x1 = rng.randn(n)
        probs_a = 1 / (1 + np.exp(-3 * x1))  # sigmoid
        cat = np.where(rng.rand(n) < probs_a, "a", "b")
        return pd.DataFrame({"x1": x1, "cat": pd.Categorical(cat)})

    def test_output_shape(self, categorical_data):
        X = categorical_data
        sampler = RFClassificationSampler(X, tune=False, random_state=42)
        result = sampler.sample(X.head(10), J=["cat"], S=["x1"], n_samples=5)
        assert result.shape == (10, 5, 1)

    def test_samples_are_valid_categories(self, categorical_data):
        X = categorical_data
        sampler = RFClassificationSampler(X, tune=False, random_state=42)
        result = sampler.sample(X.head(20), J=["cat"], S=["x1"], n_samples=10)
        unique_vals = set(result.ravel())
        assert unique_vals <= {"a", "b"}

    def test_conditional_probabilities(self, categorical_data):
        """For x1=2 (large positive), P(cat='a'|x1) should be high."""
        X = categorical_data
        sampler = RFClassificationSampler(X, tune=False, random_state=42)
        X_pos = pd.DataFrame({"x1": [2.0], "cat": pd.Categorical(["a"], categories=["a", "b"])})
        X_neg = pd.DataFrame({"x1": [-2.0], "cat": pd.Categorical(["a"], categories=["a", "b"])})

        samples_pos = sampler.sample(X_pos, J=["cat"], S=["x1"], n_samples=3000)
        samples_neg = sampler.sample(X_neg, J=["cat"], S=["x1"], n_samples=3000)

        frac_a_pos = (samples_pos[0, :, 0] == "a").mean()
        frac_a_neg = (samples_neg[0, :, 0] == "a").mean()

        # x1=2 should give much more 'a' than x1=-2
        assert frac_a_pos > 0.7
        assert frac_a_neg < 0.3

    def test_empty_S_gives_marginal(self, categorical_data):
        """With S=[], samples should match marginal class frequencies."""
        X = categorical_data
        sampler = RFClassificationSampler(X, tune=False, random_state=42)
        X_test = pd.DataFrame({"x1": [0.0], "cat": pd.Categorical(["a"], categories=["a", "b"])})
        samples = sampler.sample(X_test, J=["cat"], S=[], n_samples=5000)
        frac_a = (samples[0, :, 0] == "a").mean()
        expected_frac_a = (X["cat"] == "a").mean()
        np.testing.assert_allclose(frac_a, expected_frac_a, atol=0.05)

    def test_with_tuning(self, categorical_data):
        """Tuned sampler should produce valid samples."""
        X = categorical_data
        sampler = RFClassificationSampler(X, tune=True, n_iter=5, cv=3, random_state=42)
        result = sampler.sample(X.head(5), J=["cat"], S=["x1"], n_samples=10)
        assert result.shape == (5, 10, 1)
        assert set(result.ravel()) <= {"a", "b"}

    def test_multiclass(self):
        """Works with more than 2 classes."""
        rng = np.random.RandomState(42)
        n = 300
        x1 = rng.randn(n)
        # 3 classes with different means
        cat = np.where(x1 > 0.5, "high", np.where(x1 < -0.5, "low", "mid"))
        X = pd.DataFrame({"x1": x1, "cat": pd.Categorical(cat)})

        sampler = RFClassificationSampler(X, tune=False, random_state=42)
        X_high = pd.DataFrame({
            "x1": [2.0],
            "cat": pd.Categorical(["high"], categories=["high", "low", "mid"]),
        })
        samples = sampler.sample(X_high, J=["cat"], S=["x1"], n_samples=3000)
        frac_high = (samples[0, :, 0] == "high").mean()
        assert frac_high > 0.5

    def test_categorical_context(self):
        """Classifier can condition on categorical context features."""
        rng = np.random.RandomState(42)
        n = 300
        ctx = rng.choice(["x", "y"], n)
        # target depends on context: ctx='x' → mostly 'a', ctx='y' → mostly 'b'
        target = np.where(
            ctx == "x",
            np.where(rng.rand(n) < 0.9, "a", "b"),
            np.where(rng.rand(n) < 0.1, "a", "b"),
        )
        X = pd.DataFrame({
            "ctx": pd.Categorical(ctx),
            "target": pd.Categorical(target),
        })

        sampler = RFClassificationSampler(X, tune=False, random_state=42)
        X_x = pd.DataFrame({
            "ctx": pd.Categorical(["x"], categories=["x", "y"]),
            "target": pd.Categorical(["a"], categories=["a", "b"]),
        })
        samples = sampler.sample(X_x, J=["target"], S=["ctx"], n_samples=3000)
        frac_a = (samples[0, :, 0] == "a").mean()
        assert frac_a > 0.7

    def test_capability_flags(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "x1": rng.randn(50),
            "cat": pd.Categorical(rng.choice(["a", "b"], 50)),
        })
        sampler = RFClassificationSampler(X, tune=False)
        assert sampler.multivariate is False
        assert sampler.supports_categorical_target is True
        assert sampler.supports_categorical_context is True

    def test_passes_compatibility_with_categoricals(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "x1": rng.randn(50),
            "cat": pd.Categorical(rng.choice(["a", "b"], 50)),
        })
        sampler = RFClassificationSampler(X, tune=False)
        # Should not raise — supports both categorical roles
        sampler.check_compatibility()


# ===========================================================================
# 3.9 TypeDispatchSampler + LOO integration
# ===========================================================================

class TestTypeDispatchSamplerLOO:
    """End-to-end test: TypeDispatchSampler wrapping RF samplers for LOO/CFI
    on a mixed continuous + categorical dataset."""

    @pytest.fixture(scope="class")
    def mixed_data(self):
        rng = np.random.RandomState(42)
        n = 300
        x1 = rng.randn(n)
        cat = pd.Categorical(rng.choice(["a", "b"], n))
        # y = x1 + effect_of_cat + noise
        y = x1 + np.where(cat == "a", 2.0, -2.0) + rng.randn(n) * 0.1
        X = pd.DataFrame({"x1": x1, "cat": cat})
        return X, y

    @pytest.fixture(scope="class")
    def dispatch_sampler(self, mixed_data):
        X, _ = mixed_data
        return TypeDispatchSampler(
            X,
            continuous_sampler=RFResidualSampler(X, tune=False, random_state=42),
            categorical_sampler=RFClassificationSampler(X, tune=False, random_state=42),
        )

    def test_cfi_runs_and_returns_correct_shape(self, mixed_data, dispatch_sampler):
        X, y = mixed_data
        X_test, y_test = X.head(30), y[:30]

        def predict(X_df):
            x1_val = X_df["x1"].values.astype(float)
            cat_val = np.where(X_df["cat"] == "a", 2.0, -2.0)
            return x1_val + cat_val

        explainer = Explainer(predict, X, loss=squared_error, sampler=dispatch_sampler)
        result = explainer.cfi(X_test, y_test, n_repeats=3)

        assert result.scores.shape == (1, 3, 30, 2)
        assert result.feature_names == ["x1", "cat"]

    def test_both_features_have_positive_importance(self, mixed_data, dispatch_sampler):
        """Both x1 and cat contribute to y, so both should have positive importance."""
        X, y = mixed_data

        def predict(X_df):
            x1_val = X_df["x1"].values.astype(float)
            cat_val = np.where(X_df["cat"] == "a", 2.0, -2.0)
            return x1_val + cat_val

        explainer = Explainer(predict, X, loss=squared_error, sampler=dispatch_sampler)
        result = explainer.cfi(X, y, n_repeats=5)
        imp = result.importance()

        assert imp.loc["x1", "importance"] > 0
        assert imp.loc["cat", "importance"] > 0

    def test_dispatches_to_correct_sub_sampler(self, mixed_data, dispatch_sampler):
        """Continuous target → RFResidualSampler, categorical → RFClassificationSampler."""
        X, _ = mixed_data
        assert isinstance(dispatch_sampler._select_sampler(["x1"]), RFResidualSampler)
        assert isinstance(dispatch_sampler._select_sampler(["cat"]), RFClassificationSampler)

    def test_compatibility_check_passes(self, dispatch_sampler):
        """TypeDispatchSampler supports both categorical roles."""
        dispatch_sampler.check_compatibility(requires_multivariate=False)

    def test_compatibility_check_rejects_multivariate(self, dispatch_sampler):
        """TypeDispatchSampler is univariate — multivariate request fails."""
        with pytest.raises(ValueError, match="univariate"):
            dispatch_sampler.check_compatibility(requires_multivariate=True)

    def test_n_repeats_produce_different_scores(self, mixed_data, dispatch_sampler):
        """Different repeats should give different scores (randomness in sampling)."""
        X, y = mixed_data
        X_test, y_test = X.head(20), y[:20]

        def predict(X_df):
            x1_val = X_df["x1"].values.astype(float)
            cat_val = np.where(X_df["cat"] == "a", 2.0, -2.0)
            return x1_val + cat_val

        explainer = Explainer(predict, X, loss=squared_error, sampler=dispatch_sampler)
        result = explainer.cfi(X_test, y_test, n_repeats=2)

        # Repeat 0 and repeat 1 should differ (different random draws)
        assert not np.allclose(result.scores[0, 0, :, :], result.scores[0, 1, :, :])
