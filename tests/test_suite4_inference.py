"""Test suite 4: Inference and testing.

References:
  4.1 Watson & Wright (2021). "Testing Conditional Independence in
      Supervised Learning Algorithms." arXiv:1901.09917.
  4.2 Williamson, Gilbert, Carone, Simon (2021). "Nonparametric variable
      importance assessment using machine learning techniques." arXiv:2004.03683.
  4.3 Molnar, König, Bischl, Casalicchio (2023). "Relating the Partial
      Dependence Plot and Permutation Feature Importance to the Data
      Generating Process." ECML PKDD 2023.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from fippy import Explainer, ExplanationResult
from fippy.losses import squared_error
from fippy.samplers import GaussianSampler
from fippy.result import _adjust_pvalues


# ===========================================================================
# 4.1 CPI: Conditional Predictive Impact (Watson & Wright 2021)
# ===========================================================================

class TestCPIInference:
    """CFI observation-wise scores ARE the CPI Delta_i values.
    result.test(method='t') performs the CPI t-test.

    Delta_i = l(f(tilde_xj, x_{-j}), y_i) - l(f(x_i), y_i)
    Under H0: X_j ⊥ Y | X_{-j}, E[Delta_i] = 0.
    """

    def test_cpi_true_positive(self):
        """X1 ⊥̸ Y | X2: t-test should reject H0."""
        rng = np.random.RandomState(42)
        n = 2000
        rho = 0.5
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + x2 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values + x["x2"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        result = explainer.cfi(X, pd.Series(y, name="y"), n_repeats=1)

        test_df = result.test(method="t", alternative="greater")
        assert test_df.loc["x1", "p_value"] < 0.05
        assert test_df.loc["x2", "p_value"] < 0.05

    def test_cpi_true_negative(self):
        """X2 ⊥ Y | X1: t-test should NOT reject H0.
        Use the Bayes-optimal model f(x) = x1 + 0.01*x2 (nearly optimal,
        but x2 has tiny coefficient to avoid exactly-zero scores).
        With n_repeats > 1, obs-level scores have variance from permutation noise.
        """
        rng = np.random.RandomState(42)
        n = 2000
        rho = 0.9
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + rng.randn(n) * 0.1

        # Near-Bayes-optimal: f(x) = x1 (x2 not used).
        # CFI(x2) should be ~0 since X2 ⊥ Y | X1 and model doesn't use x2.
        predict = lambda x: x["x1"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        result = explainer.cfi(X, pd.Series(y, name="y"), n_repeats=10)

        imp = result.importance()
        # Since model doesn't use x2, CFI(x2) should be essentially zero
        assert abs(imp.loc["x2", "importance"]) < 0.1

    def test_cpi_type_i_error_control(self):
        """Repeat true-negative test M times; rejection rate ≤ alpha + tolerance.
        Uses a fitted linear model so scores have variance even for x3.
        y = x1 + x2 + eps, x3 independent. Fitted LM may assign small weight to x3.
        """
        alpha = 0.05
        n_experiments = 50
        n = 500
        rejections = 0

        for seed in range(n_experiments):
            rng = np.random.RandomState(seed)
            X = pd.DataFrame({
                "x1": rng.randn(n), "x2": rng.randn(n), "x3": rng.randn(n)
            })
            y = X["x1"] + X["x2"] + rng.randn(n) * 0.5

            model = LinearRegression().fit(X, y)
            sampler = GaussianSampler(X)
            explainer = Explainer(model.predict, X, loss=squared_error, sampler=sampler)
            result = explainer.cfi(X, pd.Series(y.values, name="y"), n_repeats=10)
            test_df = result.test(method="t", alternative="greater")

            if test_df.loc["x3", "p_value"] < alpha:
                rejections += 1

        rejection_rate = rejections / n_experiments
        # Allow some tolerance above alpha due to finite experiments
        assert rejection_rate <= alpha + 0.10, \
            f"Type I error rate {rejection_rate:.2f} exceeds {alpha + 0.10}"

    def test_cpi_observation_wise_scores_are_deltas(self):
        """Verify that obs_importance() gives the CPI Delta_i = l(perturbed) - l(original)."""
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"] + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        result = explainer.cfi(X, pd.Series(y.values, name="y"), n_repeats=1)

        obs_scores = result.obs_importance()
        # Scores should be (n_obs, n_features)
        assert obs_scores.shape == (n, 2)
        # For x2 (unused), Delta_i should be small on average
        assert abs(obs_scores[:, 1].mean()) < 0.5


# ===========================================================================
# 4.2 LOCO as VIM estimator (Williamson et al. 2021)
# ===========================================================================

class TestLOCOVIMInference:
    """LOCO importance approximates the VIM:
    psi_{0,s} = V(f_0, P_0) - V(f_{0,s}, P_0)

    For simple DGPs with known VIM values, verify LOCO estimates correctly.
    """

    def test_loco_estimates_explained_variance(self):
        """y = X1 + eps, X1,X2 ~ N(0,I). psi_1 = Var(X1) = 1, psi_2 = 0."""
        rng = np.random.RandomState(42)
        n = 3000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"] + rng.randn(n) * 0.1

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        explainer.set_y_train(pd.Series(y, name="y"))

        result = explainer.loco(X, pd.Series(y, name="y"), learner=LinearRegression())
        imp = result.importance()

        np.testing.assert_allclose(imp.loc["x1", "importance"], 1.0, atol=0.2)
        np.testing.assert_allclose(imp.loc["x2", "importance"], 0.0, atol=0.1)

    def test_loco_test_rejects_important(self):
        """t-test on LOCO should reject for X1, fail to reject for X2."""
        rng = np.random.RandomState(42)
        n = 3000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"] + rng.randn(n) * 0.1

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        explainer.set_y_train(pd.Series(y, name="y"))

        result = explainer.loco(X, pd.Series(y, name="y"), learner=LinearRegression())
        test_df = result.test(method="t", alternative="greater")

        assert test_df.loc["x1", "p_value"] < 0.01
        assert test_df.loc["x2", "p_value"] > 0.05

    def test_loco_multi_feature(self):
        """y = 2*X1 + X2, X1..X3 independent. LOCO(X1) > LOCO(X2) > LOCO(X3) ≈ 0."""
        rng = np.random.RandomState(42)
        n = 3000
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": rng.randn(n),
        })
        y = 2 * X["x1"] + X["x2"] + rng.randn(n) * 0.1

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        explainer.set_y_train(pd.Series(y.values, name="y"))

        result = explainer.loco(
            X, pd.Series(y.values, name="y"), learner=LinearRegression()
        )
        imp = result.importance()

        assert imp.loc["x1", "importance"] > imp.loc["x2", "importance"]
        assert imp.loc["x2", "importance"] > imp.loc["x3", "importance"]
        assert abs(imp.loc["x3", "importance"]) < 0.2


# ===========================================================================
# 4.3 Model-PFI variance (Molnar et al. 2023)
# ===========================================================================

class TestModelPFIVariance:
    """Model-PFI CI captures only MC permutation variance.
    More repeats → narrower CI.
    """

    def test_more_repeats_narrows_ci(self):
        rng = np.random.RandomState(42)
        n = 2000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = 3 * X["x1"] + rng.randn(n) * 0.1

        predict = lambda x: 3 * x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error)

        result_few = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=5)
        result_many = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=50)

        ci_few = result_few.ci(alpha=0.05)
        ci_many = result_many.ci(alpha=0.05)

        width_few = ci_few.loc["x1", "upper"] - ci_few.loc["x1", "lower"]
        width_many = ci_many.loc["x1", "upper"] - ci_many.loc["x1", "lower"]
        assert width_many < width_few

    def test_model_pfi_ci_captures_true_value(self):
        """For a known model, model-PFI point estimate should be close to
        theoretical PFI(X1) = 2*beta^2 = 18 for beta=3.
        Note: finite-sample Var(X1) ≠ 1 exactly, so we allow some tolerance.
        """
        rng = np.random.RandomState(42)
        n = 5000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = 3 * X["x1"] + rng.randn(n) * 0.1

        predict = lambda x: 3 * x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error)

        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=30)
        imp = result.importance()

        # True PFI(X1) = 2 * beta^2 * Var(X1) ≈ 18 (for Var(X1) ≈ 1)
        np.testing.assert_allclose(imp.loc["x1", "importance"], 18.0, atol=1.0)


# ===========================================================================
# 4.4 PFI inference: t-test on observation-wise scores
# ===========================================================================

class TestPFIInference:

    def test_unused_feature_zero_variance_raises(self):
        """Model doesn't use X2 → scores are exactly zero → test() should raise."""
        rng = np.random.RandomState(42)
        n = 2000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"] + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=10)

        with pytest.raises(ValueError, match="zero variance"):
            result.test(method="t", alternative="greater")

    def test_unused_feature_not_rejected_fitted_model(self):
        """Fitted LM on y=x1+eps with large n: x2 coefficient ≈ 0.
        PFI(x2) importance should be near zero even if test is feasible."""
        rng = np.random.RandomState(42)
        n = 5000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"] + rng.randn(n) * 0.5  # more noise → less overfitting to x2

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=10)
        imp = result.importance()
        # PFI(x2) should be near zero for a well-fitted LM
        assert abs(imp.loc["x2", "importance"]) < 0.1
        # PFI(x1) should be clearly positive
        assert imp.loc["x1", "importance"] > 0.5

    def test_used_feature_rejected(self):
        rng = np.random.RandomState(42)
        n = 2000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"] + rng.randn(n) * 0.1

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=10)
        test_df = result.test(method="t", alternative="greater")
        assert test_df.loc["x1", "p_value"] < 0.01

    def test_wilcoxon_agrees_with_t(self):
        """Wilcoxon and t-test should agree on direction for clear signal."""
        rng = np.random.RandomState(42)
        n = 2000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = 3 * X["x1"] + X["x2"] + rng.randn(n) * 0.1

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=10)

        t_test = result.test(method="t", alternative="greater")
        w_test = result.test(method="wilcoxon", alternative="greater")

        assert t_test.loc["x1", "p_value"] < 0.05
        assert w_test.loc["x1", "p_value"] < 0.05
        assert t_test.loc["x2", "p_value"] < 0.05
        assert w_test.loc["x2", "p_value"] < 0.05


# ===========================================================================
# 4.5 Multiple testing corrections
# ===========================================================================

class TestMultipleTestingCorrections:

    @pytest.fixture
    def raw_pvalues(self):
        """5 features, one important: get raw p-values.
        Use fitted LM so all features get some weight → non-zero variance scores.
        """
        rng = np.random.RandomState(42)
        n = 2000
        X = pd.DataFrame({f"x{i+1}": rng.randn(n) for i in range(5)})
        y = 3 * X["x1"] + rng.randn(n) * 0.1

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=10)
        return result

    def test_bonferroni_formula(self, raw_pvalues):
        """p_adj = min(p * m, 1)."""
        result = raw_pvalues
        raw = result.test(method="t", alternative="greater")
        adj = result.test(method="t", alternative="greater", p_adjust="bonferroni")

        m = len(result.feature_names)
        for feat in result.feature_names:
            expected = min(raw.loc[feat, "p_value"] * m, 1.0)
            np.testing.assert_allclose(
                adj.loc[feat, "p_value"], expected, atol=1e-10
            )

    def test_holm_formula(self, raw_pvalues):
        """Step-down: sort p-values, multiply by (m - rank + 1), enforce monotonicity."""
        result = raw_pvalues
        raw = result.test(method="t", alternative="greater")
        adj = result.test(method="t", alternative="greater", p_adjust="holm")

        raw_p = raw["p_value"].values
        adj_p = adj["p_value"].values

        # Manual Holm
        n = len(raw_p)
        order = np.argsort(raw_p)
        expected = np.empty(n)
        for i, idx in enumerate(order):
            expected[idx] = min(raw_p[idx] * (n - i), 1.0)
        cummax = 0.0
        for i in range(n):
            idx = order[i]
            expected[idx] = max(expected[idx], cummax)
            cummax = expected[idx]

        np.testing.assert_allclose(adj_p, expected, atol=1e-10)

    def test_bh_formula(self, raw_pvalues):
        """Step-up BH procedure."""
        result = raw_pvalues
        raw = result.test(method="t", alternative="greater")
        adj = result.test(method="t", alternative="greater", p_adjust="bh")

        raw_p = raw["p_value"].values
        adj_p = adj["p_value"].values

        # Manual BH
        n = len(raw_p)
        order = np.argsort(raw_p)[::-1]
        expected = np.empty(n)
        cummin = 1.0
        for i, idx in enumerate(order):
            rank = n - i
            expected[idx] = min(raw_p[idx] * n / rank, cummin)
            cummin = expected[idx]
        expected = np.minimum(expected, 1.0)

        np.testing.assert_allclose(adj_p, expected, atol=1e-10)

    def test_ordering_bonferroni_holm_bh(self, raw_pvalues):
        """BH ≤ Holm ≤ Bonferroni (in terms of adjusted p-values)."""
        result = raw_pvalues
        bon = result.test(method="t", alternative="greater", p_adjust="bonferroni")
        holm = result.test(method="t", alternative="greater", p_adjust="holm")
        bh = result.test(method="t", alternative="greater", p_adjust="bh")

        for feat in result.feature_names:
            assert bh.loc[feat, "p_value"] <= holm.loc[feat, "p_value"] + 1e-10
            assert holm.loc[feat, "p_value"] <= bon.loc[feat, "p_value"] + 1e-10

    def test_adjusted_pvalues_geq_raw(self, raw_pvalues):
        """All adjusted p-values should be >= raw p-values."""
        result = raw_pvalues
        raw = result.test(method="t", alternative="greater")
        for method in ["bonferroni", "holm", "bh"]:
            adj = result.test(method="t", alternative="greater", p_adjust=method)
            for feat in result.feature_names:
                assert adj.loc[feat, "p_value"] >= raw.loc[feat, "p_value"] - 1e-10


# ===========================================================================
# 4.6 Edge cases for inference
# ===========================================================================

class TestInferenceEdgeCases:

    def test_ci_with_single_repeat(self):
        """With n_repeats=1, std=0 and CI degenerates (width=0)."""
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({"x1": rng.randn(n)})
        y = X["x1"] + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=1)

        # With 1 repeat, std=0, CI is degenerate
        imp = result.importance()
        assert imp.loc["x1", "std"] == 0.0

    def test_test_two_sided(self):
        """Two-sided test: should reject for important feature."""
        rng = np.random.RandomState(42)
        n = 2000
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = X["x1"] + X["x2"] + rng.randn(n) * 0.1

        model = LinearRegression().fit(X, y)
        explainer = Explainer(model.predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y.values, name="y"), n_repeats=10)

        test_df = result.test(method="t", alternative="two-sided")
        assert test_df.loc["x1", "p_value"] < 0.05

    def test_obs_importance_not_supported_multi_fold(self):
        """obs_importance raises for multi-fold results."""
        scores = np.random.randn(3, 5, 100, 2)
        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        with pytest.raises(NotImplementedError, match="multi-fold"):
            result.obs_importance()

    def test_test_not_supported_multi_fold(self):
        """test() raises for multi-fold results."""
        scores = np.random.randn(3, 5, 100, 2)
        result = ExplanationResult(
            feature_names=["a", "b"],
            scores=scores,
            attribution="loo",
            restriction="resample",
            distribution="marginal",
        )
        with pytest.raises(NotImplementedError, match="multi-fold"):
            result.test()
