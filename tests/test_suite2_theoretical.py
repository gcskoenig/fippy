"""Test suite 2: Theoretical derivations for importance methods.

DGPs with known models (hard-coded prediction functions) where
theoretical importance values can be derived analytically.

References:
  Ewald et al. (2024), arXiv:2404.12862
  Watson & Wright (2021), arXiv:1901.09917
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from fippy import Explainer
from fippy.losses import squared_error
from fippy.samplers import GaussianSampler

from conftest import (
    make_independent_linear,
    make_correlated_gaussian,
    make_bivariate_gaussian,
    make_trivariate_gaussian,
)


# ---------------------------------------------------------------------------
# 2.1 PFI: unused features get zero importance
# ---------------------------------------------------------------------------

class TestPFIUnusedFeatureZero:
    """y = X1 + X2 + eps, model = X1 + X2. X3 unused.
    PFI(X3) = 0. PFI(X1) = PFI(X2) = 2*Var(Xj) = 2.
    """

    @pytest.fixture(scope="class")
    def setup(self):
        rng = np.random.RandomState(42)
        n = 5000
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": rng.randn(n),
        })
        y = X["x1"] + X["x2"] + rng.randn(n) * 0.1
        predict = lambda x: x["x1"].values + x["x2"].values
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y, name="y"), n_repeats=20)
        return result.importance()

    def test_unused_feature_zero(self, setup):
        imp = setup
        assert abs(imp.loc["x3", "importance"]) < 0.2

    def test_used_features_equal_2(self, setup):
        imp = setup
        np.testing.assert_allclose(imp.loc["x1", "importance"], 2.0, atol=0.3)
        np.testing.assert_allclose(imp.loc["x2", "importance"], 2.0, atol=0.3)

    def test_used_features_symmetric(self, setup):
        imp = setup
        np.testing.assert_allclose(
            imp.loc["x1", "importance"],
            imp.loc["x2", "importance"],
            atol=0.3,
        )


# ---------------------------------------------------------------------------
# 2.2 PFI: scaled coefficients
# ---------------------------------------------------------------------------

class TestPFIScaledCoefficients:
    """y = beta1*X1 + beta2*X2 + eps, model = beta1*X1 + beta2*X2.
    PFI(Xj) = 2*beta_j^2.
    """

    @pytest.fixture(scope="class")
    def setup(self):
        rng = np.random.RandomState(42)
        n = 5000
        beta1, beta2 = 3.0, 1.0
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.randn(n),
        })
        y = beta1 * X["x1"] + beta2 * X["x2"] + rng.randn(n) * 0.1
        predict = lambda x: beta1 * x["x1"].values + beta2 * x["x2"].values
        explainer = Explainer(predict, X, loss=squared_error)
        result = explainer.pfi(X, pd.Series(y, name="y"), n_repeats=20)
        return result.importance()

    def test_pfi_x1_equals_2_beta1_sq(self, setup):
        # PFI(X1) = 2 * 3^2 = 18
        np.testing.assert_allclose(setup.loc["x1", "importance"], 18.0, atol=1.5)

    def test_pfi_x2_equals_2_beta2_sq(self, setup):
        # PFI(X2) = 2 * 1^2 = 2
        np.testing.assert_allclose(setup.loc["x2", "importance"], 2.0, atol=0.5)


# ---------------------------------------------------------------------------
# 2.3 CFI: conditionally independent features get zero importance
# ---------------------------------------------------------------------------

class TestCFIConditionalIndependence:
    """DGP1: (X1,X2,X3) ~ N(0, Sigma), y = X2 + X3 + eps.
    Model = X2 + X3. CFI(X1) = 0 (unused).

    DGP2: (X1,X2) correlated, y = X1 + eps. Model = X1.
    X2 ⊥ Y | X1 and model = Bayes-optimal, so CFI(X2) = 0.
    """

    def test_cfi_unused_feature_zero(self):
        """CFI(X1) = 0 when model doesn't use X1."""
        rng = np.random.RandomState(42)
        n = 5000
        sigma = np.array([
            [1.0, 0.7, 0.0],
            [0.7, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        L = np.linalg.cholesky(sigma)
        Z = rng.randn(n, 3)
        data = Z @ L.T
        X = pd.DataFrame(data, columns=["x1", "x2", "x3"])
        y = X["x2"] + X["x3"] + rng.randn(n) * 0.1

        predict = lambda x: x["x2"].values + x["x3"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        result = explainer.cfi(X, pd.Series(y.values, name="y"), n_repeats=10)
        imp = result.importance()

        assert abs(imp.loc["x1", "importance"]) < 0.2
        assert imp.loc["x2", "importance"] > 0.3
        assert imp.loc["x3", "importance"] > 0.3

    def test_cfi_conditional_independence_bayes_optimal(self):
        """CFI(X2) = 0 when model = f*(X) = X1 and X2 ⊥ Y | X1."""
        rng = np.random.RandomState(42)
        n = 5000
        rho = 0.9
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        result = explainer.cfi(X, pd.Series(y, name="y"), n_repeats=10)
        imp = result.importance()

        # CFI(X2) should be ~0 (conditional independence + Bayes optimal)
        assert abs(imp.loc["x2", "importance"]) < 0.1
        # CFI(X1) should be positive
        assert imp.loc["x1", "importance"] > 0.1


# ---------------------------------------------------------------------------
# 2.4 CFI: exact values for Gaussian data
# ---------------------------------------------------------------------------

class TestCFIExactGaussian:
    """(X1, X2) ~ N(0, Sigma), y = X1 + X2 + eps, model = X1 + X2.
    CFI(X1) = 2*(1 - rho^2).
    """

    @pytest.mark.parametrize("rho,expected_cfi", [
        (0.0, 2.0),
        (0.5, 1.5),
        (0.9, 0.38),
    ])
    def test_cfi_matches_theory(self, rho, expected_cfi):
        rng = np.random.RandomState(42)
        n = 5000
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + x2 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values + x["x2"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        result = explainer.cfi(X, pd.Series(y, name="y"), n_repeats=10)
        imp = result.importance()

        np.testing.assert_allclose(
            imp.loc["x1", "importance"], expected_cfi, atol=0.3
        )

    def test_cfi_symmetry(self):
        """CFI(X1) = CFI(X2) when model is symmetric in X1, X2."""
        rng = np.random.RandomState(42)
        n = 5000
        rho = 0.5
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + x2 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values + x["x2"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        result = explainer.cfi(X, pd.Series(y, name="y"), n_repeats=10)
        imp = result.importance()

        np.testing.assert_allclose(
            imp.loc["x1", "importance"],
            imp.loc["x2", "importance"],
            atol=0.3,
        )


# ---------------------------------------------------------------------------
# 2.5 Marginalize-LOO: connection to explained variance and LOCO
# ---------------------------------------------------------------------------

class TestMarginalizeLOO:
    """(X1, X2) ~ N(0, Sigma), y = X1 + X2 + eps, model = X1 + X2.
    Marginalize-LOO (conditional) for X1 = Var(X1|X2) = 1 - rho^2.
    Should also ≈ LOCO with linear learner for Bayes-optimal model.
    """

    @pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
    def test_marginalize_loo_equals_conditional_variance(self, rho):
        rng = np.random.RandomState(42)
        n = 5000
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + x2 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values + x["x2"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        result = explainer.loo(
            X, pd.Series(y, name="y"), "marginalize",
            distribution="conditional", n_samples=50,
        )
        imp = result.importance()

        expected = 1 - rho ** 2
        np.testing.assert_allclose(
            imp.loc["x1", "importance"], expected, atol=0.2
        )

    def test_marginalize_loo_approx_loco(self):
        """Marginalize-LOO (conditional) ≈ LOCO for Bayes-optimal + squared error."""
        rng = np.random.RandomState(42)
        n = 5000
        rho = 0.7
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + x2 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values + x["x2"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)
        explainer.set_y_train(pd.Series(y, name="y"))

        marg_result = explainer.loo(
            X, pd.Series(y, name="y"), "marginalize",
            distribution="conditional", n_samples=50,
        )
        loco_result = explainer.loco(
            X, pd.Series(y, name="y"), learner=LinearRegression(),
        )

        marg_imp = marg_result.importance()
        loco_imp = loco_result.importance()

        np.testing.assert_allclose(
            marg_imp.loc["x1", "importance"],
            loco_imp.loc["x1", "importance"],
            atol=0.3,
        )


# ---------------------------------------------------------------------------
# 2.6 Conditional independence implies zero marginalize-LOO
# ---------------------------------------------------------------------------

class TestMarginalizeLOOConditionalIndependence:
    """(X1, X2) correlated, y = X1 + eps, model = X1.
    X2 ⊥ Y | X1, so marginalize-LOO(X2, conditional) = 0.
    """

    def test_marginalize_loo_x2_zero(self):
        rng = np.random.RandomState(42)
        n = 5000
        rho = 0.9
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        result = explainer.loo(
            X, pd.Series(y, name="y"), "marginalize",
            distribution="conditional", n_samples=50,
        )
        imp = result.importance()
        assert abs(imp.loc["x2", "importance"]) < 0.1

    def test_marginalize_loo_x1_positive(self):
        """X1 is important — removing it loses information."""
        rng = np.random.RandomState(42)
        n = 5000
        rho = 0.9
        z1, z2 = rng.randn(n), rng.randn(n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = x1 + rng.randn(n) * 0.1

        predict = lambda x: x["x1"].values
        sampler = GaussianSampler(X)
        explainer = Explainer(predict, X, loss=squared_error, sampler=sampler)

        result = explainer.loo(
            X, pd.Series(y, name="y"), "marginalize",
            distribution="conditional", n_samples=50,
        )
        imp = result.importance()
        assert imp.loc["x1", "importance"] > 0.1
