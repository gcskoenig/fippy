"""Test suite 1: Ewald et al. (2024) illustrative example.

Reference: Ewald, Bothmann, Leeb, Bischl, Casalicchio (2024).
"A Guide to Feature Importance Methods for Scientific Inference."
arXiv:2404.12862, Section 8.

DGP:
  X1, X3, X5 ~ N(0,1) iid
  X2 = X1 + eps2,  eps2 ~ N(0, 0.001)
  X4 = X3 + eps4,  eps4 ~ N(0, 0.1)
  Y  = X4 + X5 + X4*X5 + epsY,  epsY ~ N(0, 0.1)

Feature-target associations:
  X1, X2: not associated with Y (neither A1 nor A2a)
  X3: unconditionally associated (A1) via X4, but NOT conditionally (A2a)
  X4, X5: both A1 and A2a
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from fippy import Explainer
from fippy.losses import squared_error
from fippy.samplers import GaussianSampler

from conftest import make_ewald_dgp


@pytest.fixture(scope="module")
def ewald_data():
    X, y = make_ewald_dgp(n=3000, seed=42)
    return X, y


@pytest.fixture(scope="module")
def ewald_lm(ewald_data):
    X, y = ewald_data
    model = LinearRegression().fit(X, y)
    return model.predict


@pytest.fixture(scope="module")
def ewald_rf(ewald_data):
    X, y = ewald_data
    model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
    return model.predict


@pytest.fixture(scope="module")
def ewald_sampler(ewald_data):
    X, _ = ewald_data
    return GaussianSampler(X)


# ---------------------------------------------------------------------------
# PFI tests (loo + resample + marginal)
# ---------------------------------------------------------------------------

class TestEwaldPFI:

    def test_pfi_lm_x1_x2_near_zero(self, ewald_data, ewald_lm):
        """X1 and X2 are neither used by the model nor associated with Y."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["X1", "importance"]) < 0.5
        assert abs(imp.loc["X2", "importance"]) < 0.5

    def test_pfi_lm_x4_x5_positive(self, ewald_data, ewald_lm):
        """X4 and X5 are unconditionally associated with Y and used by LM."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()
        assert imp.loc["X4", "importance"] > 0.5
        assert imp.loc["X5", "importance"] > 0.5

    def test_pfi_rf_x4_x5_positive(self, ewald_data, ewald_rf):
        """RF should also detect X4 and X5 as important."""
        X, y = ewald_data
        explainer = Explainer(ewald_rf, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()
        assert imp.loc["X4", "importance"] > 0.5
        assert imp.loc["X5", "importance"] > 0.5

    def test_pfi_rf_x1_x2_near_zero(self, ewald_data, ewald_rf):
        X, y = ewald_data
        explainer = Explainer(ewald_rf, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["X1", "importance"]) < 0.5
        assert abs(imp.loc["X2", "importance"]) < 0.5

    def test_pfi_significance_x4_x5(self, ewald_data, ewald_lm):
        """t-test should reject H0 for X4, X5."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        result = explainer.pfi(X, y, n_repeats=10)
        test_df = result.test(method="t", alternative="greater")
        assert test_df.loc["X4", "p_value"] < 0.05
        assert test_df.loc["X5", "p_value"] < 0.05



# ---------------------------------------------------------------------------
# CFI tests (loo + resample + conditional)
# ---------------------------------------------------------------------------

class TestEwaldCFI:

    def test_cfi_x1_x2_near_zero(self, ewald_data, ewald_lm, ewald_sampler):
        """X1, X2 have no conditional association with Y."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error, sampler=ewald_sampler)
        result = explainer.cfi(X, y, n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["X1", "importance"]) < 0.5
        assert abs(imp.loc["X2", "importance"]) < 0.5

    def test_cfi_x3_near_zero(self, ewald_data, ewald_lm, ewald_sampler):
        """X3 is screened off by X4 — conditionally independent of Y given X_{-3}."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error, sampler=ewald_sampler)
        result = explainer.cfi(X, y, n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["X3", "importance"]) < 0.1

    def test_cfi_x4_x5_positive(self, ewald_data, ewald_lm, ewald_sampler):
        """X4 and X5 are conditionally associated with Y (A2a)."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error, sampler=ewald_sampler)
        result = explainer.cfi(X, y, n_repeats=10)
        imp = result.importance()
        assert imp.loc["X4", "importance"] > 0.1
        assert imp.loc["X5", "importance"] > 0.1

    def test_cfi_significance_x3_not_rejected(self, ewald_data, ewald_lm, ewald_sampler):
        """t-test should NOT reject for X3 (conditional independence)."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error, sampler=ewald_sampler)
        result = explainer.cfi(X, y, n_repeats=10)
        test_df = result.test(method="t", alternative="greater")
        assert test_df.loc["X3", "p_value"] > 0.05

    def test_cfi_significance_x4_x5_rejected(self, ewald_data, ewald_lm, ewald_sampler):
        """t-test should reject for X4, X5."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error, sampler=ewald_sampler)
        result = explainer.cfi(X, y, n_repeats=10)
        test_df = result.test(method="t", alternative="greater")
        assert test_df.loc["X4", "p_value"] < 0.05
        assert test_df.loc["X5", "p_value"] < 0.05

    def test_cfi_rf_x3_near_zero(self, ewald_data, ewald_rf, ewald_sampler):
        """Even RF should give CFI(X3) ≈ 0 since X4 screens off X3."""
        X, y = ewald_data
        explainer = Explainer(ewald_rf, X, loss=squared_error, sampler=ewald_sampler)
        result = explainer.cfi(X, y, n_repeats=10)
        imp = result.importance()
        assert abs(imp.loc["X3", "importance"]) < 1.0


# ---------------------------------------------------------------------------
# LOCO tests (loo + refit)
# ---------------------------------------------------------------------------

class TestEwaldLOCO:

    def test_loco_x1_x2_near_zero(self, ewald_data, ewald_lm):
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        explainer.set_y_train(y)
        result = explainer.loco(X, y, learner=LinearRegression())
        imp = result.importance()
        assert abs(imp.loc["X1", "importance"]) < 0.5
        assert abs(imp.loc["X2", "importance"]) < 0.5

    def test_loco_x4_x5_positive(self, ewald_data, ewald_lm):
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        explainer.set_y_train(y)
        result = explainer.loco(X, y, learner=LinearRegression())
        imp = result.importance()
        assert imp.loc["X4", "importance"] > 0.1
        assert imp.loc["X5", "importance"] > 0.1

    def test_loco_x3_small_lm(self, ewald_data, ewald_lm):
        """For LM, LOCO(X3) should be small since X4 captures X3's info."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        explainer.set_y_train(y)
        result = explainer.loco(X, y, learner=LinearRegression())
        imp = result.importance()
        assert imp.loc["X3", "importance"] < imp.loc["X4", "importance"]


# ---------------------------------------------------------------------------
# SAGE tests (shapley + marginalize)
# ---------------------------------------------------------------------------

class TestEwaldSAGE:

    def test_sage_x1_x2_near_zero(self, ewald_data, ewald_lm):
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        result = explainer.sage(X, y, distribution="marginal",
                                n_samples=20, n_permutations=50)
        imp = result.importance()
        assert abs(imp.loc["X1", "importance"]) < 0.5
        assert abs(imp.loc["X2", "importance"]) < 0.5

    def test_sage_x4_x5_largest(self, ewald_data, ewald_lm):
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        result = explainer.sage(X, y, distribution="marginal",
                                n_samples=20, n_permutations=50)
        imp = result.importance()
        top2 = imp["importance"].nlargest(2).index.tolist()
        assert "X4" in top2
        assert "X5" in top2

    def test_sage_efficiency(self, ewald_data, ewald_lm):
        """Sum of SAGE values ≈ v(D) = loss(empty) - loss(full)."""
        X, y = ewald_data
        explainer = Explainer(ewald_lm, X, loss=squared_error)
        result = explainer.sage(X, y, distribution="marginal",
                                n_samples=50, n_permutations=100)
        sage_sum = result.importance()["importance"].sum()

        y_arr = y.values
        loss_empty = explainer._value_fn(
            X, y_arr, [], "marginalize", "marginal", 50, None
        )
        loss_full = explainer._value_fn(
            X, y_arr, list(X.columns), "marginalize", "marginal", 50, None
        )
        total_reduction = loss_empty.mean() - loss_full.mean()
        np.testing.assert_allclose(sage_sum, total_reduction, atol=2.0)
