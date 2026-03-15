"""Shared fixtures and helpers for fippy test suites."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from fippy import Explainer
from fippy.losses import squared_error
from fippy.samplers import GaussianSampler, PermutationSampler


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------

def make_ewald_dgp(n=2000, seed=42):
    """Ewald et al. (2024) illustrative DGP (Section 8).

    X1, X3, X5 ~ N(0,1) iid
    X2 = X1 + eps2,  eps2 ~ N(0, 0.001)
    X4 = X3 + eps4,  eps4 ~ N(0, 0.1)
    Y  = X4 + X5 + X4*X5 + epsY,  epsY ~ N(0, 0.1)
    """
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n)
    X3 = rng.randn(n)
    X5 = rng.randn(n)
    X2 = X1 + rng.randn(n) * np.sqrt(0.001)
    X4 = X3 + rng.randn(n) * np.sqrt(0.1)
    Y = X4 + X5 + X4 * X5 + rng.randn(n) * np.sqrt(0.1)
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5})
    return X, pd.Series(Y, name="Y")


def make_independent_linear(n=2000, seed=42, beta=None, noise_std=0.1):
    """y = beta[0]*x1 + beta[1]*x2 + ... + noise.  Independent N(0,1) features."""
    if beta is None:
        beta = [1.0, 1.0]
    p = len(beta)
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {f"x{i+1}": rng.randn(n) for i in range(p)}
    )
    y = sum(b * X[f"x{i+1}"] for i, b in enumerate(beta)) + rng.randn(n) * noise_std
    return X, pd.Series(y.values, name="y")


def make_correlated_gaussian(n=2000, rho=0.9, seed=42):
    """(X1, X2) ~ N(0, Sigma) with Cor(X1,X2) = rho. y = X1 + noise."""
    rng = np.random.RandomState(seed)
    z1 = rng.randn(n)
    z2 = rng.randn(n)
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    y = x1 + rng.randn(n) * 0.1
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y, name="y")


def make_bivariate_gaussian(n=2000, rho=0.5, beta1=1.0, beta2=1.0,
                            noise_std=0.1, seed=42):
    """(X1, X2) ~ N(0, Sigma), y = beta1*X1 + beta2*X2 + noise."""
    rng = np.random.RandomState(seed)
    z1 = rng.randn(n)
    z2 = rng.randn(n)
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    y = beta1 * x1 + beta2 * x2 + rng.randn(n) * noise_std
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y, name="y")


def make_trivariate_gaussian(n=2000, sigma=None, seed=42):
    """(X1, X2, X3) ~ N(0, Sigma). y = X2 + X3 + noise."""
    rng = np.random.RandomState(seed)
    if sigma is None:
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
    return X, pd.Series(y.values, name="y")
