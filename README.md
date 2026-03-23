# fippy: Feature Importance in Python 🐬

[![Tests](https://github.com/gcskoenig/fippy/actions/workflows/python-package.yml/badge.svg)](https://github.com/gcskoenig/fippy/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/fippy)](https://pypi.org/project/fippy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20--%203.13-blue)](https://pypi.org/project/fippy/)

A Python package for model-agnostic feature importance with statistical inference. Fippy implements a unified framework where feature importance methods are composed from three axes:

| Axis | Options | Description |
|---|---|---|
| **Attribution** | `loo`, `shapley` | Leave-one-out vs. Shapley value attribution |
| **Restriction** | `resample`, `marginalize`, `refit` | How the removed feature is handled |
| **Distribution** | `marginal`, `conditional` | Which distribution replacement values are drawn from |

This gives rise to well-known methods as special cases:

| Method | Attribution | Restriction | Distribution | Alias |
|---|---|---|---|---|
| PFI | loo | resample | marginal | `pfi()` |
| CFI | loo | resample | conditional | `cfi()` |
| RFI | loo | resample | conditional + G | `rfi()` |
| LOCO | loo | refit | — | `loco()` |
| SAGE | shapley | marginalize | marginal or conditional | `sage()` |

## Installation

Requires Python >= 3.11.

```bash
pip install fippy
```

For development, install in editable mode:

```bash
pip install -e path/to/fippy
```

## Quick start

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from fippy import Explainer
from fippy.samplers import GaussianSampler
from fippy.losses import squared_error

# Prepare data and model
X, y = ...  # your data as pd.DataFrame and array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Create explainer
explainer = Explainer(model.predict, X_train, loss=squared_error)

# Permutation Feature Importance (marginal)
result_pfi = explainer.pfi(X_test, y_test, n_repeats=10)
result_pfi.importance()
```

### Conditional methods

For conditional feature importance, pass a sampler:

```python
sampler = GaussianSampler(X_train)
explainer = Explainer(model.predict, X_train, loss=squared_error, sampler=sampler)

# Conditional Feature Importance
result_cfi = explainer.cfi(X_test, y_test, n_repeats=10)
result_cfi.importance()
```

### Refit-based methods

For LOCO, pass a learner (unfitted sklearn estimator) and set training labels:

```python
explainer.set_y_train(y_train)
result_loco = explainer.loco(X_test, y_test, learner=RandomForestRegressor())
result_loco.importance()
```

### SAGE (Shapley-based)

```python
result_sage = explainer.sage(X_test, y_test, distribution="marginal", n_samples=50)
result_sage.importance()
```

### Key parameters

**`restriction`** — How the dropped feature is handled:
- `"resample"`: Replace the feature with a single random draw from a reference distribution. Fast, but introduces sampling noise.
- `"marginalize"`: Replace the feature with `n_samples` random draws and average the predictions. Approximates the expected prediction over the reference distribution. More accurate than `resample` but slower.
- `"refit"`: Retrain the model without the feature entirely. No sampling involved; measures importance through model performance change.

**`distribution`** — Which reference distribution replacement values are drawn from (for `resample` and `marginalize`):
- `"marginal"`: Draw from the unconditional distribution P(X_j). Breaks dependence between the dropped feature and all others. Used by PFI.
- `"conditional"`: Draw from P(X_j | X_{-j}), respecting the dependence structure. Requires a `sampler`. Used by CFI/RFI.

Not applicable for `restriction="refit"`.

**`n_repeats`** — Number of times the importance computation is repeated with fresh perturbed samples for the dropped features. The variance across repeats captures Monte Carlo noise and is used by `ci()` for confidence intervals. Higher values give more stable estimates.

**`n_samples`** — Number of replacement samples drawn per observation when using `restriction="marginalize"`. The predictions are averaged over these samples to approximate the expectation. Required for `sage()` and `loo(..., restriction="marginalize")`. Not used for `resample` (which draws a single sample) or `refit`.

**`n_permutations`** — Maximum number of random feature orderings for Shapley value estimation (default: 500). Shapley values are approximated by averaging marginal contributions over random permutations. Computation stops early if the estimates converge (controlled by `convergence_threshold`).

**`n_jobs`** — Number of parallel threads for the feature loop in LOO methods (default: 1). Uses `joblib` with thread-based parallelism. Set to `-1` to use all cores. Not supported for Shapley.

### Using the generic interface

All convenience methods are shortcuts for `loo()` and `shapley()`:

```python
# Equivalent to explainer.pfi(X_test, y_test)
result = explainer.loo(X_test, y_test, "resample", distribution="marginal")
```

## Statistical inference

`ExplanationResult` provides built-in inference on the importance scores.

```python
# Mean importance with standard deviation
result.importance()

# Confidence intervals (t-based or quantile-based)
result.ci(alpha=0.05, method="t")

# Hypothesis test H0: importance_j <= 0
result.test(method="t", alternative="greater", p_adjust="bonferroni")

# Observation-wise importance scores
result.obs_importance()

# Relative importance (as fraction of baseline loss)
result.importance(relative=True)
```

Multiple testing corrections: `"bonferroni"`, `"holm"`, `"bh"` (Benjamini-Hochberg).

## Plotting

Visualize results with horizontal bar plots showing confidence intervals:

```python
# Single plot
result.hbarplot(figsize=(8, 4))

# Side-by-side comparison
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
result_pfi.hbarplot(ax=axes[0])
result_cfi.hbarplot(ax=axes[1])
fig.tight_layout()
```

See `Example.ipynb` for a complete walkthrough with plots.

## Feature groups

Features can be grouped and assessed jointly:

```python
# As a dict
explainer.pfi(X_test, y_test, features={"size": ["width", "height"], "color": ["r", "g", "b"]})

# As a list (each element becomes one group)
explainer.pfi(X_test, y_test, features=["width", "height", "color"])
```

## Samplers

| Sampler | Distribution | Description |
|---|---|---|
| `PermutationSampler` | marginal | Draws from training marginal (used automatically) |
| `GaussianSampler` | conditional | Multivariate Gaussian conditional P(X_J \| X_S) |

Additional samplers (regression-based, ARF, TabPFN) are planned.

## Serialization

```python
result.to_csv("importance.csv")
from fippy import ExplanationResult
loaded = ExplanationResult.from_csv("importance.csv")
```

## Status

The package is under active development. The core API is stable, but additional samplers and cross-validation features are planned.

## References

The package was previously called `rfi` and accompanies our paper on Relative Feature Importance: [[arXiv]](https://arxiv.org/abs/2007.08283)
