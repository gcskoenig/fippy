# Research: fippy (Python) and xplainfi (R) Codebases

## Overview

Both packages implement model-agnostic feature importance methods with statistical inference. They share the same theoretical foundations (Ewald et al. 2024, König et al. 2021) and implement overlapping sets of methods, but differ significantly in architecture, scope, and implementation details.

- **fippy** (Python): Feature Importance in Python. Centered around conditional sampling and Shapley-based attribution. Built on pandas/numpy/PyTorch.
- **xplainfi** (R, `mlr-org/xplainfi`): Unified, extensible interface for feature importance. Built on the mlr3 ecosystem. Authored by Lukas Burk.

---

## Feature Importance Methods

| Method | fippy | xplainfi | Notes |
|--------|-------|----------|-------|
| PFI (Permutation Feature Importance) | `explainer.pfi()` | `PFI` class | Both shuffle features marginally |
| CFI (Conditional Feature Importance) | `explainer.cfi()` | `CFI` class | Both sample from P(X_j \| X_{-j}) |
| RFI (Relative Feature Importance) | `explainer.rfi(G)` | `RFI` class | Arbitrary conditioning set G |
| SAGE (marginal) | `explainer.msage()` | `MarginalSAGE` class | Shapley values with marginal sampling |
| SAGE (conditional) | `explainer.csage()` | `ConditionalSAGE` class | Shapley values with conditional sampling |
| LOCO | Not implemented | `LOCO` class | Refit-based leave-one-covariate-out |
| WVIM | Not implemented | `WVIM` class | Generalized refit-based importance |
| Decomposition | `explainer.decomposition()` | Not implemented | Decomposes FI into per-feature contributions |
| DI/AI primitives | `explainer.di_from()`, `explainer.ai_via()` | Not exposed directly | Direct/Associative importance building blocks |
| Learner FI | `LearnerExplainer` | Not separate (built into resampling) | Multi-refit importance |

### Key Definitions

- **PFI**: `PFI_j = R(X_shuffled_j) - R(X)` — performance drop when feature j is permuted marginally.
- **CFI**: `CFI_j = R(X_j_cond | X_{-j}) - R(X)` — performance drop when feature j is resampled conditionally on all other features.
- **RFI**: `RFI_j(G) = DI(j from D\{j}) - DI(j from G)` — importance relative to feature set G.
- **SAGE**: `SAGE_j = E[v(S ∪ {j}) - v(S)]` averaged over orderings — Shapley value attribution of total model performance.

---

## Samplers / Conditional Sampling

Both packages recognize that the sampling strategy is critical for conditional feature importance.

| Sampler | fippy | xplainfi | Notes |
|---------|-------|----------|-------|
| Marginal/Permutation | Built into `Sampler` base (empty G case) | `MarginalPermutationSampler` | Shuffles feature values |
| Marginal Reference | Not separate | `MarginalReferenceSampler` | Samples rows from reference data |
| Simple/Lookup | `SimpleSampler` | — | Join on categorical conditioning values |
| Gaussian | `GaussianSampler` | `ConditionalGaussianSampler` | Both use standard MVN conditioning formulas |
| Random Forest (univariate) | `UnivRFSampler`, `ContUnivRFSampler` | — | RF classifier/regressor + residual sampling |
| MDN | `MDNSampler` | — | Mixture Density Networks (PyTorch) |
| Normalizing Flow | `CNFSampler` | — | Conditional Normalizing Flows (PyTorch) |
| ARF | — | `ConditionalARFSampler` | Adversarial Random Forests (default in xplainfi) |
| Conditional Inference Tree | — | `ConditionalCtreeSampler` | Uses partykit::ctree |
| kNN | — | `ConditionalKNNSampler` | k-nearest neighbors sampling |
| Knockoff | — | `KnockoffSampler`, `KnockoffGaussianSampler` | Model-X knockoffs |
| Sequential (multivariate) | `SequentialSampler` | — | Chains univariate samplers respecting DAG ordering |

### Gaussian Conditional Sampling (shared approach)
Both compute: `P(X_J | X_G) ~ N(μ_J + Σ_JG Σ_GG^{-1} (x_G - μ_G), Σ_JJ - Σ_JG Σ_GG^{-1} Σ_GJ)`

### fippy-specific: Sequential Sampler
The `SequentialSampler` is a key architectural difference. It chains univariate samplers (one per feature) in topological order (optionally respecting a DAG), building up the joint conditional distribution one variable at a time. This allows mixing sampler types (e.g., RF for categorical, Gaussian for continuous).

### xplainfi-specific: ARF Sampler
xplainfi defaults to Adversarial Random Forests for conditional sampling, which can handle mixed data types without explicit type-specific handling.

---

## Statistical Inference

| Method | fippy | xplainfi | Notes |
|--------|-------|----------|-------|
| Confidence intervals (t-based) | `ModelExplanation.cis()` | `importance(ci_method="raw")` | Standard t-distribution CIs |
| Nadeau-Bengio corrected t-test | — | `importance(ci_method="nadeau_bengio")` | Corrects for resampling overlap |
| Quantile CIs | — | `importance(ci_method="quantile")` | Non-parametric empirical quantiles |
| CPI (Conditional Predictive Impact) | — | `importance(ci_method="cpi")` | Observation-wise loss difference tests |
| cARFi | — | Supported via CPI + ARF | Blesch et al. 2025 |
| Lei et al. inference | — | `importance(ci_method="lei")` | Distribution-free for LOCO |
| Learner correction | `LearnerExplanation.cis()` adds `c = n_test/n_train` | — | Corrects for train/test overlap |
| Multiple testing correction | — | `p_adjust` parameter | Bonferroni, Holm, BH, etc. |
| Convergence detection (SAGE) | `detect_conv()` in explainers/utils.py | Built into SAGE class | Both monitor relative SE |

### fippy CI details
- Two-sided: Molnar et al. (2023) method using t-statistic
- One-sided: Watson et al. (2019) method
- LearnerExplanation adds correction term `c = n_test/n_train` to account for train/test dependency

### xplainfi inference details
- Supports observation-wise hypothesis tests: t-test, Wilcoxon, Fisher permutation, binomial sign test
- Fisher test uses Phipson-Smyth correction: `p = (count + 1) / (B + 1)`
- CPI framework tests `H0: L(Y, f(X_tilde)) - L(Y, f(X)) <= 0` per feature

---

## Architecture Comparison

### fippy Architecture

```
Explainer (main interface)
├── predict function (model or callable)
├── Sampler (strategy pattern)
│   ├── SimpleSampler
│   ├── GaussianSampler
│   ├── MDNSampler / CNFSampler
│   ├── UnivRFSampler / ContUnivRFSampler
│   └── SequentialSampler (chains univariate samplers)
├── Loss function
└── Explanation objects (results)
    ├── ModelExplanation
    ├── LearnerExplanation
    └── DecompositionExplanation

LearnerExplainer (multi-refit variant)
CGExplainer (DAG-aware, d-separation checks)

Backend:
├── ConditionalDistributionEstimator (PyTorch base)
│   └── GaussianConditionalEstimator
└── Data generation
    ├── DirectedAcyclicGraph
    └── StructuralEquationModel
```

- **Model-agnostic**: Takes any predict function
- **Sampler stores trained models**: `_trained_sampling_funcs` dict keyed by (J, G) pairs
- **Results stored as MultiIndex DataFrames**: (sample/fit/ordering, i) × features
- **Neural network backends**: MDN and normalizing flows via PyTorch
- **DAG support**: `CGExplainer` uses NetworkX for d-separation; `SequentialSampler` uses topological ordering

### xplainfi Architecture

```
FeatureImportanceMethod (R6 base)
├── PerturbationImportance
│   ├── PFI
│   ├── CFI
│   └── RFI
├── SAGE
│   ├── MarginalSAGE
│   └── ConditionalSAGE
└── WVIM
    └── LOCO

FeatureSampler (R6 base)
├── MarginalSampler
│   ├── MarginalPermutationSampler
│   └── MarginalReferenceSampler
├── ConditionalSampler
│   ├── ConditionalARFSampler
│   ├── ConditionalCtreeSampler
│   ├── ConditionalGaussianSampler
│   └── ConditionalKNNSampler
└── KnockoffSampler
    └── KnockoffGaussianSampler
```

- **Deep mlr3 integration**: Uses Tasks, Learners, Measures, Resamplings
- **Separation of compute and aggregate**: `$compute()` stores raw scores, `$importance()` aggregates with CI methods
- **Parallelization abstraction**: `xplainfi_map()` wraps mirai, future, or sequential backends
- **Feature groups**: First-class support for grouped features throughout
- **Template method pattern**: SAGE base class defines pipeline, subclasses implement `expand_coalitions_data()`

---

## Key Differences

### 1. Scope
- **fippy** focuses on perturbation-based methods (PFI, CFI, RFI, SAGE) and decomposition. Has DI/AI primitives and causal graph support.
- **xplainfi** additionally includes refit-based methods (LOCO, WVIM) and a broader set of inference methods (CPI, Lei, Nadeau-Bengio).

### 2. Sampler Ecosystem
- **fippy** has deep learning-based samplers (MDN, normalizing flows) and the `SequentialSampler` for chaining univariate samplers along DAG structure.
- **xplainfi** has tree-based samplers (ARF, ctree), kNN, and knockoff-based samplers. ARF is the default and handles mixed types well.

### 3. Inference
- **xplainfi** has significantly more inference options: observation-wise tests (CPI, Lei), multiple testing correction, and resampling-corrected CIs. It supports formal hypothesis testing with p-values.
- **fippy** has basic t-based CIs with a learner correction term but lacks the observation-wise testing framework.

### 4. Integration
- **fippy** is framework-agnostic (takes any predict function).
- **xplainfi** is tightly integrated with mlr3 (Tasks, Learners, Measures).

### 5. Causal/Graph Support
- **fippy** has explicit DAG support via `CGExplainer` (d-separation checks), `SequentialSampler` (topological ordering), `DirectedAcyclicGraph`, and `StructuralEquationModel` classes.
- **xplainfi** does not have explicit causal graph support.

### 6. SAGE Implementation
- Both implement convergence detection via relative SE monitoring.
- **fippy** supports partial orderings (constraints on feature order) and returns orderings used.
- **xplainfi** uses checkpoint-based batch processing and batched prediction for memory management.

### 7. Decomposition
- **fippy** has a `decomposition()` method that breaks down total feature importance into contributions from individual other features, with `DecompositionExplanation` for visualization.
- **xplainfi** does not have this.

### 8. Data Generation
- **fippy** includes `DirectedAcyclicGraph` and `StructuralEquationModel` for generating synthetic data from DAGs.
- **xplainfi** includes several simulation DGPs (`sim_dgp_ewald`, `sim_dgp_correlated`, `sim_dgp_mediated`, `sim_dgp_confounded`, `sim_dgp_interactions`, `sim_dgp_independent`) for benchmarking and demonstration.

---

## Shared Theoretical Foundations

Both packages implement the framework from:
- **Ewald et al. (2024)**: Unifying feature importance framework
- **König et al. (2021)**: Relative Feature Importance
- **Covert et al. (2020)**: SAGE
- **Watson & Wright (2021)**: CPI / Conditional Predictive Impact
- **Molnar et al. (2023)**: Inference for PFI
- **Fisher et al. (2019)**: Model reliance / PFI

The core insight shared by both: feature importance depends critically on the **sampling mechanism** (marginal vs. conditional) and the **baseline/context** against which importance is measured. Both packages make these choices explicit and configurable.

---

## File Structure Reference

### fippy
```
src/fippy/
├── __init__.py
├── utils.py                          # Hash, key, partial ordering utilities
├── explainers/
│   ├── __init__.py
│   ├── explainer.py                  # Core: PFI, CFI, RFI, SAGE, decomposition (~1280 lines)
│   ├── learnerexplainer.py           # Multi-refit explainer
│   ├── cgexplainer.py                # DAG-aware explainer
│   └── utils.py                      # Convergence detection
├── explanation/
│   ├── __init__.py
│   ├── explanation.py                # Base Explanation class
│   ├── model.py                      # ModelExplanation (single model)
│   ├── learner.py                    # LearnerExplanation (multi-refit)
│   └── decomposition.py             # DecompositionExplanation
├── samplers/
│   ├── __init__.py
│   ├── sampler.py                    # Base Sampler class
│   ├── simple.py                     # SimpleSampler (categorical lookup)
│   ├── gaussian.py                   # GaussianSampler
│   ├── ensemble.py                   # UnivRFSampler, ContUnivRFSampler
│   ├── mdn.py                        # MDNSampler
│   ├── cnflow.py                     # CNFSampler (normalizing flows)
│   ├── sequential.py                 # SequentialSampler
│   └── _utils.py                     # Identity/permutation sampling helpers
├── backend/
│   ├── estimators/
│   │   ├── estimator.py              # Base ConditionalDistributionEstimator
│   │   └── gaussian/
│   │       └── gaussian_estimator.py # Gaussian conditional estimation
│   └── datagen/
│       ├── dags.py                   # DirectedAcyclicGraph
│       └── sem.py                    # StructuralEquationModel
└── plots/
    ├── _barplot.py                   # Horizontal bar plots
    ├── _barplot_multiple.py          # Grouped/wrapped bar plots
    ├── _snsstyle.py                  # Seaborn styling
    └── _utils.py                     # Plot utilities
```

### xplainfi (R)
```
R/
├── FeatureImportanceMethod.R         # Base class
├── PerturbationImportance.R          # Perturbation-based base
├── PFI.R                             # Permutation Feature Importance
├── CFI.R                             # Conditional Feature Importance
├── RFI.R                             # Relative Feature Importance
├── WVIM.R                            # Williamson's VIM (refit-based)
├── LOCO.R                            # Leave-One-Covariate-Out
├── SAGE.R                            # SAGE base class
├── MarginalSAGE.R                    # Marginal SAGE
├── ConditionalSAGE.R                 # Conditional SAGE
├── FeatureSampler.R                  # Sampler base class
├── MarginalSampler.R                 # Marginal sampler base
├── MarginalPermutationSampler.R      # Permutation sampler
├── MarginalReferenceSampler.R        # Reference data sampler
├── ConditionalSampler.R              # Conditional sampler base
├── ConditionalARFSampler.R           # Adversarial Random Forest sampler
├── ConditionalCtreeSampler.R         # Conditional inference tree sampler
├── ConditionalGaussianSampler.R      # Gaussian conditional sampler
├── ConditionalKNNSampler.R           # kNN sampler
├── KnockoffSampler.R                 # Knockoff base
├── KnockoffGaussianSampler.R         # Gaussian knockoff sampler
├── wvim_design_matrix.R              # Design matrix for WVIM
├── sim_dgp_*.R                       # Simulation DGPs (6 variants)
├── utils.R                           # Internal utilities
└── zzz.R                             # Package hooks
```
