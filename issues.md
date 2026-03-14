# Open Design Issues

## 1. `obs_importance()` and `test()` for multi-fold CV

**Status:** Deferred — currently raises `NotImplementedError` when `n_folds > 1`.

**Problem:** The 4D scores tensor `(n_folds, n_repeats, n_obs, n_features)` pads by position within each fold. Position `i` in fold 1 is a different observation than position `i` in fold 2. Naive `nanmean` over the folds axis at each position averages unrelated observations.

**Affected methods:**
- `obs_importance()` — should return per-observation scores in original data order
- `test()` — needs all per-observation scores concatenated across folds for the hypothesis test

**Not affected:**
- `importance()` and `ci()` — these average over observations first, so positional mixing doesn't matter

**Possible solutions:**
1. Store `fold_indices: list[np.ndarray]` in `ExplanationResult` (test set indices per fold). Use these to place/concatenate scores by original observation index.
2. Store scores as a list of arrays per fold instead of a single padded tensor. Simpler but breaks the uniform tensor shape.
3. Concatenate across folds for `test()` (just needs a bag of per-obs scores). For `obs_importance()`, reconstruct original order using fold indices.

**Decision needed:** Which approach? Option 1 seems cleanest — keeps the tensor format, adds minimal metadata.
