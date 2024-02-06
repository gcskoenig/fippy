#  Explainer utilities

import numpy as np


def detect_conv(scores, ii, threshold, extra_orderings=0, conv_detected=False):  # TODO (cl) make class?
    """Detect convergence for SAGE values up to the current ordering (avg over all runs)
        when 'largest sd is sufficiently low proportion of range of estimated values' (Covert
        et al., 2020; p.6)

    Args:
        scores: Dataframe of scores as in explainer l. 725
        ii: current ordering in SAGE estimation
        threshold: Threshold for convergence detection
        extra_orderings: orderings after convergence has been detected, default: 0
        conv_detected: has convergence been detected before? If True -> only count down extra_orderings
    Returns:
        Tuple (Bool, int): Convergence detected? True/False, number of extra_orderings left
    """
    # if convergence has been detected once (or generally if conv_detected == True), only count down extra_orderings
    if conv_detected:
        extra_orderings -= 1
        return True, extra_orderings
    if ii < 2:
        # the first two orderings are not sufficient to detect convergence
        return False, extra_orderings
    else:
        # input scores are nr_runs runs per ordering, mean makes it one value per ordering
        scores = scores.loc[(slice(0, ii), slice(None), slice(None))].groupby('ordering').mean()

        # TODO (cl) Use Welford's algorithm when making class and continuously update
        # diffs = scores - scores.mean()
        # diffs2 = diffs * diffs
        # diffs2_sum = diffs2.sum()
        # diffs_sum = diffs.sum()
        # variance = (diffs2_sum - ((diffs_sum * diffs_sum) / ii)) / (ii - 1)
        # ratio = ((variance ** 0.5) / np.sqrt(ii)) / (scores.max() - scores.min())
        # max_ratio = ratio.max()

        variance = np.var(scores, axis=0)*(ii/(ii-1))
        ratio = ((variance ** 0.5) / np.sqrt(ii)) / (scores.max() - scores.min())
        max_ratio = ratio.max()

        if max_ratio < threshold:
            if extra_orderings == 0:
                # stop when convergence detected
                return True, 0
            else:
                # reduce extra runs to verify flat curve after convergence has been detected
                extra_orderings -= 1
                return True, extra_orderings
        else:
            # convergence not yet detected
            return False, extra_orderings
