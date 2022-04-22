import numpy as np


def detect_conv(scores, ii, threshold, extra_orderings=0):  # TODO (cl) make class?
    """Detect convergence for SAGE values up to the current ordering (avg over all runs)
        when 'largest sd is sufficiently low proportion of range of estimated values' (Covert
        et al., 2020; p.6)

    Args:
        scores: Dataframe of scores as in explainer l. 725
        ii: current ordering in SAGE estimation
        threshold: Threshold for convergence detection
        extra_orderings: orderings after convergence has been detected, default: 0
        """
    if ii == 0 or ii == 1:
        # the first two orderings are not sufficient to detect convergence
        return False
    else:
        # input scores are nr_runs runs per ordering, mean makes it one value per ordering
        scores = scores.loc[(slice(0, ii), slice(None), slice(None))].groupby('ordering').mean()

        # TODO (cl) Use Welford's algorithm when making class and continuously update
        # diffs = scores - scores.mean()
        # diffs2 = diffs * diffs
        # diffs2_sum = diffs2.sum()
        # diffs_sum = diffs.sum()
        # variance = (diffs2_sum - ((diffs_sum * diffs_sum) / ii)) / (ii - 1)
        # ratio = ((variance ** 0.5) / np.sqrt(ii)) / (scores.max() - scores.min())   # TODO (cl) correct denominator?
        # max_ratio = ratio.max()

        variance = np.var(scores)*(ii/(ii-1))
        ratio = ((variance ** 0.5) / np.sqrt(ii)) / (scores.max() - scores.min())   # TODO (cl) correct denominator?
        max_ratio = ratio.max()

        if max_ratio < threshold:
            if extra_orderings == 0:
                # stop when convergence detected
                return True
            else:
                # extra runs to verify flat curve after convergence has been detected
                extra_orderings -= 1
                return int(extra_orderings)
        else:
            # convergence not yet detected
            return False
