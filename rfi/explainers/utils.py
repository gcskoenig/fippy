

def detect_conv(scores, ii, threshold):
    """Detect convergence for SAGE values for each run separately (cf nr_runs argument)
    when 'largest sd is sufficiently low proportion of range of estimated values' (Covert
    et al., 2020; p.6)
    Variance estimated using Welford's algorithm with K = current mean
    (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance).
    Convergence has to be detected for each run and each feature for the algorithm to break

    Args:
        scores: Dataframe of scores as in explainer l. 725
        ii: current ordering in SAGE estimation    # TODO (cl) detect ii within function?!
        threshold: Threshold for convergence detection
        """
    nr_runs = len(scores.groupby(level=1))
    if ii == 0:
        # the first ordering leads to a single SAGE value, which is not sufficient to detect convergence
        return False
    else:
        """Detect convergence up to the current ordering for each run"""
        # initiate vector of booleans for convergence detection of each run
        converged_runs = []
        for i in range(nr_runs):
            # retrieve scores for run i (averages for each ordering)
            current_scores = scores.loc[(slice(0, ii), slice(i), slice(None))].mean(level=0)
            # mean of all current scores of the current run across all orderings
            # mean = current_scores.mean()
            # difference between the current scores and their averages
            diffs = current_scores - current_scores.mean()
            # squared differences
            diffs2 = diffs*diffs
            # sum of squared diffs
            diffs2_sum = diffs2.sum()
            # sum of diffs
            diffs_sum = diffs.sum()
            # diffs_sum2 = (diffs_sum * diffs_sum)
            # diffs_sum2_n = (diffs_sum2/ii)
            variance = (diffs2_sum - ((diffs_sum * diffs_sum)/ii)) / (ii - 1)
            # ratio
            ratio = (variance ** 0.5) / (current_scores.max() - current_scores.min())
            # max ratio (since convergence has to be detected for every feature)
            max_ratio = ratio.max()
            if max_ratio < threshold:
                converged_runs.append(True)
            else:
                converged_runs.append(False)
        if sum(converged_runs) == len(converged_runs):
            # convergence across all runs has been detected
            return True
        else:
            return False
