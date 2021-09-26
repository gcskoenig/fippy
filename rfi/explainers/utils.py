# TODO so far only works across orderings (i.e. for a sample wise loss, but not for individual losses)

def detect_conv(scores, ii, threshold):
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
            mean = current_scores.mean()
            # difference between the current scores and their averages
            diffs = current_scores - mean
            # squared differences
            diffs2 = diffs*diffs
            # sum of squared diffs
            diffs2_sum = diffs2.sum()
            # sum of diffs
            diffs_sum = diffs.sum()
            # diffs_sum2 = (diffs_sum * diffs_sum)
            # diffs_sum2_n = (diffs_sum2/self.nr_orderings)
            variance = (diffs2_sum - ((diffs_sum * diffs_sum)/ii)) / (ii - 1)
            # ratio
            ratio = (variance ** 0.5) / (current_scores.max() - current_scores.min())
            # max ratio (since convergence has to be detected for every feature)
            max_ratio = ratio.max()
            print(max_ratio)
            if max_ratio < threshold:
                converged_runs.append(True)
            else:
                converged_runs.append(False)
        if sum(converged_runs) == len(converged_runs):
            # convergence across all runs has been detected
            return True, ii
        else:
            return False
