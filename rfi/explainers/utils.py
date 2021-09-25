import numpy as np
import pandas as pd


# TODO so far only works across orderings (i.e. for a sample wise loss, but not for individual losses)
# TODO Put most of detect method in update() mehtoed not to calculate the variance from scratch each time
# TODO detect() should be everything after line 64
class ConvergenceDetection:
    # TODO update docstring (and use other reference for Welford's algorithm if any)
    """Track sage values and detect convergence for each run separately (cf nr_runs argument)
     using Welford's algorithm with K = current mean
     (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance).
     Convergence has to be detected for each run and each feature for the algorithm to break.

     Args:
         scores: Dataframe of scores as updated in ai_vals
         ii: current ordering in SAGE estimation    # TODO detect ii within class?!
         threshold: Threshold for convergence detection
    """
    def __init__(self, scores, ii, threshold):
        # copy scores
        self.scores = scores
        # nr_runs
        self.nr_runs = len(scores.groupby(level=1))
        # nr of orderings
        self.nr_orderings = len(scores.groupby(level=0))
        # current ordering (which is also the sample size for the sage values convergence is checked for)
        self.current_ordering = ii
        self.thresh = threshold


    def detect(self):
        if self.current_ordering == 0:
            # the first ordering leads to a single SAGE value, which is not sufficient to detect convergence
            return False
        else:
            """Detect convergence up to the current ordering for each run"""
            # initiate vector of booleans for convergence detection of each run
            converged_runs = []
            for i in range(self.nr_runs):
                # retrieve scores for run i (averages for each ordering)
                current_scores = self.scores.loc[(slice(0, self.current_ordering), slice(i), slice(None))].mean(level=0)
                # mean of all current scores of the current run across all orderings
                mean = current_scores.mean()
                # difference between the current scores and their averages
                diffs = current_scores - mean
                # squared differences
                diffs2 = diffs*diffs
                diffs2_sum = diffs2.sum()
                # sum of squared
                # diffs.
                diffs_sum = diffs.sum()
                # diffs_sum2 = (diffs_sum * diffs_sum)
                # diffs_sum2_n = (diffs_sum2/self.nr_orderings)
                variance = (diffs2_sum - ((diffs_sum * diffs_sum)/self.nr_orderings)) / (self.nr_orderings - 1)
                # sigma = (variance ** 0.5)
                # cf. Covert et al. (2020), p. 6
                # get all max scores
                # max_scores = current_scores.max()
                # get all min scores
                # min_scores = current_scores.min()
                # gap
                # gap = max_scores - min_scores
                # ratio
                ratio = (variance ** 0.5) / (current_scores.max() - current_scores.min())
                # max ratio (since convergence has to be detected for every feature)
                max_ratio = ratio.max()
                if max_ratio < self.thresh:
                    converged_runs.append(True)
                else:
                    converged_runs.append(False)

            if sum(converged_runs) == len(converged_runs):
                # convergence across all runs has been detected
                return True, self.current_ordering
            else:
                return False
