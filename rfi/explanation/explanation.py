"""Explanations are the output of Explainers.

Aggregated or obser-wise wise results can be
accessed. Plotting functionality is available.
"""


class Explanation():
    """Stores and provides access to results from Explainer.

    Aggregated as well as observation-wise results are stored.
    Plotting functionality is available.

    Attributes:
        fsoi: Features of interest.
        fsoi_names: FSOI names
        lss: losses on perturbed (# fsoi, # runs, # observations)
    """

    def __init__(self, fsoi, lss, fsoi_names=None):
        """Inits Explainer with model, mask and potentially sampler and loss"""
        self.fsoi = fsoi # TODO evaluate, do I need to make a copy?
        self.lss = lss # TODO evaluate, do I need to make a copy?
        self.fsoi_names = fsoi_names
        if self.fsoi_names is None:
            self.fsoi_names = fsoi

    def mean_rfis(self):
        """Computes Relative Feature importance

        # TODO(gcsk): allow handing a sample as argument
        #             (without needing sampler)

        Args:
            X_test: data to use for sampler training and evaluation.
            y_test: labels for evaluation.
            G: relative feature set
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.

        Returns:
            A np.array with the relative feature importance values for
            features of interest.
        """
        return np.mean(np.mean(lss, axis=2), axis=1)