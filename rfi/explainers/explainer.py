"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""


class Explainer():
    """Uses Relative Feature Importance to compute the importance of features
    relative to any set of features G.

    Default samplers or loss function can be defined.
    Masks allows to specify for which features importance
    shall be computed.

    Attributes:
        model: Model or predict function.
        fsoi: Features of interest.
        X_train: Training data for Resampling.
        sampler: default sampler.
        loss: default loss.
    """

    def __init__(self, model, fsoi, X_train, sampler=None, loss=None):
        """Inits Explainer with model, mask and potentially sampler and loss"""
        self.model = model
        self.fsoi = fsoi
        self.X_train = X_train
        self.loss = loss

    def rfi(self, X_test, y_test, G, sampler=None, loss=None):
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
        if sampler is None:
            if self.sampler is None:
                # TODO(gcsk): raise Exception
                pass
            else:
                sampler = self.sampler
                pass

        if loss is None:
            if self.loss is None:
                # TODO(gcsk): raise Exception
                pass
            else:
                loss = self.loss
                pass

        # TODO(gcsk): check whether the sampler is trained on G
        # TODO(gcsk): if the sampler is not traine on G, train it
        # TODO(gcsk): if the sampler is trained

        # TODO(gcsk): sample the replacements

        # TODO(gcsk): RFI routine
