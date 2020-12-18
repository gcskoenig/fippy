"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""

import numpy as np
import rfi.explanation.explanation as explanation


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
        fs_names: list of strings with feature input_var_names
    """

    def __init__(self, model, fsoi, X_train, sampler=None, loss=None, fs_names=None):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        self.model = model
        self.fsoi = fsoi
        self.X_train = X_train
        self.loss = loss
        self.sampler = sampler
        self.fs_names = fs_names
        if self.fs_names is None:
            names = [ix_to_desc(jj) for jj in range(X_train.shape[0])]
            self.fs_names = names

    def rfi(self, X_test, y_test, G, sampler=None, loss=None, nr_runs=10, return_perturbed = False, verbose=False):
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
            nr_runs: how often the experiment shall be run
            verbose: whether printing in verbose mode or not.

        Returns:
            A np.array with the relative feature importance values for
            features of interest.
        """

        if sampler is None:
            if self.sampler is None:
                raise ValueError("Sampler has not been specified.")
            else:
                sampler = self.sampler
                if verbose:
                    print("Using class specified sampler.")

        if loss is None:
            if self.loss is None:
                raise ValueError("Loss has not been specified.")
            else:
                loss = self.loss
                if verbose:
                    print("Using class specified loss.")

        #check whether the sampler is trained on G
        if not sampler.is_trained(self.fsoi, G):
            if verbose:
                # Retraining will be triggered by the sample function
                print('Sampler was not trained on (fsoi, G). Retraining.')

        # sample perturbed
        perturbed_foiss = np.zeros((self.fsoi.shape[0], nr_runs,
                                   X_test.shape[0]))
        for kk in np.arange(0, nr_runs, 1):
            #TODO(gcsk) assess: does te sampler return the same sample every time?
            perturbed_foiss[:, kk, :] = np.transpose(sampler.sample(X_test, self.fsoi, G))

        lss = np.zeros((self.fsoi.shape[0], nr_runs, X_test.shape[0]))

        # compute observasitonwise loss differences for all runs and fois
        for jj in np.arange(0, self.fsoi.shape[0], 1):
            # copy of the data where perturbed variables are copied into
            X_test_perturbed = np.array(X_test)
            for kk in np.arange(0, nr_runs, 1):
                # replaced with perturbed
                X_test_perturbed[:, jj] = perturbed_foiss[jj, kk, :]
                # compute difference in observationwise loss
                lss[jj, kk, :] = (loss(self.model(X_test_perturbed), y_test) -
                                  loss(self.model(X_test), y_test))

        # return explanation object
        ex_name = 'RFI'
        result = explanation.Explanation(self.fsoi, lss, fs_names=self.fs_names)
        if return_perturbed:
            return result, perturbed_foiss
        else:
            return result