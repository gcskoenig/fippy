"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""

import numpy as np
import rfi.utils as utils
import rfi.explanation.explanation as explanation
import logging


class Explainer():
    """Implements a number of feature importance algorithms

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
            names = [utils.ix_to_desc(jj) for jj in range(X_train.shape[1])]
            self.fs_names = names

    def _sampler_specified(self):
        if self.sampler is None:
            raise ValueError("Sampler has not been specified.")
        else:
            return True

    def _loss_specified(self):
        if self.loss is None:
            raise ValueError("Loss has not been specified.")
        else:
            return True

    def rfi(self, X_test, y_test, G, sampler=None, loss=None, nr_runs=10, return_perturbed=False, train_allowed=True):
        """Computes Relative Feature importance

        # TODO(gcsk): allow handing a sample as argument
        #             (without needing sampler)

        Args:
            X_test: data to use for resampling and evaluation.
            y_test: labels for evaluation.
            G: relative feature set
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.
            nr_runs: how often the experiment shall be run
            return_perturbed: whether the sampled perturbed versions shall be returned
            train_allowed: whether the explainer is allowed to train the sampler

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of interest if return_perturbed
        """

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logging.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logging.debug("Using class specified loss.")

        #check whether the sampler is trained for each fsoi conditional on G
        for f in self.fsoi:
            if not sampler.is_trained([f], G):
                # train if allowed, otherwise raise error
                if train_allowed:
                    sampler.train([f], G)
                    logging.info('Training sampler on {}|{}'.format([f],G))
                else:
                    raise RuntimeError('Sampler is not trained on {}|{}'.format([f], G))
            else:
                logging.info('\tCheck passed: Sampler is already trained on {}|{}'.format([f],G))

        # initialize array for the perturbed samples
        nr_fsoi, nr_obs = self.fsoi.shape[0], X_test.shape[0]
        perturbed_foiss = np.zeros((nr_fsoi, nr_runs, nr_obs))

        # sample perturbed versions
        for jj in range(len(self.fsoi)):
            perturbed_foiss[jj, :, :] = sampler.sample(X_test, [self.fsoi[jj]], G, num_samples=nr_runs).reshape((nr_obs, nr_runs)).T
         
        lss = np.zeros((self.fsoi.shape[0], nr_runs, X_test.shape[0]))

        # compute observasitonwise loss differences for all runs and fois
        for jj in np.arange(0, self.fsoi.shape[0], 1):
            # copy of the data where perturbed variables are copied into
            X_test_one_perturbed = np.array(X_test)
            for kk in np.arange(0, nr_runs, 1):
                # replaced with perturbed
                X_test_one_perturbed[:, self.fsoi[jj]] = perturbed_foiss[jj, kk, :]
                # compute difference in observationwise loss
                lss[jj, kk, :] = (loss(self.model(X_test_one_perturbed), y_test) -
                                  loss(self.model(X_test), y_test))

        # return explanation object
        ex_name = 'RFI^{}'.format(G)
        result = explanation.Explanation(self.fsoi, lss, fsoi_names=self.fs_names[self.fsoi])
        if return_perturbed:
            logging.debug('Return both explanation and perturbed.')
            return result, perturbed_foiss
        else:
            logging.debug('Return explanation object only')
            return result

    def si(self, X_test, y_test, K, sampler=None, loss=None, nr_runs=10, return_perturbed=False, train_allowed=True):
        """Computes Relative Shared Importance

        # TODO(gcsk): allow handing a sample as argument
        #             (without needing sampler)

        Args:
            X_test: data to use for resampling and evaluation.
            y_test: labels for evaluation.
            G: relative feature set
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.
            nr_runs: how often the experiment shall be run
            return_perturbed: whether the sampled perturbed versions shall be returned
            train_allowed: whether the explainer is allowed to train the sampler

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of interest if return_perturbed
        """

        if sampler is None:
            if self._sampler_specified(): # may throw an error
                sampler = self.sampler
                logging.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified(): # may throw an error
                loss = self.loss
                logging.debug("Using class specified loss.")

        all_fs = np.arange(X_test.shape[1])

        #check whether the sampler is trained for the baseline perturbation
        if not sampler.is_trained(all_fs, []):
                # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(all_fs, [])
                logging.info('Training sampler on {}|{}'.format(all_fs, []))
            else:
                raise RuntimeError('Sampler is not trained on {}|{}'.format(all_fs, []))
        else:
            logging.info('\tCheck passed: Sampler is already trained on {}|{}'.format(all_fs, []))

        # check for each of the features of interest
        for f in self.fsoi:
            if not sampler.is_trained(K, [f]):
                # train if allowed, otherwise raise error
                if train_allowed:
                    sampler.train(K, [f])
                    logging.info('Training sampler on {}|{}'.format(K, [f]))
                else:
                    raise RuntimeError('Sampler is not trained on {}|{}'.format(K, [f]))
            else:
                logging.info('\tCheck passed: Sampler is already trained on {}|{}'.format(K, [f]))

        # initialize array for the perturbed samples
        nr_fsoi, nr_obs, nr_features = self.fsoi.shape[0], X_test.shape[0], len(all_fs)
        perturbed_reconstr = np.zeros((nr_fsoi, nr_obs, nr_runs, len(K)))
        perturbed_baseline = np.zeros((nr_obs, nr_runs, nr_features))

        
        # sample baseline
        sample = sampler.sample(X_test, all_fs, [], num_samples=nr_runs)
        perturbed_baseline = sample

        # sample perturbed versions
        for jj in range(len(self.fsoi)):
            sample = sampler.sample(X_test, K,[self.fsoi[jj]] , num_samples=nr_runs)
            perturbed_reconstr[jj, :, :, :] = sample

        lss = np.zeros((self.fsoi.shape[0], nr_runs, X_test.shape[0]))
        
        # compute observasitonwise loss differences for all runs and fois
        for jj in np.arange(0, self.fsoi.shape[0], 1):
            for kk in np.arange(0, nr_runs, 1):
                # replaced with perturbe
                X_test_reconstructed = np.array(perturbed_baseline[:, kk, :])
                X_test_reconstructed[:, K] = perturbed_reconstr[jj, :, kk, :]
                # compute difference in observationwise loss
                lss[jj, kk, :] = (loss(self.model(perturbed_baseline[:, kk, :]), y_test) -
                                  loss(self.model(X_test_reconstructed), y_test))

        # return explanation object
        ex_name = 'si'
        result = explanation.Explanation(self.fsoi, lss, fsoi_names=self.fs_names[self.fsoi])
        if return_perturbed:
            raise NotImplementedError('Returning perturbed not implemented yet.')
            logging.debug('Return both explanation and perturbed.')
            return result, perturbed_baseline, perturbed_reconstr
        else:
            logging.debug('Return explanation object only')
            return result

    def sage(self, X_test, y_test, nr_orderings, nr_runs=10, sampler=None, loss=None, train_allowed=True, return_orderings=False):
        """Compute Shapley Additive Global Importance values.
        
        Args:
            X_test: data to use for resampling and evaluation.
            y_test: labels for evaluation.
            nr_orderings: number of orderings in which features enter the model
            nr_runs: how often each value function shall be computed
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.
            train_allowed: whether the explainer is allowed to train the sampler

        Returns:
            result: an explanation object containing the respective pairwise lossdifferences
            with shape (nr_fsoi, nr_runs, nr_obs, nr_orderings)
            orderings (optional): an array containing the respective orderings if return_orderings
        """
        # the method is currently not build for situations where we are only interested in 
        # a subset of the model's features
        if X_test.shape[1] != self.fsoi.shape[0]:
            logging.debug('self.fsoi: {}'.format(self.fsoi))
            logging.debug('#features in model: {}'.format(X_test.shape[1]))
            raise RuntimeError('self.fsoi is not identical to all features')

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logging.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logging.debug("Using class specified loss.")

        lss = np.zeros((self.fsoi.shape[0], nr_runs, X_test.shape[0], nr_orderings))

        for ii in range(nr_orderings):
            ordering = np.random.permutation(len(self.fsoi))
            # resample multiple times
            for kk in range(nr_runs):
                # enter one feature at a time
                y_hat_base = np.repeat(np.mean(self.model(X_test)), X_test.shape[0])
                for jj in np.arange(1, len(self.fsoi), 1):
                    # compute change in performance by entering the respective feature
                    # store the result in the right place
                    # validate training of sampler
                    impute, fixed = self.fsoi[ordering[jj:]], self.fsoi[ordering[:jj]]
                    logging.debug('{}:{}:{}: fixed, impute: {}|{}'.format(ii, kk, jj, impute, fixed))
                    if not sampler.is_trained(impute, fixed):
                        # train if allowed, otherwise raise error
                        if train_allowed:
                            sampler.train(impute, fixed)
                            logging.info('Training sampler on {}|{}'.format(impute, fixed))
                        else:
                            raise RuntimeError('Sampler is not trained on {}|{}'.format(impute, fixed))
                    X_test_perturbed = np.array(X_test)
                    impute_sample = sampler.sample(X_test, impute, fixed, num_samples=1).reshape((X_test.shape[0], len(impute)))
                    X_test_perturbed[:, impute] = impute_sample
                    # sample replacement, create replacement matrix
                    y_hat_new = self.model(X_test_perturbed)
                    lss[self.fsoi[ordering[jj-1]], kk, :, ii] = loss(y_hat_base, y_test) - loss(y_hat_new, y_test)
                    y_hat_base = y_hat_new
                y_hat_new = self.model(X_test)
                lss[self.fsoi[ordering[-1]], kk, :, ii] = loss(y_hat_base, y_test) - loss(y_hat_new, y_test)

        ex_name = 'SAGE'
        result = explanation.Explanation(self.fsoi, lss, fsoi_names=self.fs_names[self.fsoi])

        if return_orderings:
            raise NotImplementedError('Returning errors is not implemented yet.')

        return result