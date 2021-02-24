"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""

import numpy as np
import rfi.utils as utils
import rfi.explanation.explanation as explanation
import logging
import rfi.explanation.decomposition as decomposition_ex

logger = logging.getLogger(__name__)


class Explainer:
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

    def __init__(self, model, fsoi, X_train, sampler=None, decorrelator=None,
                 loss=None, fs_names=None):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        self.model = model
        self.fsoi = fsoi
        self.X_train = X_train
        self.loss = loss
        self.sampler = sampler
        self.decorrelator = decorrelator
        # TODO(gcsk): all features or just features of interest?
        self.fs_names = np.array(fs_names)
        if self.fs_names is None:
            self.fs_names = np.array([utils.ix_to_desc(jj)
                                      for jj in range(X_train.shape[1])])

    def _sampler_specified(self):
        if self.sampler is None:
            raise ValueError("Sampler has not been specified.")
        else:
            return True

    def _decorrelator_specified(self):
        if self.decorrelator is None:
            raise ValueError("Decorrelator has not been specified.")
        else:
            return True

    def _loss_specified(self):
        if self.loss is None:
            raise ValueError("Loss has not been specified.")
        else:
            return True

    def rfi(self, X_test, y_test, G, R_j=None, sampler=None, loss=None, nr_runs=10, return_perturbed=False, train_allowed=True):
        """Computes Relative Feature importance

        # TODO(gcsk): allow handing a sample as argument
        #             (without needing sampler)

        Args:
            X_test: data to use for resampling and evaluation.
            y_test: labels for evaluation.
            G: relative feature set
            R_j: features, used by a predictive model
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.
            nr_runs: how often the experiment shall be run
            return_perturbed: whether the sampled perturbed versions
                shall be returned
            train_allowed: whether the explainer is allowed to train
                the sampler

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of
                interest if return_perturbed
        """

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logger.debug("Using class specified loss.")

        # check whether the sampler is trained for each fsoi conditional on G
        for f in self.fsoi:
            if not sampler.is_trained([f], G):
                # train if allowed, otherwise raise error
                if train_allowed:
                    sampler.train([f], G)
                    logger.info('Training sampler on {}|{}'.format([f], G))
                else:
                    raise RuntimeError(
                        'Sampler is not trained on {}|{}'.format([f], G))
            else:
                txt = '\tCheck passed: Sampler is already trained on'
                txt = txt + '{}|{}'.format([f], G)
                logger.debug(txt)

        # initialize array for the perturbed samples
        nr_fsoi, nr_obs = self.fsoi.shape[0], X_test.shape[0]
        perturbed_foiss = np.zeros((nr_fsoi, nr_runs, nr_obs))

        # sample perturbed versions
        for jj in range(len(self.fsoi)):
            tmp = sampler.sample(
                X_test, [self.fsoi[jj]], G, num_samples=nr_runs)
            perturbed_foiss[jj, :, :] = tmp.reshape((nr_obs, nr_runs)).T
        lss = np.zeros((self.fsoi.shape[0], nr_runs, X_test.shape[0]))

        # compute observasitonwise loss differences for all runs and fois
        for jj in np.arange(0, self.fsoi.shape[0], 1):
            # copy of the data where perturbed variables are copied into
            X_test_one_perturbed = np.array(X_test)
            for kk in np.arange(0, nr_runs, 1):
                # replaced with perturbed
                X_test_one_perturbed[:, self.fsoi[jj]] = perturbed_foiss[jj, kk, :]
                # using only seen while training features
                if R_j is not None:
                    X_test_one_perturbed_model = X_test_one_perturbed[:, R_j]
                    X_test_model = X_test[:, R_j]
                else:
                    X_test_one_perturbed_model = X_test_one_perturbed
                    X_test_model = X_test
                    
                # compute difference in observationwise loss
                loss_pert = loss(y_test, self.model(X_test_one_perturbed_model))
                loss_orig = loss(y_test, self.model(X_test_model))
                lss[jj, kk, :] = loss_pert - loss_orig

        # return explanation object
        ex_name = 'RFI^{}'.format(G)
        result = explanation.Explanation(
            self.fsoi, lss,
            fsoi_names=self.fs_names[self.fsoi],
            ex_name=ex_name)

        if return_perturbed:
            logger.debug('Return both explanation and perturbed.')
            return result, perturbed_foiss
        else:
            logger.debug('Return explanation object only')
            return result

    def rfa(self, X_test, y_test, K, sampler=None, decorrelator=None,
            loss=None, nr_runs=10, return_perturbed=False, train_allowed=True):
        """Computes Feature Association
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
            return_perturbed: whether the sampled perturbed
                versions shall be returned
            train_allowed: whether the explainer is allowed
                to train the sampler

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of
                interest if return_perturbed
        """

        if sampler is None:
            if self._sampler_specified():  # may throw an error
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if decorrelator is None:
            if self._decorrelator_specified():  # may throw error
                decorrelator = self.decorrelator
                logger.debug("Using class specified decorrelator")

        if loss is None:
            if self._loss_specified():  # may throw an error
                loss = self.loss
                logger.debug("Using class specified loss.")

        all_fs = np.arange(X_test.shape[1])

        # check whether the sampler is trained for the baseline perturbation
        if not sampler.is_trained(all_fs, []):
            # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(all_fs, [])
                logger.info('Training sampler on {}|{}'.format(all_fs, []))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {}|{}'.format(all_fs, []))
        else:
            txt = '\tCheck passed: Sampler is already trained on '
            txt = txt + '{}|{}'.format(all_fs, [])
            logger.debug(txt)

        # check for each of the features of interest
        for f in self.fsoi:
            if not sampler.is_trained(all_fs, [f]):
                # train if allowed, otherwise raise error
                if train_allowed:
                    sampler.train(all_fs, [f])
                    logger.info(
                        'Training sampler on {}|{}'.format(all_fs, [f]))
                else:
                    raise RuntimeError(
                        'Sampler is not trained on {}|{}'.format(all_fs, [f]))
            else:
                txt = '\tCheck passed: Sampler is already trained on '
                txt = txt + '{}|{}'.format(all_fs, [f])
                logger.debug(txt)

        # check whether decorrelators have been trained
        for f in self.fsoi:
            if not decorrelator.is_trained(K, [f], []):
                if train_allowed:
                    decorrelator.train(K, [f], [])
                    txt = 'Training decorrelator on '
                    txt = txt + '{} idp {} | {}'.format(K, [f], [])
                    logger.info(txt)
                else:
                    txt = 'Decorrelator is not trained on '
                    txt = txt + '{} {} | {}'.format(K, [f], [])
                    raise RuntimeError(txt)
            else:
                logger.debug('\tCheck passed: '
                             'Decorrelator is already trained on '
                             '{} {} | {}'.format(K, [f], []))

        # initialize array for the perturbed samples
        nr_fsoi, nr_features = self.fsoi.shape[0], len(all_fs)
        nr_obs = X_test.shape[0]
        perturbed_reconstr = np.zeros((nr_fsoi, nr_obs, nr_runs, nr_features))
        perturbed_baseline = np.zeros((nr_obs, nr_runs, nr_features))

        # sample baseline
        sample = sampler.sample(X_test, all_fs, [], num_samples=nr_runs)
        perturbed_baseline = sample

        # sample perturbed versions
        for jj in range(len(self.fsoi)):
            sample = sampler.sample(
                X_test, all_fs, [self.fsoi[jj]], num_samples=nr_runs)
            for kk in np.arange(nr_runs):
                sample_decorr = decorrelator.decorrelate(
                    sample[:, kk, :], K, [jj], [])
                perturbed_reconstr[jj, :, kk, :] = sample_decorr

        lss = np.zeros((self.fsoi.shape[0], nr_runs, X_test.shape[0]))

        # compute observasitonwise loss differences for all runs and fois
        for jj in np.arange(0, self.fsoi.shape[0], 1):
            for kk in np.arange(0, nr_runs, 1):
                # replaced with perturbe
                X_test_reconstructed = np.array(perturbed_baseline[:, kk, :])
                X_test_reconstructed = perturbed_reconstr[jj, :, kk, :]
                # compute difference in observationwise loss
                l_pb = loss(self.model(perturbed_baseline[:, kk, :]), y_test)
                l_rc = loss(self.model(X_test_reconstructed), y_test)
                lss[jj, kk, :] = l_pb - l_rc

        # return explanation object
        result = explanation.Explanation(
            self.fsoi, lss, fsoi_names=self.fs_names[self.fsoi], ex_name='SI')
        if return_perturbed:
            raise NotImplementedError(
                'Returning perturbed not implemented yet.')
            # logger.debug('Return both explanation and perturbed.')
            # return result, perturbed_baseline, perturbed_reconstr
        else:
            logger.debug('Return explanation object only')
            return result

    def decomposition(self, imp_type, fsoi, partial_ordering, X_test, y_test,
                      nr_orderings=None, nr_runs=3):
        """
        Given a partial ordering, this code allows to decompose
        feature importance or feature association for a given set of
        features into its respective indirect or direct components.

        Args:
            imp_type: Either 'rfi' or 'rfa'
            fois: features, for which the importance scores (of type imp_type)
                are to be decomposed
            partial_ordering: partial ordering for the decomposition
            X_test: test data
            y_test: test labels
            nr_orderings: number of total orderings to sample
                (given the partial) ordering
            nr_runs: number of runs for each feature importance score
                computation

        Returns:
            means, stds: means and standard deviations for each
                component and each feature. numpy.array with shape
                (#components, #fsoi)
        """
        if nr_orderings is None:
            nr_orderings = len(utils.flatten(partial_ordering))**2

        # values (nr_perm, nr_runs, nr_components, nr_fsoi)
        # components are: (elements of ordering,..., remainder)
        # elements of ordering are sorted in increasing order
        nr_components = len(utils.flatten(partial_ordering)) + 1
        values = np.zeros((len(fsoi), nr_components, nr_orderings, nr_runs))
        # values = np.zeros((nr_orderings, nr_runs, nr_components, len(fsoi)))

        for kk in np.arange(nr_orderings):
            rfs = np.zeros(
                (nr_runs, len(utils.flatten(partial_ordering)) + 1, len(fsoi)))

            ordering = utils.sample_partial(partial_ordering)
            logging.info('Ordering : {}'.format(ordering))

            for jj in np.arange(len(ordering) + 1):
                G = ordering[:jj]
                expl = None
                if imp_type == 'rfi':
                    expl = self.rfi(X_test, y_test, G, nr_runs=nr_runs)
                elif imp_type == 'rfa':
                    expl = self.rfa(X_test, y_test, G, nr_runs=nr_runs)
                rfs[:, jj, :] = expl.fi_vals(return_np=True).T

            # conditioning on all items in partial ordering
            values[:, -1, kk, :] = rfs[:, -1, :].T

            # importance contribution of the j-th component
            for jj in np.arange(1, len(ordering) + 1, 1):
                diffs = rfs[:, jj - 1, :] - rfs[:, jj, :]
                ix = utils.id_to_ix(ordering[jj - 1], ordering)
                values[:, ix, kk, :] = diffs.T

        component_names = np.unique(utils.flatten(partial_ordering))
        component_names = list(self.fs_names[component_names])
        component_names.append('remainder')
        fsoi_names = self.fs_names[self.fsoi]
        ex = decomposition_ex.DecompositionExplanation(self.fsoi, values,
                                                       fsoi_names,
                                                       component_names,
                                                       ex_name=None)
        return ex

    def fa(self, X_test, y_test, K, sampler=None, loss=None, nr_runs=10,
           return_perturbed=False, train_allowed=True):
        """Computes Feature Association

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
            return_perturbed: whether the sampled perturbed
                versions shall be returned
            train_allowed: whether the explainer is allowed
                to train the sampler

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of
                interest if return_perturbed
        """

        if sampler is None:
            if self._sampler_specified():  # may throw an error
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified():  # may throw an error
                loss = self.loss
                logger.debug("Using class specified loss.")

        all_fs = np.arange(X_test.shape[1])

        # check whether the sampler is trained for the baseline perturbation
        if not sampler.is_trained(all_fs, []):
            # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(all_fs, [])
                logger.info('Training sampler on {}|{}'.format(all_fs, []))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {}|{}'.format(all_fs, []))
        else:
            logger.debug(
                '\tCheck passed: '
                'Sampler is already trained on '
                '{}|{}'.format(all_fs, []))

        # check for each of the features of interest
        for f in self.fsoi:
            if not sampler.is_trained(K, [f]):
                # train if allowed, otherwise raise error
                if train_allowed:
                    sampler.train(K, [f])
                    logger.info('Training sampler on {}|{}'.format(K, [f]))
                else:
                    raise RuntimeError(
                        'Sampler is not trained on {}|{}'.format(K, [f]))
            else:
                logger.debug(
                    '\tCheck passed: '
                    'Sampler is already trained on '
                    '{}|{}'.format(K, [f]))

        # initialize array for the perturbed samples
        nr_fsoi, nr_features = self.fsoi.shape[0], len(all_fs)
        nr_obs = X_test.shape[0]
        perturbed_reconstr = np.zeros((nr_fsoi, nr_obs, nr_runs, len(K)))
        perturbed_baseline = np.zeros((nr_obs, nr_runs, nr_features))

        # sample baseline
        sample = sampler.sample(X_test, all_fs, [], num_samples=nr_runs)
        perturbed_baseline = sample

        # sample perturbed versions
        for jj in range(len(self.fsoi)):
            sample = sampler.sample(
                X_test, K, [self.fsoi[jj]], num_samples=nr_runs)
            perturbed_reconstr[jj, :, :, :] = sample

        lss = np.zeros((self.fsoi.shape[0], nr_runs, X_test.shape[0]))

        # compute observasitonwise loss differences for all runs and fois
        for jj in np.arange(0, self.fsoi.shape[0], 1):
            for kk in np.arange(0, nr_runs, 1):
                # replaced with perturbe
                X_test_reconstructed = np.array(perturbed_baseline[:, kk, :])
                X_test_reconstructed[:, K] = perturbed_reconstr[jj, :, kk, :]
                # compute difference in observationwise loss
                l_pb = loss(y_test, self.model(perturbed_baseline[:, kk, :]))
                l_rc = loss(y_test, self.model(X_test_reconstructed))
                lss[jj, kk, :] = l_pb - l_rc

        # return explanation object
        result = explanation.Explanation(
            self.fsoi, lss, fsoi_names=self.fs_names[self.fsoi], ex_name='SI')
        if return_perturbed:
            raise NotImplementedError(
                'Returning perturbed not implemented yet.')
            # logger.debug('Return both explanation and perturbed.')
            # return result, perturbed_baseline, perturbed_reconstr
        else:
            logger.debug('Return explanation object only')
            return result

    def sage(self, X_test, y_test, nr_orderings,
             nr_runs=10, sampler=None, loss=None,
             train_allowed=True, return_orderings=False):
        """
        Compute Shapley Additive Global Importance values.
        Args:
            X_test: data to use for resampling and evaluation.
            y_test: labels for evaluation.
            nr_orderings: number of orderings in which features enter the model
            nr_runs: how often each value function shall be computed
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.
            train_allowed: whether the explainer is allowed to
                train the sampler

        Returns:
            result: an explanation object containing the respective
                pairwise lossdifferences with shape
                (nr_fsoi, nr_runs, nr_obs, nr_orderings)
            orderings (optional): an array containing the respective
                orderings if return_orderings
        """
        # the method is currently not build for situations
        # where we are only interested in
        # a subset of the model's features
        if X_test.shape[1] != self.fsoi.shape[0]:
            logger.debug('self.fsoi: {}'.format(self.fsoi))
            logger.debug('#features in model: {}'.format(X_test.shape[1]))
            raise RuntimeError('self.fsoi is not identical to all features')

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logger.debug("Using class specified loss.")

        lss = np.zeros(
            (self.fsoi.shape[0], nr_runs, X_test.shape[0], nr_orderings))

        for ii in range(nr_orderings):
            ordering = np.random.permutation(len(self.fsoi))
            # resample multiple times
            for kk in range(nr_runs):
                # enter one feature at a time
                y_hat_base = np.repeat(
                    np.mean(self.model(X_test)), X_test.shape[0])
                for jj in np.arange(1, len(self.fsoi), 1):
                    # compute change in performance
                    # by entering the respective feature
                    # store the result in the right place
                    # validate training of sampler
                    impute, fixed = self.fsoi[ordering[jj:]
                                              ], self.fsoi[ordering[:jj]]
                    logger.debug('{}:{}:{}: fixed, impute: {}|{}'.format(
                        ii, kk, jj, impute, fixed))
                    if not sampler.is_trained(impute, fixed):
                        # train if allowed, otherwise raise error
                        if train_allowed:
                            sampler.train(impute, fixed)
                            logger.info(
                                'Training sampler on '
                                '{}|{}'.format(impute, fixed))
                        else:
                            raise RuntimeError(
                                'Sampler is not trained on '
                                '{}|{}'.format(impute, fixed))
                    X_test_perturbed = np.array(X_test)
                    imps = sampler.sample(X_test, impute, fixed, num_samples=1)
                    imps = imps.reshape((X_test.shape[0], len(impute)))
                    X_test_perturbed[:, impute] = imps
                    # sample replacement, create replacement matrix
                    y_hat_new = self.model(X_test_perturbed)
                    lb = loss(y_test, y_hat_base)
                    ln = loss(y_test, y_hat_new)
                    lss[self.fsoi[ordering[jj - 1]], kk, :, ii] = lb - ln
                    y_hat_base = y_hat_new
                y_hat_new = self.model(X_test)
                lss[self.fsoi[ordering[-1]], kk, :, ii] = loss(y_test, y_hat_base) - loss(y_test, y_hat_new)

        result = explanation.Explanation(
            self.fsoi, lss,
            fsoi_names=self.fs_names[self.fsoi],
            ex_name='SAGE')

        if return_orderings:
            raise NotImplementedError(
                'Returning errors is not implemented yet.')

        return result
