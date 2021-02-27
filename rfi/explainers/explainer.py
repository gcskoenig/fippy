"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""

import numpy as np
import pandas as pd
import rfi.utils as utils
import rfi.explanation.explanation as explanation
import logging
import rfi.explanation.decomposition as decomposition_ex
import enlighten  # TODO add to requirements

logger = logging.getLogger(__name__)


class Explainer:
    """Implements a number of feature importance algorithms

    Default samplers or loss function can be defined.
    Masks allows to specify for which features importance
    shall be computed.

    Attributes:
        model: Model or predict function.
        fsoi: Features of interest. Columnnames.
        X_train: Training data for Resampling. Pandas dataframe.
        sampler: default sampler.
        loss: default loss.
        fs_names: list of strings with feature input_var_names
    """

    def __init__(self, model, fsoi, X_train, sampler=None, decorrelator=None,
                 loss=None, fs_names=None):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        self.model = model
        self.fsoi = fsoi # now column names, not indexes
        self.X_train = X_train
        self.loss = loss
        self.sampler = sampler
        self.decorrelator = decorrelator
        # check whether feature set is valid
        self._valid_fset(self.fsoi)
        # feature names deprecated as we are working with dataframes now
        # TODO(gcsk): all features or just features of interest?
        # self.fs_names = np.array(fs_names)
        # if self.fs_names is None:
        #    self.fs_names = np.array([utils.ix_to_desc(jj)
        #                              for jj in range(X_train.shape[1])])

    def _valid_fset(self, fset):
        if set(fset).issubset(self.X_train.columns):
            raise ValueError("Feature set does not match "
                             "the dataframe's column names.")
        else:
            return True

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

    def rfi(self, X_eval, y_eval, G, R=None, sampler=None, loss=None, nr_runs=10, return_perturbed=False, train_allowed=True):
        """Computes Relative Feature importance

        # TODO(gcsk): allow handing a sample as argument
        #             (without needing sampler)

        Args:
            X_eval: data to use for resampling and evaluation.
            y_eval: labels for evaluation.
            G: relative feature set
            D: features, used by the predictive model
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

        if D is None:
            D = X_eval.columns

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
        nr_fsoi, nr_obs = len(self.fsoi), X_eval.shape[0]
        index = utils.create_multiindex(['sample', 'i'],
                                        [np.arange(nr_runs),
                                         np.arange(nr_obs)])
        X_fsoi_pert = pd.DataFrame([], index=index)
        # perturbed_foiss = np.zeros((nr_fsoi, nr_runs, nr_obs))

        # sample perturbed versions
        for foi in self.fsoi:
            x_foi_pert = sampler.sample(
                X_eval, [foi], G, num_samples=nr_runs)
            X_fsoi_pert[foi] = x_foi_pert

        scores = pd.DataFrame([], index=index)
        # lss = np.zeros((nr_fsoi, nr_runs, X_eval.shape[0]))

        # compute observasitonwise loss differences for all runs and fois
        for foi in self.fsoi:
            # copy of the data where perturbed variables are copied into
            for kk in np.arange(0, nr_runs, 1):
                # replaced with perturbed
                X_eval_tilde = X_eval.copy()
                X_eval_tilde[foi] = X_fsoi_pert.loc[0, :][foi]
                # X_eval_one_perturbed[:, self.fsoi[jj]]
                # = perturbed_foiss[jj, kk, :]
                # using only seen while training features

                # make sure model can handle it (selection and ordering)
                X_eval_tilde_model = X_eval_tilde[D]
                # X_eval_one_perturbed_model = X_eval_one_perturbed[:, D]
                X_eval_model = X_eval[D]

                # compute difference in observationwise loss
                loss_pert = loss(y_eval, self.model(X_eval_tilde_model))
                loss_orig = loss(y_eval, self.model(X_eval_model))
                scores.loc[kk, :][foi] = loss_pert - loss_orig
                # lss[jj, kk, :] = loss_pert - loss_orig

        # return explanation object
        ex_name = 'RFI^{}'.format(G)
        result = explanation.Explanation(
            self.fsoi, scores,
            ex_name=ex_name)

        if return_perturbed:
            logger.debug('Return both explanation and perturbed.')
            return result, X_fsoi_pert
        else:
            logger.debug('Return explanation object only')
            return result

    def rfa(self, X_eval, y_eval, K, D=None, sampler=None, decorrelator=None,
            loss=None, nr_runs=10, return_perturbed=False, train_allowed=True,
            ex_name=None):
        """Computes Feature Association

        Args:
            X_eval: data to use for resampling and evaluation.
            y_eval: labels for evaluation.
            K: features not to be reconstracted
            D: model features (including their required ordering)
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

        if D is None:
            D = X_eval.columns
        # all_fs = np.arange(X_test.shape[1])

        # check whether the sampler is trained for the baseline perturbation
        if not sampler.is_trained(D, []):
            # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(D, [])
                logger.info('Training sampler on {}|{}'.format(D, []))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {}|{}'.format(D, []))
        else:
            txt = '\tCheck passed: Sampler is already trained on '
            txt = txt + '{}|{}'.format(D, [])
            logger.debug(txt)

        # check for each of the features of interest
        for foi in self.fsoi:
            if not sampler.is_trained(D, [foi]):
                # train if allowed, otherwise raise error
                if train_allowed:
                    sampler.train(D, [foi])
                    logger.info(
                        'Training sampler on {}|{}'.format(D, [foi]))
                else:
                    raise RuntimeError(
                        'Sampler is not trained on {}|{}'.format(D, [foi]))
            else:
                txt = '\tCheck passed: Sampler is already trained on '
                txt = txt + '{}|{}'.format(D, [foi])
                logger.debug(txt)

        # check whether decorrelators have been trained
        for foi in self.fsoi:
            if not decorrelator.is_trained(K, [foi], []):
                if train_allowed:
                    decorrelator.train(K, [foi], [])
                    txt = 'Training decorrelator on '
                    txt = txt + '{} idp {} | {}'.format(K, [foi], [])
                    logger.info(txt)
                else:
                    txt = 'Decorrelator is not trained on '
                    txt = txt + '{} {} | {}'.format(K, [foi], [])
                    raise RuntimeError(txt)
            else:
                logger.debug('\tCheck passed: '
                             'Decorrelator is already trained on '
                             '{} {} | {}'.format(K, [foi], []))

        # initialize array for the perturbed samples
        nr_obs = X_eval.shape[0]

        # initialize pandas dataframes for X_eval_tilde baseline
        # and X_eval_tilde reconstrcted
        index_bsln = utils.create_multiindex(['sample', 'i'],
                                             [np.arange(nr_runs),
                                              np.arange(nr_obs)])
        X_eval_tilde_bsln = pd.DataFrame([], index=index_bsln)
        index_rcnstr = utils.create_multiindex(['foi', 'sample', 'i'],
                                               [self.fsoi,
                                                np.arange(nr_runs),
                                                np.arange(nr_obs)])
        X_eval_tilde_rcnstr = pd.DataFrame([], index=index_rcnstr)

        # sample baseline
        X_eval_tilde_bsln = sampler.sample(X_eval, D, [], num_samples=nr_runs)

        # sample perturbed versions
        for foi in self.fsoi:
            sample = sampler.sample(X_eval, D, [foi], num_samples=nr_runs)
            for kk in np.arange(nr_runs):
                sample_decorr = decorrelator.decorrelate(sample.loc[kk, :],
                                                         K, [foi], [])
                X_eval_tilde_rcnstr.loc[foi, :, :] = sample_decorr

        # create empty scores data frame
        index_scores = utils.create_multiindex(['sample', 'i'],
                                               [np.arange(nr_runs),
                                                np.arange(nr_obs)])
        scores = pd.DataFrame([], index=index_scores)

        # compute observasitonwise loss differences for all runs and fois
        for kk in np.arange(nr_runs):
            l_pb = loss(y_eval, self.model(X_eval_tilde_bsln.loc[kk, :][D]))
            for foi in self.fsoi:
                l_rc = loss(y_eval,
                            self.model(X_eval_tilde_rcnstr[foi, kk, :][D]))
                scores.loc[kk, :][foi] = l_pb - l_rc

        if ex_name is None:
            ex_name = 'Unknown rfa'

        # return explanation object
        result = explanation.Explanation(self.fsoi,
                                         scores,
                                         ex_name=ex_name)
        if return_perturbed:
            raise NotImplementedError(
                'Returning perturbed not implemented yet.')
        else:
            logger.debug('Return explanation object only')
            return result

    def decomposition(self, imp_type, fsoi, partial_ordering, X_eval, y_eval,
                      nr_orderings=None, nr_runs=3, show_pbar=True, **kwargs):
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

        explnr_fnc = None
        if imp_type == 'rfi':
            explnr_fnc = self.rfi
        elif imp_type == 'rfa':
            explnr_fnc = self.rfa
        else:
            raise ValueError('Importance type '
                             '{} not implemented'.format(imp_type))

        # values (nr_perm, nr_runs, nr_components, nr_fsoi)
        # components are: (elements of ordering,..., remainder)
        # elements of ordering are sorted in increasing order

        component_names = np.unique(utils.flatten(partial_ordering))
        component_names = list(component_names)
        component_names.insert(0, 'remainder')
        component_names.insert(0, 'total')
        nr_components = len(component_names)

        # create dataframe for computation results
        index = utils.create_multiindex(['component', 'ordering', 'run'],
                                        [component_names,
                                         np.arange(nr_orderings),
                                         np.arange(nr_runs)])
        arr = np.zeros((nr_components * nr_orderings * nr_runs,
                        len(self.fsoi)))
        decomposition = pd.DataFrame(arr, index=index, columns=self.fsoi)
        # values = np.zeros((nr_orderings, nr_runs, nr_components, len(fsoi)))

        if show_pbar:
            mgr = enlighten.get_manager()
            pbar = mgr.counter(total=nr_orderings, desc='decomposition',
                               unit='orderings')

        for kk in np.arange(nr_orderings):
            if show_pbar:
                pbar.update()

            ordering = utils.sample_partial(partial_ordering)
            logging.info('Ordering : {}'.format(ordering))

            # total values
            expl = explnr_fnc(X_eval, y_eval, [], nr_runs=nr_runs, **kwargs)
            decomposition.loc['total', kk, :] = expl.fi_vals()

            previous = decomposition.loc['total', kk, :]
            current = None

            for jj in np.arange(1, len(ordering) + 1):
                # get current new variable and respective set
                current_ix = ordering[jj - 1]
                G = ordering[:jj]

                # compute and store feature importance
                expl = explnr_fnc(X_eval, y_eval, G, nr_runs=nr_runs, **kwargs)
                current = expl.fi_vals()

                # compute difference
                decomposition.loc[current_ix, kk, :] = previous - current
                previous = current

            decomposition.loc['remainder'] = previous

        ex = decomposition_ex.DecompositionExplanation(self.fsoi,
                                                       decomposition,
                                                       ex_name=None)
        return ex

    def sage(self, type, X_test, y_test, nr_orderings,
             nr_runs=10, sampler=None, loss=None,
             train_allowed=True, return_orderings=False):
        """
        Compute Shapley Additive Global Importance values.
        Args:
            type: either 'rfi' or 'rfa', depending on whether conditional
                or marginal resampling of the remaining features shall
                be used
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
        if X_test.shape[1] != len(self.fsoi):
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
            (len(self.fsoi), nr_runs, X_test.shape[0], nr_orderings))

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
            fsoi_names=self.fsoi,
            ex_name='SAGE')

        if return_orderings:
            raise NotImplementedError(
                'Returning errors is not implemented yet.')

        return result


   # def fa(self, X_test, y_test, K, sampler=None, loss=None, nr_runs=10,
   #         return_perturbed=False, train_allowed=True):
   #      """Computes Feature Association

   #      # TODO(gcsk): allow handing a sample as argument
   #      #             (without needing sampler)

   #      Args:
   #          X_test: data to use for resampling and evaluation.
   #          y_test: labels for evaluation.
   #          G: relative feature set
   #          sampler: choice of sampler. Default None. Will throw an error
   #            when sampler is None and self.sampler is None as well.
   #          loss: choice of loss. Default None. Will throw an Error when
   #            both loss and self.loss are None.
   #          nr_runs: how often the experiment shall be run
   #          return_perturbed: whether the sampled perturbed
   #              versions shall be returned
   #          train_allowed: whether the explainer is allowed
   #              to train the sampler

   #      Returns:
   #          result: An explanation object with the RFI computation
   #          perturbed_foiss (optional): perturbed features of
   #              interest if return_perturbed
   #      """

   #      if sampler is None:
   #          if self._sampler_specified():  # may throw an error
   #              sampler = self.sampler
   #              logger.debug("Using class specified sampler.")

   #      if loss is None:
   #          if self._loss_specified():  # may throw an error
   #              loss = self.loss
   #              logger.debug("Using class specified loss.")

   #      all_fs = np.arange(X_test.shape[1])

   #      # check whether the sampler is trained for the baseline perturbation
   #      if not sampler.is_trained(all_fs, []):
   #          # train if allowed, otherwise raise error
   #          if train_allowed:
   #              sampler.train(all_fs, [])
   #              logger.info('Training sampler on {}|{}'.format(all_fs, []))
   #          else:
   #              raise RuntimeError(
   #                  'Sampler is not trained on {}|{}'.format(all_fs, []))
   #      else:
   #          logger.debug(
   #              '\tCheck passed: '
   #              'Sampler is already trained on '
   #              '{}|{}'.format(all_fs, []))

   #      # check for each of the features of interest
   #      for f in self.fsoi:
   #          if not sampler.is_trained(K, [f]):
   #              # train if allowed, otherwise raise error
   #              if train_allowed:
   #                  sampler.train(K, [f])
   #                  logger.info('Training sampler on {}|{}'.format(K, [f]))
   #              else:
   #                  raise RuntimeError(
   #                      'Sampler is not trained on {}|{}'.format(K, [f]))
   #          else:
   #              logger.debug(
   #                  '\tCheck passed: '
   #                  'Sampler is already trained on '
   #                  '{}|{}'.format(K, [f]))

   #      # initialize array for the perturbed samples
   #      nr_fsoi, nr_features = len(self.fsoi), len(all_fs)
   #      nr_obs = X_test.shape[0]
   #      perturbed_reconstr = np.zeros((nr_fsoi, nr_obs, nr_runs, len(K)))
   #      perturbed_baseline = np.zeros((nr_obs, nr_runs, nr_features))

   #      # sample baseline
   #      sample = sampler.sample(X_test, all_fs, [], num_samples=nr_runs)
   #      perturbed_baseline = sample

   #      # sample perturbed versions
   #      for jj in range(len(self.fsoi)):
   #          sample = sampler.sample(
   #              X_test, K, [self.fsoi[jj]], num_samples=nr_runs)
   #          perturbed_reconstr[jj, :, :, :] = sample

   #      lss = np.zeros((len(self.fsoi), nr_runs, X_test.shape[0]))

   #      # compute observasitonwise loss differences for all runs and fois
   #      for jj in np.arange(0, len(self.fsoi), 1):
   #          for kk in np.arange(0, nr_runs, 1):
   #              # replaced with perturbe
   #              X_test_reconstructed = np.array(perturbed_baseline[:, kk, :])
   #              X_test_reconstructed[:, K] = perturbed_reconstr[jj, :, kk, :]
   #              # compute difference in observationwise loss
   #              l_pb = loss(y_test, self.model(perturbed_baseline[:, kk, :]))
   #              l_rc = loss(y_test, self.model(X_test_reconstructed))
   #              lss[jj, kk, :] = l_pb - l_rc

   #      # return explanation object
   #      result = explanation.Explanation(
   #          self.fsoi, lss, fsoi_names=self.fsoi, ex_name='SI')
   #      if return_perturbed:
   #          raise NotImplementedError(
   #              'Returning perturbed not implemented yet.')
   #          # logger.debug('Return both explanation and perturbed.')
   #          # return result, perturbed_baseline, perturbed_reconstr
   #      else:
   #          logger.debug('Return explanation object only')
   #          return result
