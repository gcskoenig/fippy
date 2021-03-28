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
import math

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
        self.fsoi = fsoi  # now column names, not indexes
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
        fset = set(list(fset))
        if not fset.issubset(self.X_train.columns):
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

    def tdi(self, X_eval, y_eval, G, D=None, sampler=None, loss=None,
            nr_runs=10, return_perturbed=False, train_allowed=True):
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
        nr_obs = X_eval.shape[0]
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
                arr = X_fsoi_pert.loc[(kk, slice(None)), foi].to_numpy()
                X_eval_tilde[foi] = arr
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
                diffs = (loss_pert - loss_orig)
                scores.loc[(kk, slice(None)), foi] = diffs
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

    def dr_from(self, K, B, J, X_eval, y_eval, D=None, sampler=None,
                decorrelator=None, loss=None, nr_runs=10,
                return_perturbed=False, train_allowed=True,
                target='Y', marginalize=False,
                nr_resample_marginalize=5):
        """Computes Relative Feature importance

        Args:
            K: features of interest
            B: baseline features
            J: "from" conditioning set
            X_eval: data to use for resampling and evaluation.
            y_eval: labels for evaluation.
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
        if target not in ['Y', 'Y_hat']:
            raise ValueError('Y and Y_hat are the only valid targets.')

        # if marginalize:
        #     raise NotImplementedError('Marginalization not implemented yet.')

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if decorrelator is None:
            if self._decorrelator_specified():
                decorrelator = self.decorrelator
                logger.debug("Using class specified decorrelator.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logger.debug("Using class specified loss.")

        if D is None:
            D = X_eval.columns

        if not set(K).isdisjoint(set(B)):
            raise ValueError('K and B are not disjoint.')

        # check whether sampler is trained for baseline dropped features
        R = list(set(D) - set(B))
        if not sampler.is_trained(R, J):
            # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(R, J)
                logger.info('Training sampler on {}|{}'.format(R, J))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {}|{}'.format(R, J))
        else:
            txt = '\tCheck passed: Sampler is already trained on'
            txt = txt + '{}|{}'.format(R, J)
            logger.debug(txt)

        if not decorrelator.is_trained(R, J, []):
            # train if allowed, otherwise raise error
            if train_allowed:
                decorrelator.train(R, J, [])
                logger.info('Training decorrelator on {} idp {} |{}'.format(R, J, []))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {} idp {} |{}'.format(R, J, []))
        else:
            txt = '\tCheck passed: decorrelator is already trained on'
            txt = txt + '{} idp {}|{}'.format(R, J, [])
            logger.debug(txt)

        desc = 'DR({} | {} <- {})'.format(K, B, J)

        # initialize array for the perturbed samples
        nr_obs = X_eval.shape[0]
        index = utils.create_multiindex(['sample', 'i'],
                                        [np.arange(nr_runs),
                                         np.arange(nr_obs)])
        # X_fsoi_pert = pd.DataFrame([], index=index)
        # perturbed_foiss = np.zeros((nr_fsoi, nr_runs, nr_obs))

        # sample perturbed versions

        scores = pd.DataFrame([], index=index)
        # lss = np.zeros((nr_fsoi, nr_runs, X_eval.shape[0]))

        for kk in np.arange(0, nr_runs, 1):

            X_R_J = sampler.sample(X_eval, R, J,
                                   num_samples=nr_resample_marginalize)

            index = X_R_J.index

            df_yh = pd.DataFrame(index=index,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])

            for ll in np.arange(0, nr_resample_marginalize, 1):

                X_tilde_baseline = X_eval.copy()
                X_tilde_foreground = X_eval.copy()

                arr_reconstr = X_R_J.loc[ll, :][R].to_numpy()
                X_tilde_foreground[R] = arr_reconstr

                X_R_empty_linked = decorrelator.decorrelate(X_tilde_foreground,
                                                            R, J, [])
                X_tilde_baseline[R] = X_R_empty_linked[R].to_numpy()
                RoK = list(set(R) - set(K))
                X_tilde_foreground[RoK] = X_R_empty_linked[RoK].to_numpy()

                # make sure model can handle it (selection and ordering)
                X_tilde_baseline = X_tilde_baseline[D]
                X_tilde_foreground = X_tilde_foreground[D]

                y_hat_baseline = self.model(X_tilde_baseline)
                y_hat_foreground = self.model(X_tilde_foreground)

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = y_hat_baseline
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = y_hat_foreground

            df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                  'y_hat_foreground': 'float'})
            df_yh = df_yh.mean(level='i')

            # compute difference in observationwise loss
            if target == 'Y':
                loss_baseline = loss(y_eval, df_yh['y_hat_baseline'])
                loss_foreground = loss(y_eval, df_yh['y_hat_foreground'])
                diffs = (loss_baseline - loss_foreground)
                scores.loc[(kk, slice(None)), 'score'] = diffs
            elif target == 'Y_hat':
                diffs = loss(df_yh['y_hat_baseline'],
                             df_yh['y_hat_foreground'])
                scores.loc[(kk, slice(None)), 'score'] = diffs

        # return explanation object
        ex_name = desc
        result = explanation.Explanation(
            self.fsoi, scores,
            ex_name=ex_name)

        if return_perturbed:
            raise NotImplementedError('return_perturbed=True not implemented.')
        else:
            logger.debug('Return explanation object only')
            return result

    def ar_via(self, J, C, K, X_eval, y_eval, D=None, sampler=None,
               decorrelator=None, loss=None, nr_runs=10,
               return_perturbed=False, train_allowed=True,
               target='Y', marginalize=False,
               nr_resample_marginalize=5):
        """Computes Relative Feature importance

        Args:
            K: features of interest
            B: baseline features
            J: "from" conditioning set
            X_eval: data to use for resampling and evaluation.
            y_eval: labels for evaluation.
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
        if target not in ['Y', 'Y_hat']:
            raise ValueError('Y and Y_hat are the only valid targets.')

        # if marginalize:
        #     raise NotImplementedError('Marginalization not implemented yet.')

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if decorrelator is None:
            if self._decorrelator_specified():
                decorrelator = self.decorrelator
                logger.debug("Using class specified decorrelator.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logger.debug("Using class specified loss.")

        if D is None:
            D = X_eval.columns

        if not marginalize:
            nr_resample_marginalize = 1

        if not set(J).isdisjoint(set(C)):
            raise ValueError('J and C are not disjoint.')

        # check whether sampler is trained for baseline dropped features
        R = list(set(D) - set(C))
        R_ = list(set(R) - set(J))
        CuJ = list(set(C).union(J))
        if not sampler.is_trained(R_, CuJ):
            # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(R_, CuJ)
                logger.info('Training sampler on {}|{}'.format(R_, CuJ))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {}|{}'.format(R_, CuJ))
        else:
            txt = '\tCheck passed: Sampler is already trained on'
            txt = txt + '{}|{}'.format(R, J)
            logger.debug(txt)

        if not decorrelator.is_trained(K, J, C):
            # train if allowed, otherwise raise error
            if train_allowed:
                decorrelator.train(K, J, C)
                logger.info('Training decorrelator on {} idp {} |{}'.format(K, J, C))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {} idp {} |{}'.format(K, J, C))
        else:
            txt = '\tCheck passed: decorrelator is already trained on'
            txt = txt + '{} idp {}|{}'.format(K, J, C)
            logger.debug(txt)

        desc = 'AR({} | {} -> {})'.format(J, C, K)

        # initialize array for the perturbed samples
        nr_obs = X_eval.shape[0]
        index = utils.create_multiindex(['sample', 'i'],
                                        [np.arange(nr_runs),
                                         np.arange(nr_obs)])

        scores = pd.DataFrame([], index=index)

        for kk in np.arange(0, nr_runs, 1):

            # sample perturbed versions
            X_R_CuJ2 = None
            X_R_CuJ = sampler.sample(X_eval, R_, CuJ, num_samples=nr_resample_marginalize)
            index = X_R_CuJ.index

            df_yh = pd.DataFrame(index=index,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])

            for ll in np.arange(0, nr_resample_marginalize, 1):

                X_tilde_baseline = X_eval.copy()
                X_tilde_foreground = X_eval.copy()

                arr_reconstr = X_R_CuJ.loc[(ll, slice(None)), R_].to_numpy()
                X_tilde_foreground[R_] = arr_reconstr

                X_R_decorr = decorrelator.decorrelate(X_tilde_foreground, K, J, C)
                arr_decorr = X_R_decorr[R].to_numpy()

                X_tilde_baseline[R] = arr_decorr

                # make sure model can handle it (selection and ordering)
                X_tilde_baseline = X_tilde_baseline[D]
                X_tilde_foreground = X_tilde_foreground[D]

                y_hat_baseline = self.model(X_tilde_baseline)
                y_hat_foreground = self.model(X_tilde_foreground)

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = y_hat_baseline
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = y_hat_foreground

            df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                  'y_hat_foreground': 'float'})
            df_yh = df_yh.mean(level='i')

            # compute difference in observationwise loss
            if target == 'Y':
                loss_baseline = loss(y_eval, df_yh['y_hat_baseline'])
                loss_foreground = loss(y_eval, df_yh['y_hat_foreground'])
                diffs = (loss_baseline - loss_foreground)
                scores.loc[(kk, slice(None)), 'score'] = diffs
            elif target == 'Y_hat':
                diffs = loss(df_yh['y_hat_baseline'],
                             df_yh['y_hat_foreground'])
                scores.loc[(kk, slice(None)), 'score'] = diffs

        # return explanation object
        ex_name = desc
        result = explanation.Explanation(
            self.fsoi, scores,
            ex_name=ex_name)

        if return_perturbed:
            raise NotImplementedError('return_perturbed=True not implemented.')
        else:
            logger.debug('Return explanation object only')
            return result

    def tai(self, X_eval, y_eval, K, D=None, sampler=None, decorrelator=None,
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
        X_eval_tilde_bsln = pd.DataFrame([], index=index_bsln, columns=D)

        # index_rcnstr = utils.create_multiindex(['foi', 'sample', 'i'],
        #                                        [self.fsoi,
        #                                         np.arange(nr_runs),
        #                                         np.arange(nr_obs)])
        # X_eval_tilde_rcnstr = pd.DataFrame([], index=index_rcnstr, columns=D)

        # create empty scores data frame
        index_scores = utils.create_multiindex(['sample', 'i'],
                                               [np.arange(nr_runs),
                                                np.arange(nr_obs)])
        scores = pd.DataFrame([], index=index_scores)

        # sample baseline X^\emptyset
        X_eval_tilde_bsln = sampler.sample(X_eval, D, [], num_samples=nr_runs)
        lss_baseline = []
        for kk in np.arange(nr_runs):
            X_bl = X_eval_tilde_bsln.loc[(kk, slice(None)), D]
            l_pb = loss(y_eval, self.model(X_bl))
            lss_baseline.append(l_pb)

        # sample perturbed versions
        for foi in self.fsoi:
            # X^foi
            sample = sampler.sample(X_eval, D, [foi], num_samples=nr_runs)
            for kk in np.arange(nr_runs):
                # X^\emptyset,linked
                sample_decorr = decorrelator.decorrelate(sample.loc[kk, :],
                                                         K, [foi], [])
                sd_np = sample_decorr[D].to_numpy()
                # X_eval_tilde_rcnstr.loc[(foi, kk, slice(None)), D] = sd_np
                # sd = X_eval_tilde_rcnstr.loc[(foi, kk, slice(None)), D]
                l_rc = loss(y_eval, self.model(sd_np))
                scores.loc[(kk, slice(None)), foi] = lss_baseline[kk] - l_rc

        if ex_name is None:
            ex_name = 'Unknown tai'

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
                      nr_orderings=None, nr_orderings_sage=None,
                      nr_runs=3, show_pbar=True,
                      approx=math.sqrt, save_orderings=True,
                      sage_partial_ordering=None, orderings=None,
                      target='Y', **kwargs):
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
        if orderings is None:
            if nr_orderings is None:
                nr_unique = utils.nr_unique_perm(partial_ordering)
                if approx is not None:
                    nr_orderings = math.floor(approx(nr_unique))
                else:
                    nr_orderings = nr_unique
        else:
            nr_orderings = orderings.shape[0]

        logger.info('#orderings: {}'.format(nr_orderings))

        if imp_type not in ['tdi', 'tai', 'sage']:
            raise ValueError('Only tdi, tai and sage '
                             'implemented for imp_type.')

        if imp_type == 'sage' and sage_partial_ordering is None:
            raise ValueError('Please specify a sage ordering.')

        if target not in ['Y', 'Y_hat']:
            raise ValueError('Only Y and Y_hat implemented as target.')

        if nr_orderings_sage is None:
            nr_orderings_sage = nr_orderings

        # values (nr_perm, nr_runs, nr_components, nr_fsoi)
        # components are: (elements of ordering,..., remainder)
        # elements of ordering are sorted in increasing order

        component_names = np.unique(utils.flatten(partial_ordering))
        component_names = list(component_names)
        component_names.insert(0, 'remainder')
        component_names.insert(0, 'total')
        nr_components = len(component_names)

        nr_orderings_saved = 1
        if save_orderings:
            nr_orderings_saved = nr_orderings

        def rescale_fi_vals(fi_vals_old, fi_vals, component, ordering_nr):
            '''
            Necessary to compute the running mean when not saving
            every ordering.
            Assuming fi_vals are np.array, ordering_nr
            run number in range(0, nr_orderings)
            df the decomposition dataframe
            '''
            fi_vals_old_scaled = fi_vals_old * (ordering - 1) / ordering
            fi_vals_scaled = fi_vals / ordering
            fi_vals_new = fi_vals_old_scaled + fi_vals_scaled
            return fi_vals_new

        # create dataframe for computation results
        index = utils.create_multiindex(['component', 'ordering', 'sample'],
                                        [component_names,
                                         np.arange(nr_orderings_saved),
                                         np.arange(nr_runs)])
        arr = np.zeros((nr_components * nr_orderings_saved * nr_runs,
                        len(self.fsoi)))
        decomposition = pd.DataFrame(arr, index=index, columns=self.fsoi)
        # values = np.zeros((nr_orderings, nr_runs, nr_components, len(fsoi)))
        orderings_sampled = pd.DataFrame(index=np.arange(nr_orderings),
                                         columns=['ordering'])

        # in the first sage call an ordering object is returned
        # that is then fed to sage again
        # to ensure that always the same orderings are used
        # for the computation
        # allow a more reliable approximation
        sage_orderings = None

        if show_pbar:
            mgr = enlighten.get_manager()
            pbar = mgr.counter(total=nr_orderings, desc='decomposition',
                               unit='orderings')

        # ordering history helps to avoid duplicate orderings
        ord_hist = None
        for kk in np.arange(nr_orderings):
            if show_pbar:
                pbar.update()

            ordering = None
            if orderings is None:
                ordering, ord_hist = utils.sample_partial(partial_ordering,
                                                          ord_hist)
                logging.info('Ordering : {}'.format(ordering))
                orderings_sampled.loc[kk, 'ordering'] = ordering
            else:
                ordering = orderings.loc[kk, 'ordering']

            # total values
            expl = None
            if imp_type == 'tdi':
                expl = self.tdi(X_eval, y_eval, [],
                                nr_runs=nr_runs, **kwargs)
            elif imp_type == 'tai':
                expl = self.tai(X_eval, y_eval, [],
                                nr_runs=nr_runs, **kwargs)
            elif imp_type == 'sage':
                tupl = self.sage(X_eval, y_eval, partial_ordering,
                                 nr_orderings=nr_orderings_sage,
                                 nr_runs=nr_runs, target=target,
                                 G=X_eval.columns, orderings=sage_orderings,
                                 **kwargs)
                expl, sage_orderings = tupl
            fi_vals = expl.fi_vals().to_numpy()

            # store total values
            if save_orderings:
                decomposition.loc['total', kk, :] = fi_vals
            else:
                fi_old = decomposition.loc['total', 0, :].to_numpy()
                fi_vals = rescale_fi_vals(fi_old, fi_vals, 'total', kk)
                decomposition.loc['total', 0, :] = fi_vals

            previous = fi_vals
            current = None

            for jj in np.arange(1, len(ordering) + 1):
                # get current new variable and respective set
                current_ix = ordering[jj - 1]
                G = ordering[:jj]

                # compute and store feature importance
                expl = None
                if imp_type == 'tdi':
                    expl = self.tdi(X_eval, y_eval, G,
                                    nr_runs=nr_runs, **kwargs)
                elif imp_type == 'tai':
                    expl = self.tai(X_eval, y_eval, G,
                                    nr_runs=nr_runs, **kwargs)
                elif imp_type == 'sage':
                    G_ = list(set(X_eval.columns) - set(G))
                    tupl = self.sage(X_eval, y_eval, partial_ordering,
                                     nr_orderings=nr_orderings_sage,
                                     nr_runs=nr_runs, target=target,
                                     G=G_, orderings=sage_orderings,
                                     **kwargs)
                    expl, sage_orderings = tupl

                current = expl.fi_vals().to_numpy()

                fi_vals = None
                # compute difference
                fi_vals = previous - current

                # store result
                if save_orderings:
                    decomposition.loc[current_ix, kk, :] = fi_vals
                else:
                    fi_old = decomposition.loc[current_ix, 0, :].to_numpy()
                    fi_vals = rescale_fi_vals(fi_old, fi_vals, current_ix, kk)
                    decomposition.loc[current_ix, 0, :] = fi_vals

                previous = current

            # store remainder
            if save_orderings:
                decomposition.loc['remainder', kk, :] = current
            else:
                fi_old = decomposition.loc['remainder', 0, :].to_numpy()
                fi_vals = rescale_fi_vals(fi_old, current,
                                          'remainder', kk)
                decomposition.loc['remainder', 0, :] = fi_vals

        if orderings is None:
            orderings = orderings_sampled
        ex = decomposition_ex.DecompositionExplanation(self.fsoi,
                                                       decomposition,
                                                       ex_name=None)
        return ex, orderings

    def sage(self, X_test, y_test, partial_ordering,
             nr_orderings=None, approx=math.sqrt,
             save_orderings=True, nr_runs=10, sampler=None,
             loss=None, train_allowed=True, D=None,
             nr_resample_marginalize=10, target='Y',
             G=None, method='associative',
             marginalize=True, orderings=None, **kwargs):
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
        if G is None:
            G = X_test.columns

        if X_test.shape[1] != len(self.fsoi):
            logger.debug('self.fsoi: {}'.format(self.fsoi))
            logger.debug('#features in model: {}'.format(X_test.shape[1]))
            raise RuntimeError('self.fsoi is not identical to all features')

        if method not in ['associative', 'direct']:
            raise ValueError('only methods associative or direct implemented')

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logger.debug("Using class specified loss.")

        if orderings is None:
            if nr_orderings is None:
                nr_unique = utils.nr_unique_perm(partial_ordering)
                if approx is not None:
                    nr_orderings = math.floor(approx(nr_unique))
                else:
                    nr_orderings = nr_unique
        else:
            nr_orderings = orderings.shape[0]

        if D is None:
            D = X_test.columns

        nr_orderings_saved = 1
        if save_orderings:
            nr_orderings_saved = nr_orderings

        # create dataframe for computation results
        index = utils.create_multiindex(['ordering', 'sample', 'id'],
                                        [np.arange(nr_orderings_saved),
                                         np.arange(nr_runs),
                                         np.arange(X_test.shape[0])])
        arr = np.zeros((nr_orderings_saved * nr_runs * X_test.shape[0],
                        len(self.fsoi)))
        scores = pd.DataFrame(arr, index=index, columns=self.fsoi)

        orderings_sampled = None
        if orderings is None:
            orderings_sampled = pd.DataFrame(index=np.arange(nr_orderings),
                                             columns=['ordering'])

        # lss = np.zeros(
        #     (len(self.fsoi), nr_runs, X_test.shape[0], nr_orderings))
        # ord hist helps to avoid duplicate histories
        ord_hist = None
        for ii in range(nr_orderings):
            ordering = None
            if orderings is None:
                ordering, ord_hist = utils.sample_partial(partial_ordering,
                                                          ord_hist)
                orderings_sampled.loc[ii, 'ordering'] = ordering
            else:
                ordering = orderings.loc[ii, 'ordering']

            logging.info('Ordering : {}'.format(ordering))

            for jj in np.arange(1, len(ordering), 1):
                # TODO: check if jj in features for which the score shall
                # TODO: be computed
                # compute change in performance
                # by entering the respective feature
                # store the result in the right place
                # validate training of sampler
                J, C = [ordering[jj - 1]], ordering[:jj - 1]
                if method == 'associative':
                    ex = self.ar_via(J, C, G, X_test, y_test,
                                     target=target, marginalize=marginalize,
                                     nr_runs=nr_runs,
                                     nr_resample_marginalize=nr_resample_marginalize,
                                     **kwargs)
                elif method == 'direct':
                    ex = self.dr_from(J, C, G, X_test, y_test,
                                      target=target, marginalize=marginalize,
                                      nr_runs=nr_runs,
                                      nr_resample_marginalize=nr_resample_marginalize,
                                      **kwargs)
                scores_arr = ex.scores.to_numpy()
                scores.loc[(ii, slice(None), slice(None)), ordering[jj - 1]] = scores_arr

        result = explanation.Explanation(
            self.fsoi, scores,
            ex_name='SAGE')

        if orderings is None:
            orderings = orderings_sampled

        return result, orderings
