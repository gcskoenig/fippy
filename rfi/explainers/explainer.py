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

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


class Explainer:
    """Implements a number of feature importance algorithms

    Default samplers or loss function can be defined.

    Attributes:
        model: Model or predict function.
        fsoi: Features of interest. Columnnames.
        X_train: Training data for resampling. Pandas dataframe.
        sampler: default sampler.
        decorrelator: default decorrelator
        loss: default loss.
    """
    # TODO make sampler (and decorrelator?) normal arugments (no keyword arguments)
    def __init__(self, model, fsoi, X_train, sampler=None, decorrelator=None,
                 loss=None, ydim=None):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        self.model = model
        self.fsoi = fsoi  # now column names, not indexes
        self.X_train = X_train
        self.sampler = sampler
        self.decorrelator = decorrelator
        self.loss = loss
        self.ydim = ydim
        # check whether feature set is valid
        self._valid_fset(self.fsoi)
        # detect ydim if not specified
        self._detect_ydim()

    def _valid_fset(self, fset):
        """Checks whether fset is subset of features in data"""
        fset = set(list(fset))
        if not fset.issubset(self.X_train.columns):
            raise ValueError("Feature set does not match "
                             "the dataframe's column names.")
        else:
            return True

    def _sampler_specified(self):
        """Checks whether a sampler was specified"""
        if self.sampler is None:
            raise ValueError("Sampler has not been specified.")
        else:
            return True

    def _decorrelator_specified(self):
        """Checks whether a decorrelator was specified"""
        if self.decorrelator is None:
            raise ValueError("Decorrelator has not been specified.")
        else:
            return True

    def _loss_specified(self):
        """Checks whether a loss was specified"""
        if self.loss is None:
            raise ValueError("Loss has not been specified.")
        else:
            return True

    def _detect_ydim(self):
        """Detects dimension of model prediction if not provided"""
        if self.ydim is None:
            features = np.array(self.X_train.iloc[0]).reshape(1, -1)
            self.ydim = len(self.model(features)[0])
        else:
            return True

    # Elementary Feature Importance Techniques

    def di_from(self, K, B, J, X_eval, y_eval, D=None, sampler=None,
                decorrelator=None, loss=None, nr_runs=10,
                return_perturbed=False, train_allowed=True,
                target='Y', marginalize=False,
                nr_resample_marginalize=5):
        """Computes the direct importance of features K given features B
        that can be explained by variables J, short DI(X_K|X_B <- X_J)

        Args:
            K: features of interest
            B: baseline features
            J: "from" conditioning set
            X_eval: data to use for resampling and evaluation.
            y_eval: labels for evaluation.
            D: features (variables used by the predictive model)
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            decorrelator: choice of decorrelator. Default None. Will throw an error
              when decorrelator is None and self.decorrelator is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.
            nr_runs: how often the experiment shall be run
            return_perturbed: whether the sampled perturbed versions
                shall be returned
            train_allowed: whether the explainer is allowed to train
                the sampler
            target: 'Y' or 'Y_hat', indicates whether loss is taken with respect
                to Y or with respect to the prediction on unperturbed data
            marginalize: whether a marginalized prediction function is used
            nr_resample_marginalize: how many samples are drawn for the computation
                of the marginalization expectation

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of
                interest if return_perturbed
        """
        if target not in ['Y', 'Y_hat']:
            raise ValueError('Y and Y_hat are the only valid targets.')

        if not marginalize:
            # if we take expecation over one sample that coincides with taking only one sample
            nr_resample_marginalize = 1

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

        # check whether sampler is trained for the X_R|X_J
        # where R is the set of features not in B
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

        # description of the computation
        desc = 'DR({} | {} <- {})'.format(K, B, J)

        # initialize array for the perturbed samples
        nr_obs = X_eval.shape[0]
        index = utils.create_multiindex(['sample', 'i'],
                                        [np.arange(nr_runs),
                                         np.arange(nr_obs)])

        # intitialize dataframe for the loss scores
        scores = pd.DataFrame([], index=index)

        for kk in np.arange(0, nr_runs, 1):

            # sample all features not in B given J
            X_R_J = sampler.sample(X_eval, R, J,
                                   num_samples=nr_resample_marginalize)
            index = X_R_J.index

            # initialize array for predictions before and after perturbation
            df_yh = pd.DataFrame(index=index,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])

            # sample remaining features nr_resample_marginalize times
            # predict on the respective perturbation datasets
            for ll in np.arange(0, nr_resample_marginalize, 1):

                # initialize baseline and foreground sample arrays
                X_tilde_baseline = X_eval.copy()
                X_tilde_foreground = X_eval.copy()

                arr_reconstr = X_R_J.loc[ll, :][R].to_numpy()
                X_tilde_foreground[R] = arr_reconstr

                # decorrelate all features not in B given J such that the sample
                # is independent of X_J (given X_\emptyset)
                X_R_empty_linked = decorrelator.decorrelate(X_tilde_foreground,
                                                            R, J, [])

                # in the baseline data, assign the independent sample to
                # all remaining features
                X_tilde_baseline[R] = X_R_empty_linked[R].to_numpy()

                # in the foreground data, assign the independent sample
                # to all features in R but K
                RoK = list(set(R) - set(K))
                X_tilde_foreground[RoK] = X_R_empty_linked[RoK].to_numpy()

                # make sure data is formatted as the model expects (selection and ordering)
                X_tilde_baseline = X_tilde_baseline[D]
                X_tilde_foreground = X_tilde_foreground[D]

                y_hat_baseline = self.model(X_tilde_baseline)
                y_hat_foreground = self.model(X_tilde_foreground)

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = np.array(y_hat_baseline)
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = np.array(y_hat_foreground)

            # covert types of prediction dataframe
            df_yh = df_yh.astype({'y_hat_baseline': 'float64',
                                  'y_hat_foreground': 'float64'})
            # marginalize predictions
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

    def ai_via(self, J, C, K, X_eval, y_eval, D=None, sampler=None,
               decorrelator=None, loss=None, nr_runs=10,
               return_perturbed=False, train_allowed=True,
               target='Y', marginalize=False,
               nr_resample_marginalize=5):
        """Computes AI via

        Args:
            J: variables of interest
            C: baseline variables/coalition variables
            K: "via" feature set
            X_eval: data to use for resampling and evaluation.
            y_eval: labels for evaluation.
            D: features, used by the predictive model
            sampler: choice of sampler. Default None. Will throw an error
                when sampler is None and self.sampler is None as well.
            decorrelator: choice of decorrelator. Raises error when
                both decorrelator argument and self.decorrelator are None.
            loss: choice of loss. Default None. Will throw an Error when
                both loss and self.loss are None.
            nr_runs: how often the experiment shall be run
            return_perturbed: whether the sampled perturbed versions
                shall be returned
            train_allowed: whether the explainer is allowed to train
                the sampler
            target: whether loss shall be computed against Y or Y_hat
            marginalize: whether marginalization shall be performed
            nr_resample_marginalize: sample size for marginilization
                computation

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of
                interest if return_perturbed
        """
        if target not in ['Y', 'Y_hat']:
            raise ValueError('Y and Y_hat are the only valid targets.')

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

        # sampler trained for baseline non-coalition variables R_?
        R = list(set(D) - set(C))  # background non-coalition variables
        R_ = list(set(R) - set(J))  # foreground non-coalition
        CuJ = list(set(C).union(J))  # foreground coalition variables
        if not sampler.is_trained(R_, CuJ):  # sampler for foreground non-coalition
            # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(R_, CuJ)
                logger.info('Training sampler on {}|{}'.format(R_, CuJ))
            else:
                raise RuntimeError('Sampler is not trained on {}|{}'.format(R_, CuJ))
        else:
            txt = '\tCheck passed: Sampler is already trained on'
            txt = txt + '{}|{}'.format(R, J)
            logger.debug(txt)

        # decorrelator for remainder R trained?
        if not decorrelator.is_trained(R, J, C):
            # train if allowed, otherwise raise error
            if train_allowed:
                decorrelator.train(R, J, C)
                logger.info('Training decorrelator on {} idp {} |{}'.format(R, J, C))
            else:
                raise RuntimeError(
                    'Sampler is not trained on {} idp {} |{}'.format(R, J, C))
        else:
            txt = '\tCheck passed: decorrelator is already trained on'
            txt = txt + '{} idp {}|{}'.format(R, J, C)
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
            X_R_CuJ = sampler.sample(X_eval, R_, CuJ, num_samples=nr_resample_marginalize)
            index = X_R_CuJ.index
            # TODO (cl) other index?
            index_yh = utils.create_multiindex(['sample', 'i', 'target'],
                                               [np.arange(nr_resample_marginalize),
                                                np.arange(X_eval.shape[0]),
                                                np.arange(self.ydim)])

            df_yh = pd.DataFrame(index=index_yh,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])  # TODO (cl) dtype=object for lists?

            # create foreground and background samples and make predictions
            for ll in np.arange(0, nr_resample_marginalize, 1):

                X_tilde_baseline = X_eval.copy()
                X_tilde_foreground = X_eval.copy()

                # copy ll-th sample for R_ variables
                arr_reconstr = X_R_CuJ.loc[(ll, slice(None)), R_].to_numpy()
                X_tilde_foreground[R_] = arr_reconstr

                # decorellate X_R and copy
                # TODO would X_tilde_foreground[R] be ok as well?
                X_R_decorr = decorrelator.decorrelate(X_tilde_foreground, R, J, C)
                arr_decorr = X_R_decorr[R].to_numpy()

                # TODO make use of features K to selectively update
                #  background to foreground (only features K ar updated)
                X_tilde_baseline[R] = arr_decorr
                X_tilde_foreground_partial = X_tilde_baseline.copy()
                X_tilde_foreground_partial[K] = X_tilde_foreground[K].to_numpy()

                # make sure model can handle it (selection and ordering)
                X_tilde_baseline = X_tilde_baseline[D]
                X_tilde_foreground_partial = X_tilde_foreground_partial[D]

                # create and store prediction   # TODO (c)l try dtype=object for ydim > 1?
                y_hat_baseline = self.model(X_tilde_baseline)
                # print("baseline pred dtype  is", y_hat_baseline.dtype)
                # print("baseline pred 0 is type", type(y_hat_baseline[0]))
                # print("baseline pred 0 is", y_hat_baseline[0])

                y_hat_foreground = self.model(X_tilde_foreground_partial)

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = np.array(y_hat_baseline)
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = np.array(y_hat_foreground)

            # convert and aggregate predictions
            df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                  'y_hat_foreground': 'float'})
            df_yh = df_yh.mean(level='i')

            # compute difference in observation-wise loss
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

    def dis_from_ordering(self, ordering, J, X_eval, y_eval, **kwargs):
        """Computes DI from X_J for every feature and the respective coalition as
        specified by the ordering.

        Args:
            ordering: tuple with ordered columnnames
            J: "from" set (invariant for all features)
            X_eval: test data
            y_eval: test label
            **kwargs: keyword arguments that are passed di_from
        """
        # TODO add fsoi argument to make more efficient in case we are only interested in a subset

        # compute di_from for the first element
        K = [ordering[0]]
        B = []
        ex_first = self.di_from(K, B, J, X_eval, y_eval, **kwargs)

        # use scores to initialize dis scores
        index = ex_first.scores.index
        scores = pd.DataFrame([], index=index)

        scores.loc[(slice(None), slice(None)), K[0]] = ex_first.scores.to_numpy()

        for jj in np.arange(1, len(ordering)):
            K, B = [ordering[jj]], ordering[:jj]

            ex = self.di_from(K, B, J, X_eval, y_eval, **kwargs)
            scores_arr = ex.scores.to_numpy()
            scores.loc[(slice(None), slice(None)), K[0]] = scores_arr

        result = explanation.Explanation(self.fsoi, scores, ex_name='dis_from_ordering')
        return result

    def ais_via_ordering(self, ordering, K, X_eval, y_eval, **kwargs):
        """Computes AI via X_K for every variable and the respective context coalition
        as specified by the ordering. I.e for ordering=(X_1, X_2, X_3), the method returns
        AI(X_1 via X_K), AI(X_2|X_1 via X_K) and AI(X_3|X_1, X_2 via X_K).

        Args:
            ordering: tuple with ordered columnnames
            K: specifies via feature X_K (invariant for all evaluated variables X_j)
            X_eval: evaluation data
            y_eval: evaluation label
            **kwargs: keyword arguments that are passed to ai_via
        """
        # TODO add fsoi keyword argument in case we are only interested in a subset

        # compute ai_from for the first element
        J = [ordering[0]]
        C = []
        ex_first = self.ai_via(J, C, K, X_eval, y_eval, **kwargs)

        # use scores to initialize dis scores
        index = ex_first.scores.index
        scores = pd.DataFrame([], index=index)

        scores.loc[(slice(None), slice(None)), J[0]] = ex_first.scores.to_numpy()

        for jj in np.arange(1, len(ordering)):
            J, C = [ordering[jj]], ordering[:jj]

            ex = self.ai_via(J, C, K, X_eval, y_eval, **kwargs)
            scores_arr = ex.scores.to_numpy()
            scores.loc[(slice(None), slice(None)), J[0]] = scores_arr

        result = explanation.Explanation(self.fsoi, scores, ex_name='ais_via_ordering')
        return result

    def dis_from_baselinefunc(self, J, X_eval, y_eval, fsoi=None, D=None, baseline='empty',
                              baselinefunc=None, **kwargs):
        """Computes DI from X_J for every feature given a feature-dependent baseline. For example,
        the baseline can be specified to be empty or the respective remainder B:=D without K. The baseline
        can be specified in a flexible manner by providing a baselinefunc.

        Args:
            J: indices specifying the "from" set X_J
            X_eval: evaluation data
            y_eval: evalaution labels
            fsoi: features of interest, overrides self.fsoi if not None
            baseline: either 'empty' (B = emptyset) or 'remainder' (B = D without K)
            baselinefunc: function that takes lists K and D as arguments and returns the respective
                baseline set B (list). Overrides baseline
        """

        # set features of interest
        if fsoi is None:
            if self.fsoi is None:
                fsoi = self.X_train.columns
            else:
                fsoi = self.fsoi

        # set set of all features
        if D is None:
            D = self.X_train.columns

        # find baseline function
        if baselinefunc is None:
            if baseline == 'empty':
                def helper(K, D):
                    return []
                baselinefunc = helper
            elif baseline == 'remainder':
                def helper(K, D):
                    res = set(D) - set(K)
                    return list(res)
                baselinefunc = helper
            else:
                raise ValueError('No baselinefunc specified and baseline neither empty nor remainder.')

        # compute di_from for the first element
        K = [fsoi[0]]
        B = baselinefunc(K, D)
        ex_first = self.di_from(K, B, J, X_eval, y_eval, **kwargs)

        # use scores to initialize dis scores
        index = ex_first.scores.index
        scores = pd.DataFrame([], index=index)

        scores.loc[(slice(None), slice(None)), K[0]] = ex_first.scores.to_numpy()

        # iterate over the remaining features
        for jj in np.arange(1, len(fsoi)):
            K = [fsoi[jj]]
            B = baselinefunc(K, D)
            ex = self.di_from(K, B, J, X_eval, y_eval, **kwargs)
            scores_arr = ex.scores.to_numpy()
            scores.loc[(slice(None), slice(None)), K[0]] = scores_arr

        result = explanation.Explanation(self.fsoi, scores, ex_name='dis_from_fixed')
        return result

    def ais_via_contextfunc(self, K, X_eval, y_eval, fsoi=None, D=None, context='empty', contextfunc=None, **kwargs):
        """Computes AI via X_K for every variable given a variable-dependent context. For example,
        the context can be specified to be a function of the variable j and the set of all variables,
        e.g. C=D without j.

        Args:
            K: indices specifying the "via" set X_K
            X_eval: evaluation data
            y_Eval: evaluation labels
            context: either 'empty' or 'remainder (C=D without j)
            contextfunc: function that takes j and D as arguments and returns the respective baseline
                set C=f(j,D). Overrides baseline argument.
        """
        # set features of interest
        if fsoi is None:
            if self.fsoi is None:
                fsoi = self.X_train.columns
            else:
                fsoi = self.fsoi

        # set set of all features
        if D is None:
            D = self.X_train.columns

        # find baseline function
        if contextfunc is None:
            if context == 'empty':
                def helper(J, D):
                    return []
                contextfunc = helper
            elif context == 'remainder':
                def helper(J, D):
                    res = set(D) - set(J)
                    return list(res)
                contextfunc = helper
            else:
                raise ValueError('No contextfunc specified and context neither empty nor remainder.')

        # compute ai_via for the first element
        J = [fsoi[0]]
        C = contextfunc(J, D)
        ex_first = self.ai_via(J, C, K, X_eval, y_eval, **kwargs)

        # use scores to initialize ais scores
        index = ex_first.scores.index
        scores = pd.DataFrame([], index=index)

        scores.loc[(slice(None), slice(None)), J[0]] = ex_first.scores.to_numpy()

        # iterate over the remaining variables
        for jj in np.arange(1, len(fsoi)):
            J = [fsoi[jj]]
            C = contextfunc(J, D)
            ex = self.ai_via(J, C, K, X_eval, y_eval, **kwargs)
            scores_arr = ex.scores.to_numpy()
            scores.loc[(slice(None), slice(None)), J[0]] = scores_arr

        result = explanation.Explanation(self.fsoi, scores, ex_name='ais_via_fixed')
        return result

    # Advanced Feature Importance

    def sage(self, X_eval, y_eval, partial_ordering,
             target='Y', method='associative', G=None, marginalize=True,
             nr_orderings=None, approx=math.sqrt, convergence=False,
             nr_runs=10, nr_resample_marginalize=10,
             sampler=None, loss=None, fsoi=None, orderings=None,
             save_orderings=True, **kwargs):
        """
        Compute Shapley Additive Global Importance values.
        Args:
            X_test: pandas df to use for resampling and evaluation.
            y_test: labels for evaluation.
            partial_ordering: tuple of items or lists that define a
                (partial) ordering, no ordering: ([X_1, X_2, ...])
            target: whether loss should be computed with respect to 'Y' or 'Y_hat'
            method: whether conditional sampling ('associative') or marginal sampling ('direct')
                shall be used
            G: if method='associative', G specifies the via features,
                if method='direct', G specifies the from variables,
                if G=None it is set to X_test.columns
            marginalize: whether the marginalized or the non-marginalized
                prediction function shall be used
            nr_orderings: number of orderings that shall be evaluated
            nr_runs: how often each value function shall be computed
            nr_resample_marginalize: How many samples shall be used for the
                marginalization
            approx: if nr_orderings=None, approx determines the number of
                orderings w.r.t to the all possible orderings
            convergence: Whether convergence detection shall be used
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
            loss: choice of loss. Default None. Will throw an Error when
              both loss and self.loss are None.
            fsoi: specifies features of interest (for which SAGE values are computed).
                  By default set to X.columns (if fsoi is None and self.fsoi is None).
            orderings: If specified the provided orderings are used for the
                computation. overrides partial ordering, but does not override
                nr_orderings
            save_orderings: whether the aggregates scores or the score for every
                ordering shall be returned
        Returns:
            result: an explanation object containing the respective
                pairwise lossdifferences with shape
                (nr_fsoi, nr_runs, nr_obs, nr_orderings)
            orderings(optional): if save_orderings=True, an array containing the respective
                orderings is returned
        """
        if G is None:
            G = X_eval.columns

        if X_eval.shape[1] != len(self.fsoi):
            # TODO: update to check whether column names match.
            logger.debug('self.fsoi: {}'.format(self.fsoi))
            logger.debug('#features in model: {}'.format(X_eval.shape[1]))
            raise RuntimeError('self.fsoi is not identical to all features')

        if method not in ['associative', 'direct']:
            raise ValueError('only methods associative or direct implemented')

        if convergence:
            raise NotImplementedError('Convergence detection has not been implemented yet.')

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

        if fsoi is None:
            if self.fsoi is not None:
                fsoi = self.fsoi
            else:
                fsoi = X_eval.columns

        nr_orderings_saved = 1
        if save_orderings:
            nr_orderings_saved = nr_orderings

        # create dataframe for computation results
        index = utils.create_multiindex(['ordering', 'sample', 'i'],
                                        [np.arange(nr_orderings_saved),
                                         np.arange(nr_runs),
                                         np.arange(X_eval.shape[0])])
        scores = pd.DataFrame([], index=index)

        orderings_sampled = None
        if orderings is None:
            orderings_sampled = pd.DataFrame(index=np.arange(nr_orderings),
                                             columns=['ordering'])

        # lss = np.zeros(
        #     (len(self.fsoi), nr_runs, X_eval.shape[0], nr_orderings))
        # ord hist helps to avoid duplicate histories
        ord_hist = None
        for ii in range(nr_orderings):
            # TODO(gcsk,cph): convergence detection like in SAGE paper
            #  see https://github.com/iancovert/sage/blob/master/sage/permutation_estimator.py
            ordering = None
            if orderings is None:
                ordering, ord_hist = utils.sample_partial(partial_ordering,
                                                          ord_hist)
                orderings_sampled.loc[ii, 'ordering'] = ordering
            else:
                ordering = orderings.loc[ii, 'ordering']

            logging.info('Ordering : {}'.format(ordering))

            # compute scores for the ordering
            ex = None
            if method == 'associative':
                ex = self.ais_via_ordering(ordering, G, X_eval, y_eval,
                                           target=target, marginalize=marginalize,
                                           nr_runs=nr_runs, nr_resample_marginalize=nr_resample_marginalize,
                                           **kwargs)
            elif method == 'direct':
                ex = self.dis_from_baselinefunc(ordering, G, X_eval, y_eval,
                                                target=target, marginalize=marginalize,
                                                nr_runs=nr_runs, nr_resample_marginalize=nr_resample_marginalize,
                                                **kwargs)

            scores_arr = ex.scores[fsoi].to_numpy()
            scores.loc[(ii, slice(None), slice(None)), fsoi] = scores_arr

        result = explanation.Explanation(fsoi, scores, ex_name='SAGE')

        if orderings is None:
            orderings = orderings_sampled

        if save_orderings:
            return result, orderings
        else:
            return result

    # Decompositions

    def decomposition(self, imp_type, fsoi, partial_ordering, X_eval, y_eval,
                      nr_orderings=None, nr_orderings_sage=None,
                      nr_runs=3, show_pbar=True,
                      approx=math.sqrt, save_orderings=True,
                      sage_partial_ordering=None, orderings=None,
                      target='Y', D=None, **kwargs):
        """
        Given a partial ordering, this code allows to decompose
        feature importance or feature association for a given set of
        features into its respective indirect or direct components.

        Args:
            imp_type: Either 'associative', 'direct' or 'sage'
            fsoi: features, for which the importance scores (of type imp_type)
                are to be decomposed.
            partial_ordering: partial ordering for the decomposition
                of the form (1, 2, [3, 4, 5], 6) where the ordering
                within a tuple is fixed and within a list may be
                permuted.
            X_eval: test data
            y_eval: test labels
            nr_orderings: number of total orderings to sample
                (given the partial) ordering
            nr_orderings_sage: if sage is to be decomposed,
                the number of orderings for sage can be passed seperately
            nr_runs: number of runs for each feature importance score
                computation
            show_pbar: whether progress bar shall be shown
            approx: nr_orderings is set to approx(#unique_orderings) (if nr_orderings not specified)
            save_orderings: whether every ordering shall be stored
            sage_partial_ordering: if desired a separate partial ordering can be passed to sage
            orderings: orderings can be passed as well, overriding the sampling from partial_ordering
            target: either 'Y_hat' or Y
            **kwargs: are passed to the respective importance technique

        Returns:
            dex: Decomposition Explanation object
            orderings: orderings that were provided/used
        """
        if D is None:
            D = X_eval.columns

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

        if imp_type not in ['direct', 'associative', 'sage']:
            raise ValueError('Only direct, associative and sage '
                             'implemented for imp_type.')

        if imp_type == 'sage' and sage_partial_ordering is None:
            raise ValueError('Please specify a sage ordering.')

        if target not in ['Y', 'Y_hat']:
            raise ValueError('Only Y and Y_hat implemented as target.')

        if nr_orderings_sage is None:
            nr_orderings_sage = nr_orderings

        # TODO adapt to account for fsoi
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
            # TODO leverage fsoi argument in the function calls
            # TODO fix dependency on tdi/tai? remove tdi/tai or leave in the code?
            #  it should work with viafrom as well, right?
            if imp_type == 'direct':
                expl = self.dis_from_baselinefunc(D, X_eval, y_eval, baseline='remainder', fsoi=fsoi,
                                                  nr_runs=nr_runs, **kwargs)
            elif imp_type == 'associative':
                expl = self.ais_via_contextfunc(D, X_eval, y_eval, context='empty', fsoi=fsoi,
                                                nr_runs=nr_runs, **kwargs)
            elif imp_type == 'sage':
                tupl = self.sage(X_eval, y_eval, partial_ordering,
                                 nr_orderings=nr_orderings_sage,
                                 nr_runs=nr_runs, target=target,
                                 G=D, orderings=sage_orderings,
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

            G = list(ordering)

            while len(G) > 0:
                # get current new variable and respective set

                current_ix = G.pop(0)

                # compute and store feature importance
                expl = None
                if imp_type == 'direct':
                    expl = self.dis_from_baselinefunc(G, X_eval, y_eval, baseline='remainder', fsoi=fsoi,
                                                      nr_runs=nr_runs, **kwargs)
                elif imp_type == 'associative':
                    expl = self.ais_via_contextfunc(G, X_eval, y_eval, context='empty', fsoi=fsoi,
                                                    nr_runs=nr_runs, **kwargs)
                elif imp_type == 'sage':
                    tupl = self.sage(X_eval, y_eval, partial_ordering,
                                     nr_orderings=nr_orderings_sage,
                                     nr_runs=nr_runs, target=target,
                                     G=G, orderings=sage_orderings,
                                     **kwargs)
                    expl, sage_orderings = tupl

                current = expl.fi_vals().to_numpy()

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
