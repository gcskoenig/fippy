"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""

import numpy as np
import pandas as pd
import fippy.utils as utils
import fippy.explanation as explanation
import logging
import enlighten  # TODO add to requirements
import math
from fippy.explainers.utils import detect_conv
from fippy.samplers import Sampler

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


class Explainer:
    """Implements a number of feature importance algorithms.
    Default conditional samplers, training data, and loss can be specified.

    Attributes:
        model: Model or predict function.
        X_train: Data used to train the sampler / to resample permutations from.
        loss: default loss. None if not specified.
        sampler: default sampler. None if not specified.
        fsoi: Features of interest. All columns of X_train if not specified.
        encoder: specifies encoder to use for encoding categorical data
    """
    def __init__(self, predict, X_train,
                 loss=None,
                 sampler=None,
                 encoder=None, fsoi=None):
        """Inits Explainer with prediction function, training data, and optionally
        loss, sampler, encoder, and fsoi.
        
        Args:
            predict: Model or predict function.
            X_train: Data used to train the sampler / to resample permutations from.
            loss: default loss. None if not specified.
            sampler: default sampler. None if not specified.
            fsoi: Features of interest. All columns of X_train if not specified.
            encoder: specifies encoder to use for encoding categorical data
        """     
        assert isinstance(sampler, Sampler)
        self.model = predict
        if fsoi is None:
            self.fsoi = X_train.columns
        else:
            self.fsoi = fsoi
        self.X_train = X_train
        self.sampler = sampler
        self.loss = loss
        # TODO assert that encoder is of predefined type in line with our encoding strategy
        self.encoder = encoder
        # check whether feature set is valid
        self._valid_fset(self.fsoi)

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

    def _loss_specified(self):
        """Checks whether a loss was specified"""
        if self.loss is None:
            raise ValueError("Loss has not been specified.")
        else:
            return True
        
    def _check_sampler_trained(self, J, C, sampler, train_allowed):
        """Checks if sampler is trained on a given configuration and handles result"""
        # TODO check whether this function should be implemented by sampler?
        if not sampler.is_trained(J, C):  # sampler for foreground non-coalition
            # train if allowed, otherwise raise error
            if train_allowed:
                sampler.train(J, C)
                logger.info('Training sampler on {}|{}'.format(J, C))
            else:
                raise RuntimeError('Sampler is not trained on {}|{}'.format(J, C))
        else:
            txt = '\tCheck passed: Sampler is already trained on'
            txt = txt + '{}|{}'.format(J, C)
            logger.debug(txt)

    # Elementary Feature Importance Techniques
            
    def _surplus_simple(self, J, C, X_eval, y_eval, conditional, D=None, sampler=None, loss=None,
                        nr_runs=10, return_perturbed=False, train_allowed=True,
                        target='Y', marginalize=False, nr_resample_marginalize=5):
        """Computes AI via, meaning that we quantify the performance"""
        if target not in ['Y', 'Y_hat']:
            raise ValueError('Y and Y_hat are the only valid targets.')

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

        if not marginalize:
            nr_resample_marginalize = 1

        if not set(J).isdisjoint(set(C)):
            raise ValueError('J and C are not disjoint.')
        

        # sampler trained on all features except C given C?
        J, C = list(J), list(C)
        RuJ = list(set(D) - set(C))  # background non-coalition variables
        R = list(set(RuJ) - set(J)) # background non-coalition variables not in J
        JuC = list(set(J).union(set(C)))  # foreground variables

        if conditional:
            self._check_sampler_trained(R, JuC, sampler, train_allowed)
            self._check_sampler_trained(RuJ, C, sampler, train_allowed)

        letter = 'c' if conditional else 'm'
        desc = 'v^{}({}) - v^{}({})'.format(letter, JuC, letter, C)

        # initialize array for the perturbed samples
        nr_obs = X_eval.shape[0]
        index = utils.create_multiindex(['sample', 'i'],
                                        [np.arange(nr_runs),
                                         np.arange(nr_obs)])

        scores = pd.DataFrame([], index=index)

        for kk in np.arange(0, nr_runs, 1):
            # sample perturbed versions
            if conditional:
                X_RuJ_C = sampler.sample(X_eval, RuJ, C, num_samples=nr_resample_marginalize)
                X_R_JuC = sampler.sample(X_eval, R, JuC, num_samples=nr_resample_marginalize)
            else:
                X_RuJ_C = sampler.sample(X_eval, RuJ, [], num_samples=nr_resample_marginalize)
                X_R_JuC = sampler.sample(X_eval, R, [], num_samples=nr_resample_marginalize)

            # set unperturbed variabels to original variables
            X_JuC = pd.concat([X_eval[JuC]]*nr_resample_marginalize)
            X_C = pd.concat([X_eval[C]]*nr_resample_marginalize)
            X_R_JuC[JuC] = X_JuC.to_numpy()
            X_RuJ_C[C] = X_C.to_numpy()      

            X_RuJ_C = X_RuJ_C[D]
            # X_RuJ_C.columns = X_RuJ_C.columns.astype(str)
            X_RuJ_C = X_RuJ_C.rename(str, axis="columns")
            X_R_JuC = X_R_JuC[D]
            # X_R_JuC.columns = X_R_JuC.columns.astype(str)    
            X_R_JuC = X_R_JuC.rename(str, axis="columns")
            
            index = X_RuJ_C.index
            df_yh = pd.DataFrame(index=index,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])
            
            try:
                df_yh['y_hat_baseline'] = np.array(self.model(X_RuJ_C[D]))
                df_yh['y_hat_foreground'] = np.array(self.model(X_R_JuC[D]))
            except:
                print('shit.')

            # convert and aggregate predictions
            df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                  'y_hat_foreground': 'float'})
            df_yh = df_yh.groupby(level='i').mean()

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


    def di_from(self, K, B, J, X_eval, y_eval,
                D=None, loss=None, nr_runs=10,
                return_perturbed=False, train_allowed=True,
                target='Y', marginalize=False,
                nr_resample_marginalize=5,
                sampler=None,
                invert=False):
        """Computes the performance gain recieved from reconstructing features K
        from features J, given that features B are already fully reconstructed.
        So features K are sampled from P(X_K|X_J), for features B the observations
        are used, and for the remaining variables R:=D\K\B marginal sampling is used.

        Args:
            K: features of interest
            B: baseline features
            J: conditioning set when reconstructing features K
            X_eval: data to use for resampling and evaluation.
            y_eval: labels for evaluation.
            D: features (variables used by the predictive model)
            sampler: choice of sampler. Default None. Will throw an error
              when sampler is None and self.sampler is None as well.
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

        if D is None:
            D = X_eval.columns

        if len(D) == len(J):
            return self._surplus_simple(K, B, X_eval, y_eval, False, 
                                        D, sampler, loss, nr_runs, return_perturbed, train_allowed, target, marginalize, nr_resample_marginalize)

        if target not in ['Y', 'Y_hat']:
            raise ValueError('Y and Y_hat are the only valid targets.')

        if not marginalize:
            # if we take expecation over one sample that coincides with taking only one sample
            nr_resample_marginalize = 1

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logger.debug("Using class specified sampler.")


        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logger.debug("Using class specified loss.")

        if not set(K).isdisjoint(set(B)):
            raise ValueError('K and B are not disjoint.')
        
        # check if necessary samplers trained
        R = list(set(D) - set(B))
        RR = list(set(R) - set(K))

        self._check_sampler_trained(K, J, sampler, train_allowed)
        self._check_sampler_trained(R, [], sampler, train_allowed)

        # description of the computation
        desc = 'DI({} <- {} | {})'.format(K, J, B)

        # initialize array for the perturbed samples
        nr_obs = X_eval.shape[0]
        index = utils.create_multiindex(['sample', 'i'],
                                        [np.arange(nr_runs),
                                         np.arange(nr_obs)])

        # intitialize dataframe for the loss scores
        scores = pd.DataFrame([], index=index)

        # TODO adaptation:
        # - sample all features not in B or K from the marginal
        # - sample K from J
        # - use observations for B

        for kk in np.arange(0, nr_runs, 1):
            # sample features D \ B \ K from the marginal

            # sample features K given J
            X_K_J = sampler.sample(X_eval, K, J, num_samples=1)
            X_R = sampler.sample(X_eval, R, [], num_samples=nr_resample_marginalize)
            index = X_R.index

            # initialize array for predictions before and after perturbation
            df_yh = pd.DataFrame(index=index,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])

            # sample remaining features nr_resample_marginalize times
            # predict on the respective perturbation datasets
            for ll in np.arange(0, nr_resample_marginalize, 1):

                # create baseline dataframe
                X_tilde_baseline = X_eval.copy()
                X_tilde_baseline[R] = X_R.loc[(ll, slice(None)), R].to_numpy()

                # create foreground dataframe
                X_tilde_foreground = X_tilde_baseline.copy()
                X_tilde_foreground[K] = X_K_J.loc[(0, slice(None)), K].to_numpy()

                # make sure data is formatted as the model expects (selection and ordering)
                X_tilde_baseline = X_tilde_baseline[D]
                X_tilde_foreground = X_tilde_foreground[D]

                # encode data if necessary
                if self.encoder is not None:
                    X_tilde_baseline = self.encoder.transform(X_tilde_baseline)
                    X_tilde_foreground = self.encoder.transform(X_tilde_foreground)

                # create and store prediction
                y_hat_baseline = self.model(X_tilde_baseline)
                y_hat_foreground = self.model(X_tilde_foreground)

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = np.array(y_hat_baseline)
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = np.array(y_hat_foreground)

            # covert types of prediction dataframe
            df_yh = df_yh.astype({'y_hat_baseline': 'float64',
                                  'y_hat_foreground': 'float64'})
            # marginalize predictions
            df_yh = df_yh.groupby(level='i').mean()

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

    def ai_via(self, J, C, K, X_eval, y_eval, D=None, sampler=None, loss=None, 
               nr_runs=10, return_perturbed=False, train_allowed=True,
               target='Y', marginalize=False,
               nr_resample_marginalize=5):
        """Computes AI via, meaning that we quantify the performance
        gain when reconstructing features K from features J and C,
        given that in the baseline distribution all features are
        reconstructed with access to C.

        Args:
            J: variables of interest
            C: baseline variables/coalition variables
            K: "via" feature set
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
            target: whether loss shall be computed against Y or Y_hat
            marginalize: whether marginalization shall be performed
            nr_resample_marginalize: sample size for marginilization
                computation

        Returns:
            result: An explanation object with the RFI computation
            perturbed_foiss (optional): perturbed features of
                interest if return_perturbed
        """

        if D is None:
            D = X_eval.columns

        if len(D) == len(K):
            return self._surplus_simple(J, C, X_eval, y_eval, True, 
                                        D, sampler, loss, nr_runs, return_perturbed, train_allowed, target, marginalize, nr_resample_marginalize)

        if target not in ['Y', 'Y_hat']:
            raise ValueError('Y and Y_hat are the only valid targets.')

        if sampler is None:
            if self._sampler_specified():
                sampler = self.sampler
                logger.debug("Using class specified sampler.")

        if loss is None:
            if self._loss_specified():
                loss = self.loss
                logger.debug("Using class specified loss.")

        if not marginalize:
            nr_resample_marginalize = 1

        if not set(J).isdisjoint(set(C)):
            raise ValueError('J and C are not disjoint.')
        

        # sampler trained on all features except C given C?
        RuJ = list(set(D) - set(C))  # background non-coalition variables
        R = list(set(RuJ) - set(J))  # background non-coalition variables
        JuC = list(set(J).union(set(C)))  # foreground variables
        KnJ = list(set(K).intersection(set(J)))
        KnR = list(set(K).intersection(R))
        RuJoK = list(set(RuJ) - set(K))

        self._check_sampler_trained(RuJ, C, sampler, train_allowed)
        self._check_sampler_trained(KnR, JuC, sampler, train_allowed)

        # # sampler trained on subset of K not in JuC given JuC?
        # JuC = list(set(J).union(set(C)))
        # KnJuC = list(set(K).intersection(set(JuC)))
        # KR = list(set(K) - set(KnJuC))

        # self._check_sampler_trained(KR, JuC, sampler, train_allowed)

        desc = 'AR({} | {} -> {})'.format(J, C, K)

        # initialize array for the perturbed samples
        nr_obs = X_eval.shape[0]
        index = utils.create_multiindex(['sample', 'i'],
                                        [np.arange(nr_runs),
                                         np.arange(nr_obs)])

        scores = pd.DataFrame([], index=index)

        for kk in np.arange(0, nr_runs, 1):

            # sample perturbed versions
            X_RuJ_C = sampler.sample(X_eval, RuJ, C, num_samples=nr_resample_marginalize)
            X_KnR_JuC = sampler.sample(X_eval, KnR, JuC, num_samples=nr_resample_marginalize)
            index = X_RuJ_C.index

            df_yh = pd.DataFrame(index=index,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])

            # create foreground and background samples and make predictions
            for ll in np.arange(0, nr_resample_marginalize, 1):

                # create baseline dataframe
                X_tilde_baseline = X_eval.copy()
                X_tilde_baseline[RuJ] = X_RuJ_C.loc[(ll, slice(None)), RuJ].to_numpy()

                # create foreground dataframe by taking baseline (where C reconstructed)
                # and additionally adding info from J (and C) to K
                X_tilde_foreground_partial = X_tilde_baseline.copy()
                X_tilde_foreground_partial[KnJ] = X_eval.loc[:, KnJ].to_numpy() # C already reconstructed in the baseline
                X_tilde_foreground_partial[KnR] = X_KnR_JuC.loc[(ll, slice(None)), KnR].to_numpy()

                # make sure model can handle it (selection and ordering)
                X_tilde_baseline = X_tilde_baseline[D]
                X_tilde_foreground_partial = X_tilde_foreground_partial[D]

                # encode if necessary
                if self.encoder is not None:
                    X_tilde_baseline = self.encoder.transform(X_tilde_baseline)
                    X_tilde_foreground_partial = self.encoder.transform(X_tilde_foreground_partial)

                # create and store prediction
                y_hat_baseline = self.model(X_tilde_baseline)
                y_hat_foreground = self.model(X_tilde_foreground_partial)

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = np.array(y_hat_baseline)
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = np.array(y_hat_foreground)

            # convert and aggregate predictions
            df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                  'y_hat_foreground': 'float'})
            df_yh = df_yh.groupby(level='i').mean()

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

    # Elementary Feature Importance Techniques Applied To Multiple Features
    
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

    # PFI RFI CFI Wrapper Functions

    def rfi(self, G, X_eval, y_eval, fsoi=None, D=None, **kwargs):
        if D is None:
            D = self.X_train.columns
        # ex_full gives PFIs, i.e. for feature j: R(\tilde{X}_j^empty) - R(X)
        # ex_partly gives R(\tilde{X}_j^empty) - R(\tilde{X}_j^G)
        # combined we get R(\tilde{X}_j^G) - R(X) = RFI_j
        ex_full = self.dis_from_baselinefunc(D, X_eval, y_eval, fsoi=fsoi, D=D, baseline='remainder', **kwargs) # this returns the full model performance?
        ex_partly = self.dis_from_baselinefunc(G, X_eval, y_eval, fsoi=fsoi, D=D, baseline='remainder', **kwargs)
        rfi_scores = ex_full.scores - ex_partly.scores
        result = explanation.Explanation(self.fsoi, rfi_scores, ex_name=f'rfi_{G}')
        return result
    
    def pfi(self, X_eval, y_eval, fsoi=None, **kwargs):
        """Computes PFI on a given evaluation dataset.

        Args:
            X_eval: evaluation data
            y_eval: evaluation labels
            fsoi: features of interest, overrides self.fsoi if not None
        """
        ex = self.dis_from_baselinefunc(self.X_train.columns, X_eval, y_eval, fsoi=fsoi, baseline='remainder', **kwargs)
        ex.ex_name = 'pfi'
        return ex
            
    def cfi(self, X_eval, y_eval, fsoi=None, **kwargs):
        """Computes CFI on a given evaluation dataset.

        Args:
            X_eval: evaluation data
            y_eval: evaluation labels
            fsoi: features of interest, overrides self.fsoi if not None
        """
        ex = self.ais_via_contextfunc(self.X_train.columns, X_eval, y_eval, fsoi=fsoi, context='remainder', **kwargs)
        ex.ex_name = 'cfi'
        return ex
    
    # SAGE Value Function Wrappers
             
    def csagevf(self, S, X_eval, y_eval, C=[], **kwargs):
        """Computes the conditional SAGE value function for a given feature set S.

        Args:
            S: features of interest
            X_eval: test data
            y_eval: test labels
            **kwargs: keyword arguments that are passed to ai_via
        """
        ex = self.ai_via(S, C, self.X_train.columns, X_eval, y_eval, marginalize=True, **kwargs)
        ex.ex_name = 'csagevf S={} C={}'.format(S, C)
        return ex
    
    def msagevf(self, S, X_eval, y_eval, C=[], **kwargs):
        """Computes the marginal SAGE value function for a given feature set S.

        Args:
            S: features of interest
            X_eval: test data
            y_eval: test labels
            **kwargs: keyword arguments that are passed to di_from
        """
        ex = self.di_from(S, C, self.X_train.columns, X_eval, y_eval, marginalize=True, **kwargs)
        ex.ex_name = 'msagevf S={} C={}'.format(S, C)
        return ex

    def msagevfs(self, X_eval, y_eval, C='empty', **kwargs):
        """Computes the marginal SAGE value function for a given feature set S.

        Args:
        """
        if not isinstance(C, str) and C in ['empty', 'remainder']:
            raise NotImplementedError('Only empty and remainder are implemented for C.')
        ex = self.dis_from_baselinefunc(self.X_train.columns, X_eval, y_eval, baseline=C, marginalize=True, **kwargs)
        ex.ex_name = 'msagevfs C={}'.format(C)
        return ex

    def csagevfs(self, X_eval, y_eval, C='empty', **kwargs):
        """Computes the conditional SAGE value function for a given feature set S.

        Args:
        """
        if not isinstance(C, str) and C in ['empty', 'remainder']:
            raise NotImplementedError('Only empty and remainder are implemented for C.')
        ex = self.ais_via_contextfunc(self.X_train.columns, X_eval, y_eval, context=C, marginalize=True, **kwargs)
        ex.ex_name = 'csagevfs C={}'.format(C)
        return ex
    
    # Advanced Feature Importance

    def sage(self, X_eval, y_eval, partial_ordering,
             target='Y', method='associative', G=None, marginalize=True,
             nr_orderings=None, approx=math.sqrt, detect_convergence=False, thresh=0.01,
             extra_orderings=0, nr_runs=10, nr_resample_marginalize=10,
             sampler=None, loss=None, fsoi=None, orderings=None,
             save_orderings=True, save_each_obs=False, **kwargs):
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
            detect_convergence: bool, toggle convergence detection
            thresh: threshold for convergence detection
            extra_orderings: extra runs (i.t.o. orderings) after convergence has been detected
            nr_runs: how often each value function shall be computed
            nr_resample_marginalize: How many samples shall be used for the
                marginalization
            approx: if nr_orderings=None, approx determines the number of
                orderings w.r.t to the all possible orderings
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

        if detect_convergence:
            assert 0 < thresh < 1

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

        nr_datapoints_saved = 1
        if save_each_obs:
            nr_datapoints_saved = X_eval.shape[0]

        # create dataframe for computation results
        index = utils.create_multiindex(['ordering', 'sample', 'i'],
                                        [np.arange(nr_orderings_saved),
                                         np.arange(nr_runs),
                                         np.arange(nr_datapoints_saved)])
        scores = pd.DataFrame([], index=index)

        orderings_sampled = None
        if orderings is None:
            index_orderings = utils.create_multiindex(['ordering', 'sample'],
                                                      [np.arange(nr_orderings_saved),
                                                       np.arange(nr_runs)])
            orderings_sampled = pd.DataFrame(index=index_orderings,
                                             columns=['ordering'])
        # lss = np.zeros(
        #     (len(self.fsoi), nr_runs, X_eval.shape[0], nr_orderings))
        # ord hist helps to avoid duplicate histories
        ord_hist = None

        max_orderings = np.full(nr_runs, nr_orderings)

        for jj in range(nr_runs):

            if detect_convergence:
                # inputs to detect_conv, reset for every run
                convergence_var = False
                extra_orderings_r = extra_orderings

            for ii in range(nr_orderings):
                ordering = None
                if orderings is None:
                    ordering, ord_hist = utils.sample_partial(partial_ordering,
                                                              ord_hist)
                    orderings_sampled.loc[(ii, jj), 'ordering'] = ordering
                else:
                    ordering = orderings.loc[(ii, jj), 'ordering']

                logging.info('Ordering : {}'.format(ordering))

                # compute scores for the ordering
                ex = None
                if method == 'associative':
                    ex = self.ais_via_ordering(ordering, G, X_eval, y_eval,
                                               target=target, marginalize=marginalize,
                                               nr_runs=1, nr_resample_marginalize=nr_resample_marginalize,
                                               **kwargs)
                elif method == 'direct':
                    # TODO check whether this should be dis_from_baselinefunc
                    ex = self.dis_from_ordering(ordering, G, X_eval, y_eval,
                                                target=target, marginalize=marginalize,
                                                nr_runs=1, nr_resample_marginalize=nr_resample_marginalize,
                                                **kwargs)

                if save_each_obs:
                    scores_arr = ex.scores[fsoi].to_numpy()
                else:
                    scores_arr = ex.fi_vals()[fsoi].to_numpy()
                scores.loc[(ii, jj, slice(None)), fsoi] = scores_arr

                if detect_convergence:
                    # detect_conv returns whether conv detected (bool) and the number of extra orderings left
                    scores_run = scores.loc[(slice(None), jj, slice(None)), fsoi]
                    convergence_var, extra_orderings_r = detect_conv(scores_run, ii, thresh, extra_orderings=extra_orderings_r,
                                                                     conv_detected=convergence_var)
                    # if convergence has been detected and no extra orderings left break out of loop
                    if convergence_var and extra_orderings_r == 0:
                        max_orderings[jj] = ii    # to determine max. orderings to trim dataset
                        print('Detected convergence after ordering no.', ii+1)
                        break

        if orderings is None:
            orderings = orderings_sampled

        if detect_convergence:
            # trim scores to dim of actual number of orderings ii after convergence (potentially < nr_orderings)
            scores = scores.loc[(slice(0, max_orderings.max()), slice(None), slice(None))]
            orderings = orderings[0:nr_runs*(max_orderings.max()+1)]

        result = explanation.Explanation(fsoi, scores, ex_name='SAGE')

        if save_orderings:
            return result, orderings
        else:
            return result

    def csage(self, X_eval, y_eval, partial_ordering=None, **kwargs):
        """Compute conditional SAGE values for a given partial ordering.

        Args:
            X_eval: evaluation data
                The evaluation data used to compute the conditional SAGE values.
            y_eval: evaluation labels
                The evaluation labels corresponding to the evaluation data.
            partial_ordering: partial ordering for the computation
                The partial ordering used for the computation of conditional SAGE values.
                Provided as a list of tuples and lists, where elements within the tuples can be permuted.
                [(X_eval.colums)] implies no ordering (the default)
            **kwargs: keyword arguments that are passed to sage
                Additional keyword arguments that can be passed to the `sage` method.

        Returns:
            res: computed conditional SAGE values
                The computed conditional SAGE values based on the provided evaluation data,
                evaluation labels, and partial ordering.
        """
        if partial_ordering is None:
            partial_ordering = [tuple(X_eval.columns)]
        res = self.sage(X_eval, y_eval, partial_ordering, method='associative', **kwargs)
        if isinstance(res, tuple): # if the orderings are passed as well
            res[0].ex_name = 'csage '
        else:
            res.ex_name = 'csage'
        return res
    
    def msage(self, X_eval, y_eval, partial_ordering=None, **kwargs):
        """
        Compute marginal SAGE values for a given partial ordering.

        Args:
            X_eval: evaluation data
            y_eval: evaluation labels
            partial_ordering: partial ordering for the computation
            **kwargs: keyword arguments that are passed to sage
        """
        if partial_ordering is None:
            partial_ordering = [tuple(X_eval.columns)]
        res = self.sage(X_eval, y_eval, partial_ordering, method='direct', **kwargs)
        if isinstance(res, tuple): # in case orderings passed as well
            res[0].ex_name = 'msage '
        else:
            res.ex_name = 'msage'
        return res

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
        ex = explanation.DecompositionExplanation(self.fsoi, decomposition, ex_name=None)
        return ex, orderings
