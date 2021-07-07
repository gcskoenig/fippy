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
                 loss=None):
        """Inits Explainer with sem, mask and potentially sampler and loss"""
        self.model = model
        self.fsoi = fsoi  # now column names, not indexes
        self.X_train = X_train
        self.sampler = sampler
        self.decorrelator = decorrelator
        self.loss = loss
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

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = y_hat_baseline
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = y_hat_foreground

            # covert types of prediction dataframe
            df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                  'y_hat_foreground': 'float'})
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

            df_yh = pd.DataFrame(index=index,
                                 columns=['y_hat_baseline',
                                          'y_hat_foreground'])

            # create foreground and background samples and make predictions
            for ll in np.arange(0, nr_resample_marginalize, 1):

                X_tilde_baseline = X_eval.copy()
                X_tilde_foreground = X_eval.copy()

                # copy ll-th sample for R_ variables
                arr_reconstr = X_R_CuJ.loc[(ll, slice(None)), R_].to_numpy()
                X_tilde_foreground[R_] = arr_reconstr

                # decorellate X_R and copy
                X_R_decorr = decorrelator.decorrelate(X_tilde_foreground, R, J, C)
                arr_decorr = X_R_decorr[R].to_numpy()

                # TODO make use of features K to selectively update
                #  background to foreground (only features K ar updated)
                X_tilde_baseline[R] = arr_decorr

                # make sure model can handle it (selection and ordering)
                X_tilde_baseline = X_tilde_baseline[D]
                X_tilde_foreground = X_tilde_foreground[D]

                # create and store prediction
                y_hat_baseline = self.model(X_tilde_baseline)
                y_hat_foreground = self.model(X_tilde_foreground)

                df_yh.loc[(ll, slice(None)), 'y_hat_baseline'] = y_hat_baseline
                df_yh.loc[(ll, slice(None)), 'y_hat_foreground'] = y_hat_foreground

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

    def sage(self, X_test, y_test, partial_ordering,
             target='Y', method='associative', G=None, marginalize=True,
             nr_orderings=None, approx=math.sqrt, convergence=False,
             nr_runs=10, nr_resample_marginalize=10,
             sampler=None, loss=None, D=None, orderings=None,
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
            D: specifies features of interest (for which SAGE values are computed).
                By default set to X.columns (if D=None).
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
            G = X_test.columns

        if X_test.shape[1] != len(self.fsoi):
            logger.debug('self.fsoi: {}'.format(self.fsoi))
            logger.debug('#features in model: {}'.format(X_test.shape[1]))
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
                # TODO(gcsk,cph): convergence detection like in SAGE paper
                #  see https://github.com/iancovert/sage/blob/master/sage/permutation_estimator.py

                # TODO: check if jj in features for which the score shall
                # TODO: be computed
                # compute change in performance
                # by entering the respective feature
                # store the result in the right place
                # validate training of sampler
                J, C = [ordering[jj - 1]], ordering[:jj - 1]
                ex = None

                # we have already asserted that method is either associative or direct
                if method == 'associative':
                    ex = self.ai_via(J, C, G, X_test, y_test,
                                     target=target, marginalize=marginalize,
                                     nr_runs=nr_runs,
                                     nr_resample_marginalize=nr_resample_marginalize,
                                     **kwargs)
                elif method == 'direct':
                    ex = self.di_from(J, C, G, X_test, y_test, nr_runs=nr_runs, target=target, marginalize=marginalize,
                                      nr_resample_marginalize=nr_resample_marginalize, **kwargs)

                scores_arr = ex.scores.to_numpy()
                scores.loc[(ii, slice(None), slice(None)), ordering[jj - 1]] = scores_arr

        result = explanation.Explanation(self.fsoi, scores, ex_name='SAGE')

        if orderings is None:
            orderings = orderings_sampled

        if save_orderings:
            return result, orderings
        else:
            return result

    def viafrom(self, imp_type, fsoi, X_eval, y_eval, target='Y', nr_runs=10,
                show_pbar=True, components=None, **kwargs):
        """
        Computes a simple fast decomposition of DI(X_j|X_{-j}), AI(X_j|X_\emptyset) or (conditional) SAGE.

        For DI(X_j|X_{-j}), each DI_from(X_j|X_{-j} <- X_k) is computed (for k \in components)
            and compared to empty "from set"
        For AI(X_j|X_\emptyset) each AI_via(X_j|X_\emptyset -> X_k|X_{-k}) is computed and compared with
            full "via" set
        For conditional SAGE, SAGE is computed where every AI_via statement is replaced as described above

        Arguments:
            imp_type: either 'ai_via', 'di_from' or 'sage'. for sage the default method is used,
                but can be passed via the respective keyword arguments
            fsoi: features of interest
            X_eval: X_eval dataset
            y_eval: y_eval data
            target: whether one wants to evaluate against 'Y' or 'Y_hat'
            nr_runs: over how many runs the result shall be averaged
            show_pbar: whether progress bar shall be presented
            components: for which components the result shall be computed.
                if components==None, then X_eval.columns is assigned
            **kwargs: keyword arguments are passed to ai_via, di_from or sage
        """
        if imp_type not in ['ai_via', 'di_from', 'sage']:
            raise ValueError('Only ai_via, sage and di_from'
                             'implemented for imp_type.')

        if target not in ['Y', 'Y_hat']:
            raise ValueError('Only Y and Y_hat implemented as target.')

        # values (nr_perm, nr_runs, nr_components, nr_fsoi)
        # components are: (elements of ordering,..., remainder)
        # elements of ordering are sorted in increasing order

        if components is None:
            components = X_eval.columns
        components = list(components)
        components.append('total')
        nr_components = len(components)

        # create dataframe for computation results
        # orderings is just for compatibility with other decompositions
        index = utils.create_multiindex(['component', 'ordering', 'sample'],
                                        [components,
                                         np.arange(1),
                                         np.arange(nr_runs)])
        arr = np.zeros((nr_components * nr_runs * 1, len(fsoi)))
        decomposition = pd.DataFrame(arr, index=index, columns=fsoi)

        if show_pbar:
            mgr = enlighten.get_manager()
            pbar = mgr.counter(total=nr_components * len(fsoi),
                               desc='naive_decomposition',
                               unit='{} runs'.format(imp_type))

        # helper funciton to compute the remainder for a specific feature
        def get_rmd(fs, f):
            """fs: all features, f: feature"""
            rmd = list(set(fs).difference([f]))
            return rmd

        if imp_type == 'sage':
            # TODO implement direct version as well
            #  then (G is not remainder but only one item, baseline is empty G)

            # make sure that associative method is used
            if 'method' in kwargs:
                if kwargs['method'] != 'associative':
                    raise NotImplementedError('for sage only the method associative is implemented for viafrom')
            else:
                kwargs['method'] = 'associative'

            # compute SAGE with G=full feature set
            expl, ordering = self.sage(X_eval, y_eval, [tuple(fsoi)],
                                       **kwargs)
            fi_vals_total = expl.fi_vals()[fsoi].to_numpy()
            decomposition.loc[idx['total', 0, :], fsoi] = fi_vals_total

            # compute SAGE without each of the components and compute difference
            for component in get_rmd(components, 'total'):
                rmd = get_rmd(X_eval.columns, component)
                expl, ordering = self.sage(X_eval, y_eval, [tuple(fsoi)],
                                           G=rmd, **kwargs)
                fi_vals = expl.fi_vals()[fsoi].to_numpy()
                diff = fi_vals_total - fi_vals
                decomposition.loc[idx[component, 0, :], fsoi] = diff

            # store decomposition result
            ex = decomposition_ex.DecompositionExplanation(self.fsoi,
                                                           decomposition,
                                                           ex_name=None)
            return ex

        # iterate over features
        for foi in fsoi:

            # compute baseline
            fi_vals_total = None
            if imp_type == 'ar_via':  # compute ar of foi over emptyset
                expl = self.ai_via([foi], [], X_eval.columns,
                                   X_eval, y_eval, nr_runs=nr_runs,
                                   target=target, **kwargs)
                fi_vals_total = expl.fi_vals().to_numpy()
            elif imp_type == 'dr_from':  # compute total PFI (over rmd)
                rmd = get_rmd(X_eval.columns, foi)
                expl = self.di_from([foi], rmd, X_eval.columns,
                                    X_eval, y_eval, nr_runs=nr_runs,
                                    target=target, **kwargs)
                fi_vals_total = expl.fi_vals().to_numpy()
            decomposition.loc[idx['total', 0, :], foi] = fi_vals_total

            # iterate over components
            for component in get_rmd(components, 'total'):
                if show_pbar:
                    pbar.update()

                fi_vals = None
                if imp_type == 'ar_via':
                    rmd = get_rmd(X_eval.columns, component)
                    expl = self.ai_via([foi], [], rmd,
                                       X_eval, y_eval, nr_runs=nr_runs,
                                       target=target, **kwargs)
                    fi_vals = expl.fi_vals().to_numpy()
                    diff = fi_vals_total - fi_vals
                    decomposition.loc[idx[component, 0, :], foi] = diff
                elif imp_type == 'dr_from':
                    rmd = get_rmd(X_eval.columns, foi)
                    expl = self.di_from([foi], rmd, [component],
                                        X_eval, y_eval, nr_runs=nr_runs,
                                        target=target, **kwargs)
                    fi_vals = expl.fi_vals().to_numpy()
                    decomposition.loc[idx[component, 0, :], foi] = fi_vals

        ex = decomposition_ex.DecompositionExplanation(self.fsoi,
                                                       decomposition,
                                                       ex_name=None)
        return ex

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
            imp_type: Either 'associative', 'direct' or 'sage'
            fsoi: features, for which the importance scores (of type imp_type)
                are to be decomposed.
            partial_ordering: partial ordering for the decomposition
                of the form (1, 2, [3, 4, 5], 6) where the ordering
                within a tuple is fixed and within a list may be
                permuted.
            X_test: test data
            y_test: test labels
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

    # def tdi(self, X_eval, y_eval, G, D=None, sampler=None, loss=None,
    #         nr_runs=10, return_perturbed=False, train_allowed=True):
    #     """Computes Relative Feature importance for every feature
    #          meaning we get an explanation where for every feature we get
    #          DI(X_j|X_-j <- X_G)
    #
    #     Args:
    #         X_eval: data to use for resampling and evaluation.
    #         y_eval: labels for evaluation.
    #         G: relative feature set
    #         D: features, used by the predictive model
    #         sampler: choice of sampler. Default None. Will throw an error
    #           when sampler is None and self.sampler is None as well.
    #         loss: choice of loss. Default None. Will throw an Error when
    #           both loss and self.loss are None.
    #         nr_runs: how often the experiment shall be run
    #         return_perturbed: whether the sampled perturbed versions
    #             shall be returned
    #         train_allowed: whether the explainer is allowed to train
    #             the sampler
    #
    #     Returns:
    #         result: An explanation object with the RFI computation
    #         perturbed_foiss (optional): perturbed features of
    #             interest if return_perturbed
    #     """
    #
    #     if sampler is None:
    #         if self._sampler_specified():
    #             sampler = self.sampler
    #             logger.debug("Using class specified sampler.")
    #
    #     if loss is None:
    #         if self._loss_specified():
    #             loss = self.loss
    #             logger.debug("Using class specified loss.")
    #
    #     if D is None:
    #         D = X_eval.columns
    #
    #     # check whether the sampler is trained for each fsoi conditional on G
    #     for f in self.fsoi:
    #         if not sampler.is_trained([f], G):
    #             # train if allowed, otherwise raise error
    #             if train_allowed:
    #                 sampler.train([f], G)
    #                 logger.info('Training sampler on {}|{}'.format([f], G))
    #             else:
    #                 raise RuntimeError(
    #                     'Sampler is not trained on {}|{}'.format([f], G))
    #         else:
    #             txt = '\tCheck passed: Sampler is already trained on'
    #             txt = txt + '{}|{}'.format([f], G)
    #             logger.debug(txt)
    #
    #     # initialize array for the perturbed samples
    #     nr_obs = X_eval.shape[0]
    #     index = utils.create_multiindex(['sample', 'i'],
    #                                     [np.arange(nr_runs),
    #                                      np.arange(nr_obs)])
    #     X_fsoi_pert = pd.DataFrame([], index=index)
    #
    #     # sample perturbed versions
    #     for foi in self.fsoi:
    #         x_foi_pert = sampler.sample(
    #             X_eval, [foi], G, num_samples=nr_runs)
    #         X_fsoi_pert[foi] = x_foi_pert
    #
    #     scores = pd.DataFrame([], index=index)
    #
    #     # compute observasitonwise loss differences for all runs and fois
    #     for foi in self.fsoi:
    #         # copy of the data where perturbed variables are copied into
    #         for kk in np.arange(0, nr_runs, 1):
    #             # replaced with perturbed
    #             X_eval_tilde = X_eval.copy()
    #             arr = X_fsoi_pert.loc[(kk, slice(None)), foi].to_numpy()
    #             X_eval_tilde[foi] = arr
    #             # X_eval_one_perturbed[:, self.fsoi[jj]]
    #             # = perturbed_foiss[jj, kk, :]
    #             # using only seen while training features
    #
    #             # make sure model can handle it (selection and ordering)
    #             X_eval_tilde_model = X_eval_tilde[D]
    #             # X_eval_one_perturbed_model = X_eval_one_perturbed[:, D]
    #             X_eval_model = X_eval[D]
    #
    #             # compute difference in observationwise loss
    #             loss_pert = loss(y_eval, self.model(X_eval_tilde_model))
    #             loss_orig = loss(y_eval, self.model(X_eval_model))
    #             diffs = (loss_pert - loss_orig)
    #             scores.loc[(kk, slice(None)), foi] = diffs
    #             # lss[jj, kk, :] = loss_pert - loss_orig
    #
    #     # return explanation object
    #     ex_name = 'RFI^{}'.format(G)
    #     result = explanation.Explanation(
    #         self.fsoi, scores,
    #         ex_name=ex_name)
    #
    #     if return_perturbed:
    #         logger.debug('Return both explanation and perturbed.')
    #         return result, X_fsoi_pert
    #     else:
    #         logger.debug('Return explanation object only')
    #         return result

    # def tai(self, X_eval, y_eval, K, D=None, sampler=None, decorrelator=None,
    #         loss=None, nr_runs=10, return_perturbed=False, train_allowed=True,
    #         ex_name=None):
    #     """Computes total feature association
    #
    #
    #     Args:
    #         X_eval: data to use for resampling and evaluation.
    #         y_eval: labels for evaluation.
    #         K: features not to be reconstructed?
    #         D: model features (including their required ordering)
    #         sampler: choice of sampler. Default None. Will throw an error
    #           when sampler is None and self.sampler is None as well.
    #         decorrelator: choice of decorrelator. Same behaviour for None
    #           as for sampler
    #         loss: choice of loss. Default None. Will throw an Error when
    #           both loss and self.loss are None.
    #         nr_runs: how often the experiment shall be run
    #         return_perturbed: whether the sampled perturbed
    #             versions shall be returned
    #         train_allowed: whether the explainer is allowed
    #             to train the sampler
    #
    #     Returns:
    #         result: An explanation object with the RFI computation
    #         perturbed_foiss (optional): perturbed features of
    #             interest if return_perturbed
    #     """
    #
    #     if sampler is None:
    #         if self._sampler_specified():  # may throw an error
    #             sampler = self.sampler
    #             logger.debug("Using class specified sampler.")
    #
    #     if decorrelator is None:
    #         if self._decorrelator_specified():  # may throw error
    #             decorrelator = self.decorrelator
    #             logger.debug("Using class specified decorrelator")
    #
    #     if loss is None:
    #         if self._loss_specified():  # may throw an error
    #             loss = self.loss
    #             logger.debug("Using class specified loss.")
    #
    #     if D is None:
    #         D = X_eval.columns
    #     # all_fs = np.arange(X_test.shape[1])
    #
    #     # TODO: this sampler is not necessary anymore?
    #     # check whether the sampler is trained for the baseline perturbation
    #     if not sampler.is_trained(D, []):
    #         # train if allowed, otherwise raise error
    #         if train_allowed:
    #             sampler.train(D, [])
    #             logger.info('Training sampler on {}|{}'.format(D, []))
    #         else:
    #             raise RuntimeError(
    #                 'Sampler is not trained on {}|{}'.format(D, []))
    #     else:
    #         txt = '\tCheck passed: Sampler is already trained on '
    #         txt = txt + '{}|{}'.format(D, [])
    #         logger.debug(txt)
    #
    #     # check for each of the features of interest
    #     for foi in self.fsoi:
    #         if not sampler.is_trained(D, [foi]):
    #             # train if allowed, otherwise raise error
    #             if train_allowed:
    #                 sampler.train(D, [foi])
    #                 logger.info(
    #                     'Training sampler on {}|{}'.format(D, [foi]))
    #             else:
    #                 raise RuntimeError(
    #                     'Sampler is not trained on {}|{}'.format(D, [foi]))
    #         else:
    #             txt = '\tCheck passed: Sampler is already trained on '
    #             txt = txt + '{}|{}'.format(D, [foi])
    #             logger.debug(txt)
    #
    #     # check whether decorrelators have been trained
    #     for foi in self.fsoi:
    #         if not decorrelator.is_trained(K, [foi], []):
    #             if train_allowed:
    #                 decorrelator.train(K, [foi], [])
    #                 txt = 'Training decorrelator on '
    #                 txt = txt + '{} idp {} | {}'.format(K, [foi], [])
    #                 logger.info(txt)
    #             else:
    #                 txt = 'Decorrelator is not trained on '
    #                 txt = txt + '{} {} | {}'.format(K, [foi], [])
    #                 raise RuntimeError(txt)
    #         else:
    #             logger.debug('\tCheck passed: '
    #                          'Decorrelator is already trained on '
    #                          '{} {} | {}'.format(K, [foi], []))
    #
    #     # initialize array for the perturbed samples
    #     nr_obs = X_eval.shape[0]
    #
    #     # initialize pandas dataframes for X_eval_tilde baseline
    #     # and X_eval_tilde reconstrcted
    #     # TODO evaluate whether those two lines are still necessary?
    #     index_bsln = utils.create_multiindex(['sample', 'i'],
    #                                          [np.arange(nr_runs),
    #                                           np.arange(nr_obs)])
    #     X_eval_tilde_bsln = pd.DataFrame([], index=index_bsln, columns=D)
    #
    #     # index_rcnstr = utils.create_multiindex(['foi', 'sample', 'i'],
    #     #                                        [self.fsoi,
    #     #                                         np.arange(nr_runs),
    #     #                                         np.arange(nr_obs)])
    #     # X_eval_tilde_rcnstr = pd.DataFrame([], index=index_rcnstr, columns=D)
    #
    #     # create empty scores data frame
    #     index_scores = utils.create_multiindex(['sample', 'i'],
    #                                            [np.arange(nr_runs),
    #                                             np.arange(nr_obs)])
    #     scores = pd.DataFrame([], index=index_scores)
    #
    #     # sample baseline X^\emptyset
    #     # TODO evaluate whether this here is unneccesarily complex
    #     # TODO could we directly evaluate the difference with the formula for AI_via from the paper?
    #     X_eval_tilde_bsln = sampler.sample(X_eval, D, [], num_samples=nr_runs)
    #     lss_baseline = []
    #     for kk in np.arange(nr_runs):
    #         X_bl = X_eval_tilde_bsln.loc[(kk, slice(None)), D]
    #         l_pb = loss(y_eval, self.model(X_bl))
    #         lss_baseline.append(l_pb)
    #
    #     # sample perturbed versions
    #     for foi in self.fsoi:
    #         # X^foi
    #         sample = sampler.sample(X_eval, D, [foi], num_samples=nr_runs)
    #         for kk in np.arange(nr_runs):
    #             # X^\emptyset,linked
    #             sample_decorr = decorrelator.decorrelate(sample.loc[kk, :],
    #                                                      K, [foi], [])
    #             sd_np = sample_decorr[D].to_numpy()
    #             # X_eval_tilde_rcnstr.loc[(foi, kk, slice(None)), D] = sd_np
    #             # sd = X_eval_tilde_rcnstr.loc[(foi, kk, slice(None)), D]
    #             l_rc = loss(y_eval, self.model(sd_np))
    #             scores.loc[(kk, slice(None)), foi] = lss_baseline[kk] - l_rc
    #
    #     if ex_name is None:
    #         ex_name = 'Unknown tai'
    #
    #     # return explanation object
    #     result = explanation.Explanation(self.fsoi,
    #                                      scores,
    #                                      ex_name=ex_name)
    #     if return_perturbed:
    #         raise NotImplementedError(
    #             'Returning perturbed not implemented yet.')
    #     else:
    #         logger.debug('Return explanation object only')
    #         return result

