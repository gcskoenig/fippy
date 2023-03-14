"""Categorical Sampler
Random Forest based
only works for univariate targets.
"""
import logging

import numpy as np
import pandas as pd
from rfi.samplers.sampler import Sampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss, mean_squared_error, make_scorer
import torch
import category_encoders as ce

def get_param_grid(G):
    if len(G) < 4:
        max_features = list(range(1, len(G) + 1))
    elif len(G) == 4:
        max_features = [2, 3, 4]
    else:
        max_features = [2, 3, 5]

    param_grid = {
        'bootstrap': [True],
        # 'criterion': ['entropy'],
        'max_depth': [80, 90, 100, 110],
        'max_features': max_features,
        'min_samples_leaf': [5, 10, 50, 100],
        'min_samples_split': [5, 10, 50, 100],
        'n_estimators': [100, 200, 300, 1000]
    }

    return param_grid


class UnivRFSampler(Sampler):
    """
    Simple sampling from observed data
    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, **kwargs):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train, **kwargs)

    def train(self, J, G, verbose=True, score=None, tuning=False):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            J: features of interest
            G: arbitrary set of variables
            verbose: printing
        """
        assert len(J) == 1
        assert type(self.X_train.dtypes[J[0]] == 'category')
        # if score is None:
        #     score = log_loss

        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        log_loss_scorer = make_scorer(log_loss, greater_is_better=False)

        JuG = list(set(J).union(G))

        if not self._train_J_degenerate(J, G) and not self._train_G_degenerate(J, G):
            # adjust max_features in param_grid to actual number of features

            rf = RandomForestClassifier()  # Instantiate the grid search model

            if tuning:
                param_grid = get_param_grid(G)
                rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                               n_iter=50, verbose=0,
                                               n_jobs=-1, scoring=log_loss_scorer)  # Fit the random search model

            X_train_G, X_train_J = self.X_train[Sampler._order_fset(G)], self.X_train[J[0]]

            if len(set(self.cat_inputs).intersection(G)) > 0:
                enc_G = ce.OneHotEncoder()
                enc_G.fit(X_train_G)
                X_train_G_enc = enc_G.transform(X_train_G)
            else:
                X_train_G_enc = X_train_G

            if tuning:
                rf_random.fit(X_train_G_enc, X_train_J)
                model = rf_random.best_estimator_
            else:
                rf.fit(X_train_G_enc, X_train_J)
                model = rf

            def samplefunc(eval_context, num_samples=1, **kwargs):
                arrs = []
                for snr in range(num_samples):
                    X_eval = pd.DataFrame(data=eval_context, columns=Sampler._order_fset(X_train_G.columns))
                    if len(set(self.cat_inputs).intersection(G)):
                        X_eval_enc = enc_G.transform(X_eval)
                    else:
                        X_eval_enc = X_eval

                    pred_proba = torch.tensor(model.predict_proba(X_eval_enc))
                    sample = torch.multinomial(pred_proba, 1).numpy().flatten()
                    sample = model.classes_[sample]
                    arrs.append(sample.reshape(1, -1, len(J)))
                res = np.concatenate(arrs, axis=0)
                res = np.swapaxes(res, 0, 1)
                return res

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)


class ContUnivRFSampler(Sampler):
    """
    Simple sampling from observed data
    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, **kwargs):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train, **kwargs)

    def train(self, J, G, verbose=True, score=None, tuning=False):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            J: features of interest
            G: arbitrary set of variables
            verbose: printing
        """
        assert len(J) == 1
        assert len(set(self.cat_inputs).intersection(J)) == 0
        assert score is None
        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        # if score is None:
        #     score = mean_squared_error

        JuG = list(set(J).union(G))

        if not self._train_J_degenerate(J, G):
            # TODO assert that target variable is continuous
            rf = RandomForestRegressor()  # Instantiate the grid search model
            if tuning:
                param_grid = get_param_grid(G)

                rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                               n_iter=50, verbose=0,
                                               n_jobs=-1, scoring=mse_scorer)  # Fit the random search model
            X_train_G, X_train_J = self.X_train[Sampler._order_fset(G)], self.X_train[J[0]]
            if len(set(self.cat_inputs).intersection(G)) > 0:
                enc_G = ce.OneHotEncoder()
                enc_G.fit(X_train_G)
                X_train_G_enc = enc_G.transform(X_train_G)
            else:
                X_train_G_enc = X_train_G
            if tuning:
                rf_random.fit(X_train_G_enc, X_train_J)
                model = rf_random.best_estimator_
            else:
                rf.fit(X_train_G_enc, X_train_J)
                model = rf
            resids = X_train_J - model.predict(X_train_G_enc)

            def samplefunc(eval_context, num_samples=1, **kwargs):
                arrs = []
                for snr in range(num_samples):
                    X_eval = pd.DataFrame(data=eval_context, columns=Sampler._order_fset(X_train_G.columns))
                    if len(set(self.cat_inputs).intersection(G)):
                        X_eval_enc = enc_G.transform(X_eval)
                    else:
                        X_eval_enc = X_eval

                    pred = torch.tensor(model.predict(X_eval_enc))
                    sample = pred + np.random.choice(resids, pred.shape[0])
                    arrs.append(sample.reshape(1, -1, len(J)))
                res = np.concatenate(arrs, axis=0)
                res = np.swapaxes(res, 0, 1)
                return res

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)
