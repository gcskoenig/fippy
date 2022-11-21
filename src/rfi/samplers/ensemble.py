"""Categorical Sampler
Random Forest based
only works for univariate targets.
"""
import numpy as np
import pandas as pd
from rfi.samplers.sampler import Sampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import torch
import category_encoders as ce

class UnivRFSampler(Sampler):
    """
    Simple sampling from observed data
    Attributes:
        see rfi.samplers.Sampler
    """

    def __init__(self, X_train, **kwargs):
        """Initialize Sampler with X_train and mask."""
        super().__init__(X_train, **kwargs)

    def train(self, J, G, verbose=True):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            J: features of interest
            G: arbitrary set of variables
            verbose: printing
        """
        assert len(J) == 1
        J = Sampler._to_array(list(J))
        G = Sampler._to_array(list(G))
        super().train(J, G, verbose=verbose)

        JuG = list(set(J).union(G))

        if not self._train_J_degenerate(J, G, verbose=verbose):
            # TODO assert that variables are categorical
            # TODO raise error if that is not the case
            # adjust max_features in param_grid to actual number of features
            # if len(G) < 4:
            #     max_features = list(range(1, len(G)+1))
            # elif len(G) == 4:
            #     max_features = [2, 3, 4]
            # else:
            #     max_features = [2, 3, 5]

            # param_grid = {
            #     'bootstrap': [True],
            #     'criterion': ['entropy'],
            #     'max_depth': [80, 90, 100, 110],
            #     'max_features': max_features,
            #     'min_samples_leaf': [5, 10, 50, 100],
            #     'min_samples_split': [5, 10, 50, 100],
            #     'n_estimators': [100, 200, 300, 1000]
            # }

            rf = RandomForestClassifier()  # Instantiate the grid search model
            # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
            #                                n_iter=100, verbose=0,
            #                                n_jobs=-1, scoring='neg_log_loss')  # Fit the random search model
            X_train_G, X_train_J = self.X_train[Sampler._order_fset(G)], self.X_train[J[0]]
            enc_G = ce.OneHotEncoder()
            enc_G.fit(X_train_G)
            X_train_G_enc = enc_G.transform(X_train_G)
            rf.fit(X_train_G_enc, X_train_J)
            model = rf  # _random.best_estimator_

            def samplefunc(eval_context, num_samples=1, **kwargs):
                arrs = []
                for snr in range(num_samples):
                    X_eval = pd.DataFrame(data=eval_context, columns=Sampler._order_fset(G))
                    X_eval_enc = enc_G.transform(X_eval)

                    # TODO: use the model to sample the target variable
                    pred_proba = torch.tensor(model.predict_proba(X_eval_enc))
                    sample = torch.multinomial(pred_proba, 1).numpy().flatten()
                    sample = model.classes_[sample]
                    arrs.append(sample.reshape(1, -1, len(J)))
                res = np.concatenate(arrs, axis=0)
                res = np.swapaxes(res, 0, 1)
                return res

            self._store_samplefunc(J, G, samplefunc, verbose=verbose)