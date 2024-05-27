from fippy.samplers import Sampler
from fippy.explainers import Explainer
from fippy.explanation import LearnerExplanation
from sklearn.base import clone, is_classifier
from sklearn.model_selection import train_test_split
from fippy.utils import create_multiindex
import numpy as np
import pandas as pd
import tqdm

class LearnerExplainer:
    """Implements learner versions of various feature importance algorithms.
    Instead of only computing the method for a specific model, the learner version
    computes the feature importance over various refits.

    Attributes:
        learner: Sklearn learner (with fit and predict methods that can be cloned).
        X: Features
        y: target
        loss: default loss. None if not specified.
        sampler: default sampler. None if not specified.
        fsoi: Features of interest. All columns of X_train if not specified.
        encoder: specifies encoder to use for encoding categorical data
    """
    def __init__(self, learner, X, y,
                 loss=None,
                 sampler=None, fsoi=None, encoder=None):
        """Inits Explainer with learner, features, target, and optionally
        with loss, encoder, and features of interest.
        
        Args:
            learner: Object with fit and predict methods.
            X: Features
            y: target
            loss: default loss. None if not specified.
            sampler: default sampler. None if not specified.
            fsoi: Features of interest. All columns of X if not specified.
            encoder: specifies encoder to use for encoding categorical data
        """     
        assert isinstance(sampler, Sampler)
        self.learner = learner
        try:
            self._isclassifier = is_classifier(learner)
        except Exception as exp:
            raise ValueError('Learner must work with sklearn.base.is_classifier function.')
        if fsoi is None:
            self.fsoi = X.columns
        else:
            self.fsoi = fsoi
        self.X = X
        self.y = y
        self.sampler = sampler
        self.loss = loss
        self.encoder = encoder
               
    def learner_importance(self, method, nr_refits, test_size, replace=False, sampler=None, loss=None, **kwargs):
        """
        Compute feature importance over multiple refits of the model.
        
        Args:
            method: Name of the feature importance method to compute.
            nr_refits: Number of refits to perform.
            replace: Whether to sample with replacement.
        """
        if method not in ['cfi']:
            raise NotImplementedError('Method not implemented.')
        
        tmp0, tmp1 = train_test_split(self.y, test_size=test_size)
        nr_test = tmp1.shape[0]
        nr_train = tmp0.shape[0]
        
        index = create_multiindex(['model', 'i'],
                                  [np.arange(nr_refits),
                                   np.arange(nr_test)])
                
        scoress = []
        
        for mm in tqdm.tqdm(range(nr_refits)):
            model = clone(self.learner)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
            model.fit(X_train, y_train)
            self.sampler.update_data(X_train)
            predict = model.predict if not self._isclassifier else model.predict_proba
            
            wrk = Explainer(predict, X_train, loss=self.loss, sampler=self.sampler, encoder=self.encoder, fsoi=self.fsoi)
            if method == 'cfi':
                ex = wrk.cfi(X_test, y_test, fsoi=self.fsoi, nr_runs=1, **kwargs)
            scores = ex.scores[self.fsoi]
            mix = pd.MultiIndex.from_arrays([[mm]*nr_test, X_test.index.values], names=['fit', 'i'])
            scores.index = mix
            scoress.append(scores)
            
        scores = pd.concat(scoress)
        ex_description = 'Learner ' + ex.ex_description + f' with {nr_refits} refits, test size {test_size}, and '
        ex_description += f'{"bootstrapping" if replace else "subsampling"}.'
        lex = LearnerExplanation(self.fsoi, scores, (nr_train, nr_test), ex_name='learner-'+ex.ex_name, 
                                 ex_description=ex_description)
        return lex