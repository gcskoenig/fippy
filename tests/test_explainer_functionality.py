import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import category_encoders as ce

# import logging
# logging.basicConfig(level=logging.INFO)

## to specify by user
savepath = 'tests/results/'

## load and prepare data
sns.get_dataset_names()
df = sns.load_dataset('mpg')
df.dtypes
df.drop('name', axis=1, inplace=True)
if df.isna().any().any():
    df.dropna(inplace=True)

X, y = df.drop('mpg', axis=1), df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## create and fit pipeline
ohe = ce.OneHotEncoder()
rf = RandomForestRegressor()
pipe = make_pipeline(ohe, rf)

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


## setup fippy explainer to explain the model
from fippy.explainers import Explainer
from fippy.samplers import UnivRFSampler, ContUnivRFSampler, SequentialSampler
from fippy.explanation import Explanation


class TestExplainer:

    @staticmethod
    def test_setup_explainer():
        # create univariate categorical and continuous samplers and compile them to sequential sampler
        cat_fs = X_train.select_dtypes(include='object').columns # specify categorical features
        cat_sampler = UnivRFSampler(X_train, cat_inputs=cat_fs)
        cont_sampler = ContUnivRFSampler(X_train, cat_inputs=cat_fs)
        sampler = SequentialSampler(X_train, categorical_fs=cat_fs,
                                cont_sampler=cont_sampler, cat_sampler=cat_sampler)

        # create explainer
        TestExplainer.wrk = Explainer(pipe.predict, mean_squared_error, sampler, X_train)
        assert True


    def test_pfi(self):
        TestExplainer.test_setup_explainer()
        ex_pfi = TestExplainer.wrk.pfi(X_test, y_test)
        ex_pfi.hbarplot()
        ex_pfi.fi_means_stds()
        assert True
        

    def test_cfi(self):
        TestExplainer.test_setup_explainer()
        ex_cfi = TestExplainer.wrk.cfi(X_test, y_test)
        ex_cfi.hbarplot()
        ex_cfi.fi_means_stds()
        assert True


    def test_csage(self):
        TestExplainer.test_setup_explainer()
        ex_csage, sage_orderings = TestExplainer.wrk.csage(X_test, y_test, nr_orderings=20, nr_runs=3)
        ex_csage.hbarplot()
        ex_csage.fi_means_stds()
        assert True

    def test_msage(self):
        TestExplainer.test_setup_explainer()
        ex_msage, sage_orderings = TestExplainer.wrk.msage(X_test, y_test, nr_runs=3, detect_convergence=True)
        ex_msage.hbarplot()
        ex_msage.fi_means_stds()
        assert True
