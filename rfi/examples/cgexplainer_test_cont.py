"""
Draft of experiment file for CGExplainer
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
from rfi.explainers.cgexplainer import CGExplainer
from rfi.explainers.explainer import Explainer
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
from sklearn.metrics import mean_squared_error
import mlflow


# TODO: set seeds

# import and prepare data
df = pd.read_csv("examples_cg/dag_s_test.csv")
col_names = df.columns.tolist()
col_names.remove("1")
X = df[col_names]
y = df["1"]

# split data for train and test purpose
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# load fitted model to infer SAGE values for
lm = pickle.load(open("examples_cg/dag_s_lm.sav", "rb"))
# load corresponding adjacency matrix for CGExplainer
amat = pickle.load(open("examples_cg/dag_s.p", "rb"))


# model prediction linear model
def model_predict(x):
    return lm.predict(x)


# set up sampler and decorrelator (same for Explainer and CGExplainer)
sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)

# features of interest
fsoi = X_train.columns


# standard explainer
wrk = Explainer(model_predict, fsoi, X_train, loss=mean_squared_error, sampler=sampler,
                decorrelator=decorrelator)

# CGExplainer
wrk_cg = CGExplainer(model_predict, fsoi, X_train, amat, loss=mean_squared_error, sampler=sampler,
                     decorrelator=decorrelator)

# NO partial order
partial_order = [tuple(X_train.columns)]

# track time with time module
start_time = time.time()
ex_d_sage, orderings_sage = wrk.sage(X_test, y_test, partial_order, nr_orderings=10,
                                     nr_runs=10, nr_resample_marginalize=10)
time_wo_cg = time.time() - start_time

start_time_cg = time.time()
# TODO fix issue with kwargs (cf cgexplainer), while not done: nr_runs=10
ex_d_sage_cg, orderings_cg = wrk_cg.sage(X_test, y_test, partial_order, nr_orderings=10,
                                         nr_runs=10, nr_resample_marginalize=10)
time_w_cg = time.time() - start_time_cg


# save the orderings to be able to trace back whether d-separated features (from y) have value function 0
# orderings too large to save as csv?! depends on no_orderings, etc
#orderings_sage.to_csv(f'/some/results_folder/orderings_sage_data_model.csv')
#orderings_cg.to_csv(f'/some/results_folder/orderings_cg_data_model.csv')
# save ex_d_sage.fi_vals (fis for every run) and fi_means_stds (fi over all runs)

