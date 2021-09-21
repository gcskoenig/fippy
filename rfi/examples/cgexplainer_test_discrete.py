"""
Draft of experiment file for CGExplainer
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import time
from rfi.explainers.cgexplainer import CGExplainer
from rfi.explainers.explainer import Explainer
from rfi.samplers.simple import SimpleSampler_old
from rfi.decorrelators.simple import SimpleDecorrelator


# TODO(cl) unrelated to this file: I still think column names are converted to numeric values

# import and prepare data
df = pd.read_csv("rfi/examples/cg_files/asia_s.csv")
col_names = df.columns.tolist()
col_names.remove("dysp")
X = df[col_names]
y = df["dysp"]

# split data for train and test purpose
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

rf = pickle.load(open("rfi/examples/cg_files/asia_cnb.sav", "rb"))
amat = pickle.load(open("rfi/examples/cg_files/asia.p", "rb"))

print("locked and loaded")
# NOTE: Very inefficient but I would say not too much of a problem because if we would simply use a larger model
# we would also have a higher runtime and it is the same for CGExplainer and Explainer


def model_predict(x):
    # single value prediction
    value_pred = rf.predict(x)
    probs = rf.predict_proba(x)
    prediction = []
    for i in range(len(value_pred)):
        prediction.append(probs[i][value_pred[i]])
    return prediction


def loss_fn(y_true, y_est):
    return np.mean(-np.log(y_est))


print("model and loss defined")


sampler = SimpleSampler_old(X_train)
decorrelator = SimpleDecorrelator(X_train)
fsoi = X_train.columns

wrk = Explainer(model_predict, fsoi, X_train, loss=loss_fn, sampler=sampler,
                decorrelator=decorrelator)


wrk_cg = CGExplainer(model_predict, fsoi, X_train, amat, loss=loss_fn, sampler=sampler,
                     decorrelator=decorrelator)

print("Explainers initiated")

partial_order = (tuple(X_train.columns))

start_time = time.time()
ex_d_sage = wrk.sage(X_test, y_test, [partial_order], nr_runs=2, nr_resample_marginalize=2)
time_wo_cg = time.time() - start_time

# start_time_cg = time.time()
# # TODO fix issue with kwargs (cf cgexplainer), while not done: nr_runs=10
# ex_d_sage_cg = wrk_cg.sage(X_test, y_test, [partial_order], nr_runs=10, nr_resample_marginalize=10)
# print("time w/ CG:", time.time() - start_time_cg)
print("time w/o CG:", time_wo_cg)

# NOTE: gain here sometimes is marginal because few variables, i.e. few cond indeps + model that evaluates fast
# + some randomness in the choice of coalitions/permutations

print(ex_d_sage[0].fi_means_stds())
# print(ex_d_sage_cg[0].fi_means_stds())

# NOTE: at first glance, fi_means do not seem to be all downwards or all upwards biased (too few evidence)
