# Imported files in folder that is ignored

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
from rfi.explainers.cgexplainer import CGExplainer
from rfi.explainers.explainer import Explainer
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
from sklearn.metrics import mean_squared_error


# TODO(cl) unrelated to this file: I still think column names are converted to numeric values

# import and prepare data
df = pd.read_csv("examples_cg/cma_l.csv")
df = df[1000:2000]
col_names = df.columns.tolist()
col_names.remove("y")
X = df[col_names]
y = df["y"]

# split data for train and test purpose
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

lm = pickle.load(open("examples_cg/cma_l.sav", "rb"))
amat = pickle.load(open("examples_cg/cma_l_amat.p", "rb"))

y_pred = lm.predict(X_test)


def model_predict(x):
    # single value prediction
    return lm.predict(x)


sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)
fsoi = X_train.columns

wrk = Explainer(model_predict, fsoi, X_train, loss=mean_squared_error, sampler=sampler,
                  decorrelator=decorrelator)


wrk_cg = CGExplainer(model_predict, fsoi, X_train, amat, loss=mean_squared_error, sampler=sampler,
                  decorrelator=decorrelator)

partial_order = (tuple(X_train.columns))

start_time = time.time()
ex_d_sage = wrk.sage(X_test, y_test, [partial_order], nr_runs=10, nr_resample_marginalize=10)
time_wo_cg = time.time() - start_time

start_time_cg = time.time()
# TODO fix issue with kwargs (cf cgexplainer), while not done: nr_runs=10
ex_d_sage_cg = wrk_cg.sage(X_test, y_test, [partial_order], nr_runs=10, nr_resample_marginalize=10)
print("time w/ CG:", time.time() - start_time_cg)
print("time w/o CG:", time_wo_cg)

# NOTE: gain here sometimes is marginal because few variables, i.e. few cond indeps + model that evaluates fast
# + some randomness in the choice of coalitions/permutations

print(ex_d_sage[0].fi_means_stds())
print(ex_d_sage_cg[0].fi_means_stds())

# NOTE: at first glance, fi_means do not seem to be all downwards or all upwards biased (too few evidence)
