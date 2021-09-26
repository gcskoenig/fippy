"""
Experiment file for CGExplainer with continuous DAGs
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rfi.explainers.cgexplainer import CGExplainer
from rfi.explainers.explainer import Explainer
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
import pickle
import time
import argparse


parser = argparse.ArgumentParser(
    description="Experiment to compare SAGE estimation with and without d-separation tests")

parser.add_argument(
    "-d",
    "--data",
    type=str,
    default="dag_s",
    help="What data to use?")

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="lm",
    help="linear model ('lm') or random forest regression ('rf')?",
)

args = parser.parse_args()

# seed
np.random.seed(1902)

# df to store some metadata
# file for comparison
try:
    metadata = pd.read_csv("experiments_cg/results/continuous/metadata.csv")
except:
    col_names = ["data", "model", "runtime sage", "runtime cg", "runtime cg cd"]
    metadata = pd.DataFrame(columns=col_names)

# import and prepare data
df = pd.read_csv(f"experiments_cg/data/{args.data}.csv")
df = df[0:1000]
col_names = df.columns.tolist()
col_names.remove("1")
X = df[col_names]
y = df["1"]

# split data for train and test purpose
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# load fitted model to infer SAGE values for
model = pickle.load(open(f"experiments_cg/models/{args.data}_{args.model}.sav", "rb"))
# load corresponding adjacency matrix for CGExplainer
amat = pickle.load(open(f"experiments_cg/data/{args.data}.p", "rb"))


# model prediction linear model
def model_predict(x):
    return model.predict(x)


# set up sampler and decorrelator (same for Explainer and CGExplainer)
sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)

# features of interest
fsoi = X_train.columns

# SAGE explainer
wrk = Explainer(model_predict, fsoi, X_train, loss=mean_squared_error, sampler=sampler,
                decorrelator=decorrelator)

# NO partial order
partial_order = [tuple(X_train.columns)]

# track time with time module # TODO add convergence detection to sage and second cg_sage
start_time = time.time()
ex_d_sage, orderings_sage = wrk.sage(X_test, y_test, partial_order, nr_runs=2, nr_orderings=10)
time_sage = time.time() - start_time

# CGExplainer
wrk_cg = CGExplainer(model_predict, fsoi, X_train, amat, loss=mean_squared_error, sampler=sampler,
                     decorrelator=decorrelator)

# CG Sage run with same orderings as SAGE run
start_time_cg = time.time()
ex_d_cg, ordering_cg = wrk_cg.sage(X_test, y_test, partial_order, nr_orderings=orderings_sage.shape[0],
                      nr_runs=1, orderings=orderings_sage)
time_cg = time.time() - start_time_cg

# Separate CG SAGE run with convergence detection # TODO add convergence detection
start_time_cg_cd = time.time()
ex_d_cg_cd, orderings_cg_cd = wrk_cg.sage(X_test, y_test, partial_order, nr_runs=1, nr_orderings=1)
time_cg_cd = time.time() - start_time_cg_cd

# save  orderings
orderings_sage.to_csv(f'experiments_cg/results/continuous/orderings_sage_{args.data}_{args.model}.csv')
orderings_cg_cd.to_csv(f'experiments_cg/results/continuous/orderings_cg_cd_{args.data}_{args.model}.csv')

# save the SAGE/cg values for every ordering (note, not split by runs anymore)
sage_values_ordering = ex_d_sage.scores.mean(level=0)
sage_values_ordering.to_csv(f"experiments_cg/results/continuous/sage_o_{args.data}_{args.model}.csv")
cg_values = ex_d_cg.scores.mean(level=0)
cg_values.to_csv(f"experiments_cg/results/continuous/cg_o_{args.data}_{args.model}.csv")
cg_cd_values = ex_d_cg_cd.scores.mean(level=0)
cg_cd_values.to_csv(f"experiments_cg/results/continuous/cg_cd_o_{args.data}_{args.model}.csv")

# fi_vals for the runs
ex_d_sage.fi_vals().to_csv(f"experiments_cg/results/continuous/sage_r_{args.data}_{args.model}.csv")
ex_d_cg.fi_vals().to_csv(f"experiments_cg/results/continuous/cg_r_{args.data}_{args.model}.csv")
ex_d_cg_cd.fi_vals().to_csv(f"experiments_cg/results/continuous/cg_cd_r_{args.data}_{args.model}.csv")


# fi_mean vals across runs + stds
ex_d_sage.fi_means_stds().to_csv(f"experiments_cg/results/continuous/sage_{args.data}_{args.model}.csv")
ex_d_cg.fi_means_stds().to_csv(f"experiments_cg/results/continuous/cg_{args.data}_{args.model}.csv")
ex_d_cg_cd.fi_vals().to_csv(f"experiments_cg/results/continuous/cg_cd_{args.data}_{args.model}.csv")

content = [args.data, args.model, time_sage, time_cg, time_cg_cd]
# fill evaluation table with current run
metadata.loc[len(metadata)] = content
metadata.to_csv("experiments_cg/results/continuous/metadata.csv", index=False)
