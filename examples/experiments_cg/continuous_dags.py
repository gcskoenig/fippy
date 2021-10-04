"""
Experiment file for CGExplainer with continuous DAGs

Command line args:
    --data CSV file in folder ~/data/ (string without suffix)
    --amat indicator for true or estimated adjacency matrix ('true' or 'est')
    --model choice between linear model ('lm') and random forest regression ('rf')
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
    "-a",
    "--amat",
    type=str,
    default="true",
    help="Adjacency matrix true or est?")

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="lm",
    help="linear model ('lm') or random forest regression ('rf')?",
)

arguments = parser.parse_args()

# seed
np.random.seed(1902)


def main(args):

    if args.amat == "true":
        savepath = f"examples/experiments_cg/results/continuous/true_amat/{args.data}"
    elif args.amat == "est":
        savepath = f"examples/experiments_cg/results/continuous/est_amat/{args.data}"
    else:
        raise NameError("Adjacency matrix can be either true ('true') or estimated ('est')")

    # df to store some metadata
    col_names = ["data", "model", "adjacency matrix", "runtime sage", "runtime cg", "runtime cg cd"]
    metadata = pd.DataFrame(columns=col_names)

    # import and prepare data
    df = pd.read_csv(f"examples/experiments_cg/data/{args.data}.csv")
    df = df[0:100]
    col_names = df.columns.tolist()
    col_names.remove("1")
    X = df[col_names]
    y = df["1"]

    # split data for train and test purpose
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    try:
        model_details = pd.read_csv("examples/experiments_cg/models/model_details.csv")
    except:
        # initiate df for details of models
        col_names = ["data", "model", "target", "mse", "R2"]
        model_details = pd.DataFrame(columns=col_names)

    # fit model
    if args.model == "lm":
        # fit model
        model = LinearRegression()
        model.fit(X_train, y_train)
        # model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # fill df with info about model
        model_details.loc[len(model_details)] = [args.data, "lin reg", "1", mse, r2]
        model_details.to_csv(
            "examples/experiments_cg/models/model_details.csv", index=False
        )
    else:
        # fit model
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        # model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # fill df with info about model
        model_details.loc[len(model_details)] = [args.data, "rf reg", "1", mse, r2]
        model_details.to_csv(
            "examples/experiments_cg/models/model_details.csv", index=False
        )

    # load corresponding adjacency matrix for CGExplainer
    if args.amat == "true":
        amat = pickle.load(open(f"examples/experiments_cg/data/{args.data}.p", "rb"))
    else:
        amat = pickle.load(open(f"examples/experiments_cg/data/{args.data}_est.p", "rb"))

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

    # track time with time module
    start_time = time.time()
    ex_d_sage, orderings_sage = wrk.sage(X_test, y_test, partial_order, nr_orderings=20, nr_runs=5,
                                         detect_convergence=True, thresh=0.4)
    time_sage = time.time() - start_time

    # CGExplainer
    wrk_cg = CGExplainer(model_predict, fsoi, X_train, amat, loss=mean_squared_error, sampler=sampler,
                         decorrelator=decorrelator)

    # CG Sage run with same orderings as SAGE run
    start_time_cg = time.time()
    ex_d_cg, ordering_cg = wrk_cg.sage(X_test, y_test, partial_order, nr_orderings=orderings_sage.shape[0],
                                       nr_runs=5, orderings=orderings_sage)
    time_cg = time.time() - start_time_cg

    # Separate CG SAGE run with convergence detection
    start_time_cg_cd = time.time()
    ex_d_cg_cd, orderings_cg_cd = wrk_cg.sage(X_test, y_test, partial_order, nr_runs=5, nr_orderings=20,
                                              detect_convergence=True, thresh=0.4)
    time_cg_cd = time.time() - start_time_cg_cd

    # save  orderings
    orderings_sage.to_csv(f'{savepath}/order_sage_{args.data}_{args.model}.csv')
    orderings_cg_cd.to_csv(f'{savepath}/order_cg_cd_{args.data}_{args.model}.csv')

    # save the SAGE/cg values for every ordering (Note: not split by runs anymore)
    sage_values_ordering = ex_d_sage.scores.mean(level=0)
    sage_values_ordering.to_csv(f"{savepath}/sage_o_{args.data}_{args.model}.csv")
    cg_values = ex_d_cg.scores.mean(level=0)
    cg_values.to_csv(f"{savepath}/cg_o_{args.data}_{args.model}.csv")
    cg_cd_values = ex_d_cg_cd.scores.mean(level=0)
    cg_cd_values.to_csv(f"{savepath}/cg_cd_o_{args.data}_{args.model}.csv")

    # fi_values for the runs
    ex_d_sage.fi_vals().to_csv(f"{savepath}/sage_r_{args.data}_{args.model}.csv")
    ex_d_cg.fi_vals().to_csv(f"{savepath}/cg_r_{args.data}_{args.model}.csv")
    ex_d_cg_cd.fi_vals().to_csv(f"{savepath}/cg_cd_r_{args.data}_{args.model}.csv")

    # fi_mean values across runs + stds
    ex_d_sage.fi_means_stds().to_csv(f"{savepath}/sage_{args.data}_{args.model}.csv")
    ex_d_cg.fi_means_stds().to_csv(f"{savepath}/cg_{args.data}_{args.model}.csv")
    ex_d_cg_cd.fi_means_stds().to_csv(f"{savepath}/cg_cd_{args.data}_{args.model}.csv")

    content = [args.data, args.model, args.amat, time_sage, time_cg, time_cg_cd]
    # fill evaluation table with current run
    metadata.loc[len(metadata)] = content
    metadata.to_csv(f"{savepath}/metadata_{args.data}_{args.model}.csv", index=False)


if __name__ == "__main__":
    main(arguments)
