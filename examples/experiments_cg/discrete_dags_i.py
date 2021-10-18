"""
Experiment file for CGExplainer with discrete DAGs

Command line args:
    --data CSV file in folder ~/data/ (string without suffix)
    --model choice between categorical naive Bayes ('cnb') and random forest classification ('rf')
    --size slice dataset to df[0:size] (int)
    --runs nr_runs in explainer.sage()
    --orderings nr_orderings in explainer.sage()
    --thresh threshold for convergence detection

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from rfi.explainers.cgexplainer import CGExplainer
from rfi.explainers.explainer import Explainer
from rfi.samplers.simple import SimpleSampler
from rfi.decorrelators.naive import NaiveDecorrelator
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
    default="cnb",
    help="categorical naive Bayes ('cnb') or random forest classification ('rf')?",
)

parser.add_argument(
    "-n",
    "--size",
    type=int,
    default=None,
    help="Custom sample size to slice df",
)

parser.add_argument(
    "-r",
    "--runs",
    type=int,
    default=5,
    help="Number of runs",
)

parser.add_argument(
    "-o",
    "--orderings",
    type=int,
    default=20,
    help="Number of orderings",
)

parser.add_argument(
    "-t",
    "--thresh",
    type=float,
    default=0.025,
    help="Threshold for convergence detection",
)

parser.add_argument(
    "-s",
    "--split",
    type=float,
    default=0.2,
    help="Train test split",
)

parser.add_argument(
    "-y",
    "--target",
    type=str,
    default="dysp",
    help="Target node of models",
)

arguments = parser.parse_args()
# seed
np.random.seed(1902)


def main(args):
    savepath_true = f"examples/experiments_cg/results/discrete/true_amat/{args.data}"
    savepath_est = f"examples/experiments_cg/results/discrete/est_amat/{args.data}"
    # df to store some metadata
    col_names = ["data", "model", "runtime sage", "runtime cg",
                 "runtime cg cd", "runtime cg est", "runtime cg est"]
    metadata = pd.DataFrame(columns=col_names)
    # import and prepare data
    df = pd.read_csv(f"examples/experiments_cg/data/{args.data}.csv")
    if args.size is not None:
        df = df[0:args.size]
    col_names = df.columns.tolist()
    col_names.remove(args.target)
    X = df[col_names]
    y = df[args.target]
    # split data for train and test purpose
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.split, random_state=42
    )
    # initiate df for details of models
    col_names = ["data", "model", "target", "accuracy"]
    model_details = pd.DataFrame(columns=col_names)

    # fit model
    if args.model == "cnb":
        # fit model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        # model evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # fill df with info about model
        model_details.loc[len(model_details)] = [args.data, "cnb", args.target, acc]
        model_details.to_csv(
            f"examples/experiments_cg/results/discrete/true_amat/{args.data}/model_details_cnb.csv", index=False
        )
    else:
        # fit model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        # model evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # fill df with info about model
        model_details.loc[len(model_details)] = [args.data, "rf cf", args.target, acc]
        model_details.to_csv(
            f"examples/experiments_cg/results/discrete/true_amat/{args.data}/model_details_rf.csv", index=False
        )
    # load adjacency matrices for CGExplainer
    amat_true = pickle.load(open(f"examples/experiments_cg/data/{args.data}.p", "rb"))
    amat_est = pickle.load(open(f"examples/experiments_cg/data/{args.data}_est.p", "rb"))

    def model_predict(x_in):
        predictio = model.predict_proba(x_in)
        return predictio[:, 1]


    def loss_fn(y_true, y_predict):
        # y_predict.where(y_predict < 0.5, 0, inplace=True)
        # y_predict.where(y_predict >= 0.5, 1, inplace=True)
        predi = pd.DataFrame()
        predi["0"] = 1 - y_predict
        predi["1"] = y_predict
        return log_loss(y_true, predi)

    # set up sampler and decorrelator (same for Explainer and CGExplainer)
    sampler = SimpleSampler(X_train)
    decorrelator = NaiveDecorrelator(X_train, sampler=sampler)
    # features of interest
    fsoi = X_train.columns
    # SAGE explainer
    wrk = Explainer(model_predict, fsoi, X_train, loss=loss_fn, sampler=sampler,
                    decorrelator=decorrelator)
    # NO partial order
    partial_order = [tuple(X_train.columns)]
    # track time with time module
    start_time = time.time()
    ex_d_sage, orderings_sage = wrk.sage(X_test, y_test, partial_order, nr_orderings=args.orderings,
                                         nr_runs=args.runs, detect_convergence=True, thresh=args.thresh)
    time_sage = time.time() - start_time
    # CGExplainer
    wrk_cg_true = CGExplainer(model_predict, fsoi, X_train, amat_true, loss=loss_fn,
                              sampler=sampler, decorrelator=decorrelator)
    # CG Sage run with same orderings as SAGE run
    start_time_cg = time.time()
    ex_d_cg, ordering_cg = wrk_cg_true.sage(X_test, y_test, partial_order,
                                            nr_orderings=orderings_sage.shape[0],
                                            nr_runs=args.runs, orderings=orderings_sage)
    time_cg = time.time() - start_time_cg
    # CGExplainer (with estimated amat)
    wrk_cg_est = CGExplainer(model_predict, fsoi, X_train, amat_est, loss=loss_fn,
                              sampler=sampler, decorrelator=decorrelator)
    # CG Sage run with same orderings as SAGE run
    start_time_cg_est = time.time()
    ex_d_cg_est, ordering_cg_est = wrk_cg_est.sage(X_test, y_test, partial_order,
                                            nr_orderings=orderings_sage.shape[0],
                                            nr_runs=args.runs, orderings=orderings_sage)
    time_cg_est = time.time() - start_time_cg_est
    # save  orderings
    orderings_sage.to_csv(f'{savepath_true}/order_sage_{args.data}_{args.model}.csv')
    # save the SAGE/cg values for every ordering (Note: not split by runs anymore)
    sage_values_ordering = ex_d_sage.scores.mean(level=0)
    sage_values_ordering.to_csv(f"{savepath_true}/sage_o_{args.data}_{args.model}.csv")
    cg_values = ex_d_cg.scores.mean(level=0)
    cg_values.to_csv(f"{savepath_true}/cg_o_{args.data}_{args.model}.csv")
    cg_values_est = ex_d_cg_est.scores.mean(level=0)
    cg_values_est.to_csv(f"{savepath_est}/cg_o_{args.data}_{args.model}.csv")
    # fi_values for the runs
    ex_d_sage.fi_vals().to_csv(f"{savepath_true}/sage_r_{args.data}_{args.model}.csv")
    ex_d_cg.fi_vals().to_csv(f"{savepath_true}/cg_r_{args.data}_{args.model}.csv")
    ex_d_cg_est.fi_vals().to_csv(f"{savepath_est}/cg_r_{args.data}_{args.model}.csv")
    # fi_mean values across runs + stds
    ex_d_sage.fi_means_stds().to_csv(f"{savepath_true}/sage_{args.data}_{args.model}.csv")
    ex_d_cg.fi_means_stds().to_csv(f"{savepath_true}/cg_{args.data}_{args.model}.csv")
    ex_d_cg_est.fi_means_stds().to_csv(f"{savepath_est}/cg_{args.data}_{args.model}.csv")
    time_cg_cd = "n/a"
    time_cg_cd_est = "n/a"
    content = [args.data, args.model, time_sage, time_cg, time_cg_cd, time_cg_est, time_cg_cd_est]
    # fill evaluation table with current run
    metadata.loc[len(metadata)] = content
    metadata.to_csv(f"{savepath_true}/metadata_{args.data}_{args.model}.csv", index=False)


if __name__ == "__main__":
    main(arguments)
