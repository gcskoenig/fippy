# TODO Is there a nicer way for the experiment file than the command line arguments
"""
Experiment file for csl sage with mixed (cont/discrete) DAGs

Compute SAGE and store

Command line args:
    --data CSV file in folder ~/data/ (string without suffix)
    --model choice between linear model ('lm') and random forest regression ('rf')
    --size slice dataset to df[0:size] (int)
    --runs nr_runs in explainer.sage()
    --orderings nr_orderings in explainer.sage()
    --thresh threshold for convergence detection

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rfi.explainers.explainer import Explainer
from rfi.samplers import SequentialSampler, GaussianSampler, UnivRFSampler  # TODO (cl) review pr for sampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
import time
import argparse


parser = argparse.ArgumentParser(
    description="One SAGE Evaluation")

parser.add_argument(
    "-d",
    "--data",
    type=str,
    default="dag_s",
    help="Dataset from ~/data/ folder; string with suffix; default: 'dag_s.csv'")

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="rf",
    help="So far only random forest classifier ('rf') implemented; default: 'rf'")

parser.add_argument(
    "-n",
    "--size",
    type=int,
    default=None,
    help="Custom sample size to slice df, default: None",   # TODO (cl) slice or random draw?
)

parser.add_argument(
    "-r",
    "--runs",
    type=int,
    default=5,
    help="Number of runs for each SAGE estimation; default: 5",
)

parser.add_argument(
    "-o",
    "--orderings",
    type=int,
    default=20,
    help="Number of orderings; default : 20",
)

parser.add_argument(
    "-t",
    "--thresh",
    type=float,
    default=0.025,
    help="Threshold for convergence detection; default: 0.025",     # TODO (cl) rewrite convergence detection file
)

parser.add_argument(
    "-s",
    "--split",
    type=float,
    default=0.2,
    help="Train test split; default: 0.2 (test set size)",
)

parser.add_argument(
    "-rs",
    "--randomseed",
    type=int,
    default=1902,
    help="Numpy random seed; default: 1902",
)

arguments = parser.parse_args()

# seed
np.random.seed(arguments.randomseed)


def main(args):
    # TODO (cl) create directory here
    savepath = f"scripts/csl-experiments/results/continuous/{args.data}"

    # df to store some metadata TODO (cl) do we need to store any other data?
    col_names_meta = ["data", "model", "runtime"]
    metadata = pd.DataFrame(columns=col_names_meta)

    # import and prepare data TODO (cl) or provide full path as command line argument?
    df = pd.read_csv(f"scripts/csl-experiments/data/{args.data}.csv")
    if args.size is not None:
        df = df[0:args.size]
    col_names = df.columns.tolist()
    target = np.random.choice(col_names)    # TODO (cl) can classifier handle either data class (cont/discrete)?
    col_names.remove(target)
    X = df[col_names]
    y = df[target]

    # split data for train and test purpose
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.split, random_state=args.randomseed
    )

    # capture model performance
    col_names_model = ["data", "model", "target", "acc"]
    model_details = pd.DataFrame(columns=col_names_model)

    # fit model
    model = RandomForestClassifier(n_estimators=100)     # TODO (cl) command line argument?
    model.fit(X_train, y_train)
    # model evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # fill df with info about model
    model_details.loc[len(model_details)] = [args.data, "rf classifier", target, acc]
    model_details.to_csv(
        f"{savepath}/model_details_rf.csv", index=False
    )

    # model prediction linear model
    def model_predict(x):
        return model.predict(x)

    # set up sampler and decorrelator TODO (cl) let sampler know what rows are cat and cont
    cat = []
    cont = []
    sampler_cat = UnivRFSampler(X_train[cat])
    sampler_cont = GaussianSampler(X_train[cont])
    # TODO (cl) do I have to provide adjacency matrix? If yes, infer graph first before executing script
    # TODO (cl) provide column names of categorical data
    sampler = SequentialSampler(X_train, adj_mix, cat, cont_sampler=sampler_cont,
                                    cat_sampler=sampler_cat)
    # TODO (cl) what decorrelator to use?
    decorrelator = NaiveGaussianDecorrelator(X_train)

    # features of interest
    fsoi = X_train.columns

    # SAGE explainer    # TODO What loss function?
    wrk = Explainer(model_predict, fsoi, X_train, loss=accuracy_score, sampler=sampler,
                    decorrelator=decorrelator)

    # NO partial order
    partial_order = [tuple(X_train.columns)]

    # track time with time module
    start_time = time.time()
    ex_d_sage, orderings_sage = wrk.sage(X_test, y_test, partial_order, nr_orderings=args.orderings,
                                         nr_runs=args.runs, detect_convergence=False, thresh=args.thresh)
    time_sage = time.time() - start_time

    # save  orderings
    orderings_sage.to_csv(f'{savepath}/order_sage_{args.data}_{args.model}.csv')

    # save SAGE values for every ordering (Note: not split by runs anymore)
    sage_values_ordering = ex_d_sage.scores.mean(level=0)
    sage_values_ordering.to_csv(f"{savepath}/sage_o_{args.data}_{args.model}.csv")

    # fi_values for the runs
    ex_d_sage.fi_vals().to_csv(f"{savepath}/sage_r_{args.data}_{args.model}.csv")

    # fi_mean values across runs + stds
    ex_d_sage.fi_means_stds().to_csv(f"{savepath}/sage_{args.data}_{args.model}.csv")

    content = [args.data, args.model, time_sage]
    # fill evaluation table with current run
    metadata.loc[len(metadata)] = content
    metadata.to_csv(f"{savepath}/metadata_{args.data}_{args.model}.csv", index=False)


if __name__ == "__main__":
    main(arguments)
