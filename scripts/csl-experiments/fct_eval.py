""" Separate function evaluations of all functions potentially spared when d-sep found. """
# TODO (cl): add note about complexity to functions (e.g. nx.dseparated scales linearly in d)
# TODO (cl): check what's not listed in requirements (should be underlined)


import networkx as nx
import numpy as np
import scipy.special as sp
import pandas as pd
from time import time
# from rfi.samplers import SequentialSampler, GaussianSampler, UnivRFSampler
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import GaussianDecorrelator, NaiveGaussianDecorrelator
from rfi.decorrelators.naive import NaiveDecorrelator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import convert_amat

# fix some inputs   # TODO (cl): command line or other way of inputting this?
nr_resample_marginalize = 5
nr_runs = 5
orderings = 20
target = "10"

# load data # TODO: execute this file once for every data set or with flexible data input?
df = pd.read_csv("scripts/csl-experiments/data/dag_s_0.22222.csv")
df = df[0:1000]   # for test purposes only
adj_mat = pd.read_csv("scripts/csl-experiments/data/true_amat/dag_s_0.22222.csv")
adj_mat = convert_amat(adj_mat, col_names=True)

# initiate time # TODO (cl): more elegant way?
time_dsep, time_train_sampler, time_train_decor, time_sampler, time_decor, time_model, time_loss = 0, 0, 0, 0, 0, 0, 0
spared_train_sampler, spared_train_decor, spared_sampler, spared_decor, spared_model, spared_loss = 0, 0, 0, 0, 0, 0
time_complete, spared_complete = 0, 0   # TODO (cl): wrap around the if statements
# count d-seps
dseps = 0

# create graph

g = nx.DiGraph(adj_mat)
predictors = df.columns.drop(target)    # drop target node TODO flexible target node

# split data for train and test purpose
X_train, X_test, y_train, y_test = train_test_split(
    df[predictors], df[target], test_size=0.2, random_state=42
)
# fit model # TODO (cl) flexible model (or train model once during experiment and save it + call it?)
# TODO (cl): if not saved model, use same seed for model training as in experiments
model = LinearRegression()
model.fit(X_train, y_train)

# set up sampler and decorrelator
sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)

# NOTE: we go through permutations like in SAGE sampling
# TODO (cl): go through same permutations as in SAGE experiments (i.e. use the saved permutations)
orderings_count = 0
while orderings_count < orderings:
    # TODO (cl): Either randomly sampled perturbations OR saved orderings from experiment
    perturbed = np.random.permutation(predictors)

    for i in range(len(perturbed)):
        J = perturbed[i]    # feature of interest
        C = perturbed[0:i]      # coalition
        D = list(predictors)
        K = list(predictors)
        R = list(set(D) - set(C))  # background non-coalition variables
        R_ = list(set(R) - set(J))  # foreground non-coalition
        CuJ = list(set(C).union(J))  # foreground coalition variables

        # first: nx.d-separated as 'additional effort'
        start_dsep = time()
        d_sep = nx.d_separated(g, set(J), {target}, set(C))
        time_single_dsep = time() - start_dsep
        time_dsep += time_single_dsep

        # TODO (cl) how much does everything depend on coalition size etc.?
        if not d_sep:
            # train sampler and decorrelator
            start_train_sampler = time()
            if not sampler.is_trained(R_, CuJ):  # sampler for foreground non-coalition
                # train if allowed, otherwise raise error
                sampler.train(R_, CuJ)
            else:
                txt = '\tCheck passed: Sampler is already trained on'
                txt = txt + '{}|{}'.format(R, J)
                print(txt)
            time_single_ts = time() - start_train_sampler
            time_train_sampler += time_single_ts

            # TODO: What datasets to actually use here? Train set is good (WHY AGAIN, note on May 24)
            start_train_decor = time()
            if not decorrelator.is_trained(R, J, C):
                # train if allowed, otherwise raise error
                decorrelator.train(R, J, C)
            else:
                txt = '\tCheck passed: decorrelator is already trained on'
                txt = txt + '{} idp {}|{}'.format(R, J, C)
                print(txt)
            time_single_td = time() - start_train_decor
            time_train_decor += time_single_td

            runs = 0
            while runs < nr_runs:   # TODO go through runs to avg out inconsistencies (alt.: just multiply by nr_runs)
                start_sample = time()
                sample = sampler.sample(X_test, R_, CuJ, num_samples=nr_resample_marginalize)
                index = sample.index
                time_single_sample = time() - start_sample
                time_sampler += time_single_sample
                df_yh = pd.DataFrame(index=index,
                                     columns=['y_hat_baseline',
                                              'y_hat_foreground'])  # for model eval and loss later on

                resample = 0
                while resample < nr_resample_marginalize:   # cf. ll. 374ff. in explainer.py
                    X_tilde_baseline = X_test.copy()
                    X_tilde_foreground = X_test.copy()
                    arr_reconstruction = sample.loc[resample, R_].to_numpy()
                    X_tilde_foreground[R_] = arr_reconstruction
                    start_decor = time()
                    decor = decorrelator.decorrelate(X_tilde_foreground, R, J, C)
                    time_single_decor = time() - start_decor
                    time_decor += time_single_decor
                    arr_decorr = decor[R].to_numpy()

                    X_tilde_baseline[R] = arr_decorr
                    X_tilde_foreground_partial = X_tilde_baseline.copy()
                    X_tilde_foreground_partial[K] = X_tilde_foreground[K].to_numpy()

                    # make sure model can handle it (selection and ordering)
                    X_tilde_baseline = X_tilde_baseline[D]
                    X_tilde_foreground_partial = X_tilde_foreground_partial[D]

                    # model evaluations (lin. reg., for example, is O(n))
                    start_model = time()
                    # create and store prediction
                    y_hat_baseline = model.predict(X_tilde_baseline)
                    y_hat_foreground = model.predict(X_tilde_foreground_partial)
                    time_single_model = time() - start_model
                    time_model += time_single_model
                    resample += 1

                    df_yh.loc[resample-1, 'y_hat_baseline'] = np.array(y_hat_baseline)
                    df_yh.loc[resample-1, 'y_hat_foreground'] = np.array(y_hat_foreground)

                # convert and aggregate predictions
                df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                      'y_hat_foreground': 'float'})
                df_yh = df_yh.groupby(level='i').mean()

                # typically, two loss evaluations (i.e. target = Y in ai_via)
                start_loss = time()
                loss_baseline = mean_squared_error(y_test, df_yh['y_hat_baseline'])
                loss_foreground = mean_squared_error(y_test, df_yh['y_hat_foreground'])
                diffs = (loss_baseline - loss_foreground)
                time_single_loss = time() - start_loss
                time_loss += time_single_loss
                runs += 1

        ### FROM HERE ON: IF D-SEP IS PRESENT, i.e. what rly is saved
        if d_sep:
            dseps += 1
            # train sampler and decorrelator
            start_train_sampler = time()
            if not sampler.is_trained(R_, CuJ):  # sampler for foreground non-coalition
                # train if allowed, otherwise raise error
                sampler.train(R_, CuJ)
            else:
                txt = '\tCheck passed: Sampler is already trained on'
                txt = txt + '{}|{}'.format(R, J)
                print(txt)
            time_single_ts = time() - start_train_sampler
            spared_train_sampler += time_single_ts

            start_train_decor = time()
            if not decorrelator.is_trained(R, J, C):
                # train if allowed, otherwise raise error
                decorrelator.train(R, J, C)
            else:
                txt = '\tCheck passed: decorrelator is already trained on'
                txt = txt + '{} idp {}|{}'.format(R, J, C)
                print(txt)
            time_single_td = time() - start_train_decor
            spared_train_decor += time_single_td

            runs = 0
            while runs < nr_runs:
                start_sample = time()
                sample = sampler.sample(X_test, R_, CuJ, num_samples=nr_resample_marginalize)
                index = sample.index
                time_single_sample = time() - start_sample
                spared_sampler += time_single_sample
                df_yh = pd.DataFrame(index=index,
                                     columns=['y_hat_baseline',
                                              'y_hat_foreground'])  # for model eval and loss later on

                resample = 0
                while resample < nr_resample_marginalize:   # cf. ll. 374ff. in explainer.py
                    X_tilde_baseline = X_test.copy()
                    X_tilde_foreground = X_test.copy()
                    arr_reconstruction = sample.loc[resample, R_].to_numpy()
                    X_tilde_foreground[R_] = arr_reconstruction
                    start_decor = time()
                    decor = decorrelator.decorrelate(X_tilde_foreground, R, J, C)
                    time_single_decor = time() - start_decor
                    spared_decor += time_single_decor
                    arr_decorr = decor[R].to_numpy()

                    X_tilde_baseline[R] = arr_decorr
                    X_tilde_foreground_partial = X_tilde_baseline.copy()
                    X_tilde_foreground_partial[K] = X_tilde_foreground[K].to_numpy()

                    # make sure model can handle it (selection and ordering)
                    X_tilde_baseline = X_tilde_baseline[D]
                    X_tilde_foreground_partial = X_tilde_foreground_partial[D]

                    # model evaluations (lin. reg., for example, is O(n))
                    start_model = time()
                    # create and store prediction
                    y_hat_baseline = model.predict(X_tilde_baseline)
                    y_hat_foreground = model.predict(X_tilde_foreground_partial)
                    time_single_model = time() - start_model
                    spared_model += time_single_model
                    resample += 1

                    df_yh.loc[resample-1, 'y_hat_baseline'] = np.array(y_hat_baseline)
                    df_yh.loc[resample-1, 'y_hat_foreground'] = np.array(y_hat_foreground)

                # convert and aggregate predictions
                df_yh = df_yh.astype({'y_hat_baseline': 'float',
                                      'y_hat_foreground': 'float'})
                df_yh = df_yh.groupby(level='i').mean()

                # typically, two loss evaluations (i.e. target = Y in ai_via)
                start_loss = time()
                loss_baseline = mean_squared_error(y_test, df_yh['y_hat_baseline'])     # TODO: flexible loss argument
                loss_foreground = mean_squared_error(y_test, df_yh['y_hat_foreground'])
                diffs = (loss_baseline - loss_foreground)
                time_single_loss = time() - start_loss
                spared_loss += time_single_loss
                runs += 1
    orderings_count += 1

# save the results in a table
summary1 = pd.DataFrame(columns=['dsep', 'train sampler', 'train decor', 'sampler', 'decor', 'model', 'loss'])
times1 = [time_dsep, time_train_sampler, time_train_decor, time_sampler, time_decor, time_model, time_loss]
summary1.loc[0] = times1

summary2 = pd.DataFrame(columns=['dsep', 'train sampler saved', 'train decor saved', 'sampler saved', 'decor saved',
                                 'model saved', 'loss saved'])
times2 = [time_dsep, spared_train_sampler, spared_train_decor, spared_sampler, spared_decor,
          spared_model, spared_loss]
summary2.loc[0] = times2

# TODO for total time, sum up then round
total = len(predictors)*orderings   # number of orderings*number of predictors
print(summary1)
print(summary2)
print(dseps)
print(total)
# TODO: For true and estimated adjacency matrix

