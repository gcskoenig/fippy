import shap
import lightgbm as lgb

import mlflow
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import importlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from os.path import dirname, abspath
import numpy as np
import inspect
import logging

from rfi.backend.utils import flatten_dict
from rfi.backend.causality import DirectedAcyclicGraph
import rfi.explainers.explainer as explainer
from rfi.utils import search_nonsorted, check_existing_hash


logger = logging.getLogger(__name__)
ROOT_PATH = dirname(abspath(__file__))


@hydra.main(config_path='configs', config_name='config.yaml')
def main(args: DictConfig):

    exp_name = 'census'

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)

    # Dataset loading
    data_df, y = shap.datasets.adult()
    target_var = 'salary'
    input_vars = list(data_df.columns)
    data_df[target_var] = y
    cat_inputs = search_nonsorted(data_df.columns.values, data_df.select_dtypes(exclude=[np.floating]).columns.values)

    train_df, test_df = train_test_split(data_df, test_size=args.data.test_ratio, random_state=args.data.split_seed)
    y_train, X_train = train_df.loc[:, target_var].values, train_df.loc[:, input_vars].values
    y_test, X_test = test_df.loc[:, target_var].values, test_df.loc[:, input_vars].values

    # Adding default estimator params
    default_names, _, _, default_values, _, _, _ = \
        inspect.getfullargspec(instantiate(args.estimator, context_size=0).__class__.__init__)
    if default_values is not None:
        args.estimator['defaults'] = {
            n: str(v) for (n, v) in zip(default_names[len(default_names) - len(default_values):], default_values)
        }
    logger.info(OmegaConf.to_yaml(args, resolve=True))

    # Experiment tracking
    mlflow.set_tracking_uri(args.exp.mlflow_uri)
    mlflow.set_experiment(exp_name)

    # Checking if run exist
    if check_existing_hash(args, exp_name):
        logger.info('Skipping existing run.')
        return
    else:
        logger.info('No runs found - perfoming one.')

    mlflow.start_run()
    mlflow.log_params(flatten_dict(args))
    mlflow.log_param('data/n_train', len(train_df))
    mlflow.log_param('data/n_test', len(test_df))

    # Saving artifacts
    train_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/train_df.csv'), index=False)
    test_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/test_df.csv'), index=False)
    # df = pd.concat([train_df, test_df], keys=['train', 'test']).reset_index().drop(columns=['level_1'])
    # g = sns.pairplot(df, plot_kws={'alpha': 0.25}, hue='level_0')
    # g.fig.suptitle(exp_name)
    # plt.savefig(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/data.png'))

    results = {}

    # Initialising risks
    risks = {}
    for risk in args.predictors.risks:
        risks[risk] = getattr(importlib.import_module('sklearn.metrics'), risk)

    # Fitting predictive model
    models = {}
    for pred_model in args.predictors.pred_models:
        logger.info(f'Fitting {pred_model._target_} for target = {target_var} and inputs {input_vars}')
        model = instantiate(pred_model, categorical_feature=cat_inputs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        models[pred_model._target_] = model
        for risk, risk_func in risks.items():
            results[f'test_{risk}_{pred_model._target_}'] = risk_func(y_test, y_pred)

    sampler = instantiate(args.estimator.sampler, X_train=X_train, fit_method=args.estimator.fit_method,
                          fit_params=args.estimator.fit_params, cat_inputs=cat_inputs)

    G = np.arange(2, 12)
    foi = [0]
    estimator = sampler.train(foi, G)
    results['rfi/gof/mean_log_lik_1'] = estimator.log_prob(inputs=X_test[:, foi], context=X_test[:, G]).mean()
    estimator.sample(context=X_test[:, G], num_samples=10)

    G = np.arange(2, 12)
    foi = [1]
    estimator = sampler.train(foi, G)
    results['rfi/gof/mean_log_lik_2'] = estimator.log_prob(inputs=X_test[:, foi], context=X_test[:, G]).mean()
    estimator.sample(context=X_test[:, G], num_samples=10)

    mlflow.log_metrics(results, step=0)


if __name__ == "__main__":
    main()
