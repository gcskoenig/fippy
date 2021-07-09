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

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    args.exp.pop('rfi')

    # Adding default estimator params
    default_names, _, _, default_values, _, _, _ = \
        inspect.getfullargspec(instantiate(args.estimator, context_size=0).__class__.__init__)
    if default_values is not None:
        args.estimator['defaults'] = {
            n: str(v) for (n, v) in zip(default_names[len(default_names) - len(default_values):], default_values)
        }
    logger.info(OmegaConf.to_yaml(args, resolve=True))

    # Data-generating DAG
    data_path = hydra.utils.to_absolute_path(f'{ROOT_PATH}/{args.data.relative_path}')
    exp_name = args.data.relative_path.split('/')[-1]
    adjacency_matrix = np.load(f'{data_path}/DAG{args.data.sample_ind}.npy').astype(int)
    if exp_name == 'sachs_2005':
        var_names = np.load(f'{data_path}/sachs-header.npy')
    else:
        var_names = [f'x{i}' for i in range(len(adjacency_matrix))]
    dag = DirectedAcyclicGraph(adjacency_matrix, var_names)

    # Experiment tracking
    exp_name = f'sage/{exp_name}'
    mlflow.set_tracking_uri(args.exp.mlflow_uri)
    mlflow.set_experiment(exp_name)

    # Checking if run exist
    if check_existing_hash(args, exp_name):
        logger.info('Skipping existing run.')
        return
    else:
        logger.info('No runs found - perfoming one.')

    # Loading Train-test data
    data = np.load(f'{data_path}/data{args.data.sample_ind}.npy')
    if args.data.standard_normalize:
        if 'normalise_params' in args.data:
            standard_normalizer = StandardScaler(**args.data.normalise_params)
        else:
            standard_normalizer = StandardScaler()
        data = standard_normalizer.fit_transform(data)
    data_train, data_test = train_test_split(data, test_size=args.data.test_ratio, random_state=args.data.split_seed)
    train_df = pd.DataFrame(data_train, columns=dag.var_names)
    test_df = pd.DataFrame(data_test, columns=dag.var_names)

    mlflow.start_run()
    mlflow.log_params(flatten_dict(args))
    mlflow.log_param('data_generator/dag/n', len(var_names))
    mlflow.log_param('data_generator/dag/m', int(adjacency_matrix.sum()))
    mlflow.log_param('data/n_train', len(train_df))
    mlflow.log_param('data/n_test', len(test_df))

    # Saving artifacts
    train_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/train.csv'), index=False)
    test_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/test.csv'), index=False)
    dag.plot_dag()
    plt.savefig(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/dag.png'))

    mlflow.log_param('features_sequence', str(list(dag.var_names)))

    for var_ind, target_var in enumerate(dag.var_names):

        var_results = {}

        # Considering all the variables for input
        input_vars = [var for var in dag.var_names if var != target_var]
        y_train, X_train = train_df.loc[:, target_var], train_df.loc[:, input_vars]
        y_test, X_test = test_df.loc[:, target_var], test_df.loc[:, input_vars]

        # Initialising risks
        risks = {}
        for risk in args.predictors.risks:
            risks[risk] = getattr(importlib.import_module('sklearn.metrics'), risk)

        # Fitting predictive model
        models = {}
        for pred_model in args.predictors.pred_models:
            logger.info(f'Fitting {pred_model._target_} for target = {target_var} and inputs {input_vars}')
            model = instantiate(pred_model)
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_test.values)
            models[pred_model._target_] = model
            for risk, risk_func in risks.items():
                var_results[f'test_{risk}_{pred_model._target_}'] = risk_func(y_test.values, y_pred)

        # =================== Global SAGE ===================
        logger.info(f'Analysing the importance of features: {input_vars}')

        sampler = instantiate(args.estimator.sampler, X_train=X_train, fit_method=args.estimator.fit_method,
                              fit_params=args.estimator.fit_params)

        log_lik = []
        sage_explainer = explainer.Explainer(None, input_vars, X_train, sampler=sampler, loss=None)
        # Generating the same orderings across all the models and losses
        np.random.seed(args.exp.sage.orderings_seed)
        fixed_orderings = [np.random.permutation(input_vars) for _ in range(args.exp.sage.nr_orderings)]

        for model_name, model in models.items():
            for risk, risk_func in risks.items():
                sage_explainer.model = model.predict
                explanation, test_log_lik = sage_explainer.sage(X_test, y_test, loss=risk_func, fixed_orderings=fixed_orderings,
                                                                nr_runs=args.exp.sage.nr_runs, return_test_log_lik=True,
                                                                nr_resample_marginalize=args.exp.sage.nr_resample_marginalize)
                log_lik.extend(test_log_lik)
                fi = explanation.fi_vals().mean()

                for fsoi, input_var in enumerate(input_vars):
                    var_results[f'sage/mean_{risk}_{model_name}_{input_var}'] = fi[input_var]

        var_results['sage/mean_log_lik'] = np.mean(log_lik)
        var_results['sage/num_fitted_estimators'] = len(log_lik)

        mlflow.log_metrics(var_results, step=var_ind)

    mlflow.end_run()


if __name__ == "__main__":
    main()
