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
    data_df['Salary'] = y

    target_var = {'Salary'}
    all_inputs_vars = set(data_df.columns) - target_var
    sensetive_vars = set(list(args.exp.sensetive_vars))
    wo_sens_inputs_vars = all_inputs_vars - sensetive_vars
    cat_vars = set(data_df.select_dtypes(exclude=[np.floating]).columns.values)
    logger.info(f'Target var: {target_var}, all_inputs: {all_inputs_vars}, sensetive_vars: {sensetive_vars}, '
                f'cat_vars: {cat_vars}')

    train_df, test_df = train_test_split(data_df, test_size=args.data.test_ratio, random_state=args.data.split_seed)
    y_train, X_train, X_train_wo_sens = train_df[target_var], train_df[all_inputs_vars], train_df[wo_sens_inputs_vars]
    y_test, X_test, X_test_wo_sens = test_df[target_var], test_df[all_inputs_vars], test_df[wo_sens_inputs_vars]

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

    pred_results = {}

    # Initialising risks
    risks = {}
    pred_funcs = {}
    for risk in args.predictors.risks:
        risks[risk['name']] = getattr(importlib.import_module('sklearn.metrics'), risk['name'])
        pred_funcs[risk['name']] = risk['method']

    # Fitting predictive model
    models = {}
    models_pred_funcs = {}
    for pred_model in args.predictors.pred_models:
        cat_vars_wo_sens = cat_vars - sensetive_vars
        logger.info(f'Fitting {pred_model._target_} for target = {target_var} and inputs {wo_sens_inputs_vars} '
                    f'(categorical {cat_vars_wo_sens})')
        model = instantiate(pred_model, categorical_feature=search_nonsorted(list(wo_sens_inputs_vars), list(cat_vars_wo_sens)))
        model.fit(X_train_wo_sens, y_train)
        models[pred_model._target_] = model
        for risk, risk_func in risks.items():
            if pred_funcs[risk] == 'predict_proba':
                models_pred_funcs[risk] = lambda X_test: getattr(model, pred_funcs[risk])(X_test)[:, 1]
            else:
                models_pred_funcs[risk] = lambda X_test: getattr(model, pred_funcs[risk])(X_test)
            y_pred = models_pred_funcs[risk](X_test_wo_sens)
            pred_results[f'test_{risk}_{pred_model._target_}'] = risk_func(y_test, y_pred)

    mlflow.log_metrics(pred_results, step=0)

    sampler = instantiate(args.estimator.sampler, X_train=X_train.values, fit_method=args.estimator.fit_method,
                          fit_params=args.estimator.fit_params,
                          cat_inputs=search_nonsorted(list(all_inputs_vars), list(cat_vars)))

    wo_sens_inputs_vars_list = sorted(list(wo_sens_inputs_vars))
    mlflow.log_param('features_sequence', str(wo_sens_inputs_vars_list))
    mlflow.log_param('exp/cat_vars', str(sorted(list(cat_vars))))

    for i, fsoi_var in enumerate(wo_sens_inputs_vars_list):
        logger.info(f'Analysing the importance of feature: {fsoi_var}')

        interpret_results = {}
        fsoi = search_nonsorted(list(all_inputs_vars), [fsoi_var])
        R_j = search_nonsorted(list(all_inputs_vars), list(wo_sens_inputs_vars))

        # Permutation feature importance
        G_pfi, name_pfi = [], 'pfi'
        sampler.train(fsoi, G_pfi)

        # Conditional feature importance
        G_cfi, name_cfi = search_nonsorted(list(all_inputs_vars), list(wo_sens_inputs_vars - {fsoi_var})), 'cfi'
        estimator = sampler.train(fsoi, G_cfi)
        interpret_results['cfi/gof/mean_log_lik'] = estimator.log_prob(inputs=X_test.values[:, fsoi],
                                                                       context=X_test.values[:, G_cfi]).mean()

        # Relative feature importance (sensetive ignored vars)
        G_rfi, name_rfi = search_nonsorted(list(all_inputs_vars), list(sensetive_vars)), 'rfi'
        estimator = sampler.train(fsoi, G_rfi)
        interpret_results['rfi/gof/mean_log_lik'] = estimator.log_prob(inputs=X_test.values[:, fsoi],
                                                                       context=X_test.values[:, G_rfi]).mean()

        for model_name, model in models.items():
            for risk, risk_func in risks.items():
                rfi_explainer = explainer.Explainer(models_pred_funcs[risk], fsoi, X_train.values, sampler=sampler,
                                                    loss=risk_func, fs_names=list(all_inputs_vars))
                for G, name in zip([G_pfi, G_cfi, G_rfi], [name_pfi, name_cfi, name_rfi]):
                    mb_explanation = rfi_explainer.rfi(X_test.values, y_test, G, R_j, nr_runs=args.exp.rfi.nr_runs)
                    interpret_results[f'{name}/mean_{risk}_{model_name}'] = np.abs(mb_explanation.fi_vals(return_np=True)).mean()

        mlflow.log_metrics(interpret_results, step=i)


if __name__ == "__main__":
    main()
