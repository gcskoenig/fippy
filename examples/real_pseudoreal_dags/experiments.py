import numpy as np
import logging
import hydra
import pandas as pd
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import importlib
from os.path import abspath, dirname

from rfi.backend.utils import flatten_dict
from rfi.backend.causality import DirectedAcyclicGraph
import rfi.explainers.explainer as explainer
from rfi.utils import search_nonsorted


logger = logging.getLogger(__name__)
ROOT_PATH = dirname(abspath(__file__))


@hydra.main(config_path='configs', config_name='config.yaml')
def main(args: DictConfig):

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
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

    # Train-test data
    data = np.load(f'{data_path}/data{args.data.sample_ind}.npy')
    data_train, data_test = train_test_split(data, test_size=args.data.test_ratio, random_state=args.data.split_seed)
    train_df = pd.DataFrame(data_train, columns=dag.var_names)
    test_df = pd.DataFrame(data_test, columns=dag.var_names)

    # Experiment tracking
    mlflow.set_tracking_uri(args.exp.mlflow_uri)
    mlflow.set_experiment(exp_name)
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
    if len(dag.var_names) <= 20:
        df = pd.concat([train_df, test_df], keys=['train', 'test']).reset_index().drop(columns=['level_1'])
        g = sns.pairplot(df, plot_kws={'alpha': 0.25}, hue='level_0')
        g.fig.suptitle(exp_name)
        plt.savefig(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/data.png'))

    metrics = {}

    for var_ind, target_var in enumerate(dag.var_names):

        # Considering all the variables for input
        input_vars = [var for var in dag.var_names if var != target_var]
        y_train, X_train = train_df.loc[:, target_var].values, train_df.loc[:, input_vars].values
        y_test, X_test = test_df.loc[:, target_var].values, test_df.loc[:, input_vars].values

        # Fitting predictive model
        logger.info(f'Fitting predictive model for target = {target_var} and inputs {input_vars}')
        model = instantiate(args.pred_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        risk_func = getattr(importlib.import_module(f'sklearn.metrics'), args.exp.rfi.risk)
        risk = risk_func(y_test, y_pred)
        rfi_results = {'test_loss': risk}

        # Conditional goodness-of-fit
        context_vars = [var for var in dag.var_names if var != target_var]
        gof_results = {}

        if len(context_vars) > 0:
            logger.info(f'Fitting CDE for target = {target_var} and inputs {input_vars}')
            estimator = instantiate(args.estimator, context_size=len(context_vars))
            if args.estimator.fit_params is None:
                getattr(estimator, args.estimator.fit_method)(train_inputs=y_train, train_context=X_train)
            else:
                getattr(estimator, args.estimator.fit_method)(train_inputs=y_train, train_context=X_train,
                                                              **args.estimator.fit_params)
            gof_results = {
                'gof/log_lik': estimator.log_prob(inputs=test_df.loc[:, target_var].values,
                                                  context=test_df.loc[:, context_vars].values).mean(),
                'gof/context_size': len(context_vars)
            }
            mlflow.log_metrics(gof_results, step=var_ind)

        # Relative feature importance
        sampler = instantiate(args.estimator.sampler, X_train=X_train, X_val=X_test, fit_method=args.estimator.fit_method,
                              fit_params=args.estimator.fit_params)

        # 1. G = MB(target_var), FoI = input_vars / MB(target_var)
        G_vars = list(dag.get_markov_blanket(target_var))
        fsoi_vars = [var for var in input_vars if var not in list(dag.get_markov_blanket(target_var))]
        G = search_nonsorted(input_vars, G_vars)
        fsoi = search_nonsorted(input_vars, fsoi_vars)
        if len(fsoi) > 0:
            test_log_probs = sampler.train(fsoi, G)
            rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=risk_func,
                                                fs_names=input_vars)
            mb_explanation = rfi_explainer.rfi(X_test, y_test, G, nr_runs=args.exp.rfi.nr_runs)
            rfi_results['rfi/mb_cond_size'] = len(G_vars)
            rfi_results['rfi/mb_mean_rfi'] = mb_explanation.rfi_means().mean()
            rfi_results['rfi/mb_mean_log_lik'] = np.mean(test_log_probs) if len(G_vars) > 0 else np.nan

        # 2. G = input_vars / MB(target_var), FoI = MB(target_var)
        fsoi_vars = list(dag.get_markov_blanket(target_var))
        G_vars = [var for var in input_vars if var not in list(dag.get_markov_blanket(target_var))]
        G = search_nonsorted(input_vars, G_vars)
        fsoi = search_nonsorted(input_vars, fsoi_vars)
        if len(fsoi) > 0:
            test_log_probs = sampler.train(fsoi, G)
            rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=risk_func,
                                                fs_names=input_vars)
            non_mb_explanation = rfi_explainer.rfi(X_test, y_test, G, nr_runs=args.exp.rfi.nr_runs)
            rfi_results['rfi/non_mb_cond_size'] = len(G_vars)
            rfi_results['rfi/non_mb_mean_rfi'] = non_mb_explanation.rfi_means().mean()
            rfi_results['rfi/non_mb_mean_log_lik'] = np.mean(test_log_probs) if len(G_vars) > 0 else np.nan

        mlflow.log_metrics(rfi_results, step=var_ind)

        results = {**rfi_results, **gof_results}
        metrics = {k: metrics.get(k, []) + [results.get(k, np.nan)] for k in set(list(metrics.keys()) + list(results.keys()))}

    # Logging mean statistics
    mlflow.log_metrics({k: np.nanmean(v) for (k, v) in metrics.items()}, step=len(dag.var_names))
    mlflow.end_run()


if __name__ == "__main__":
    main()
