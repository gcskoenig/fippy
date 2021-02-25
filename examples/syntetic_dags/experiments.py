import mlflow
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import inspect
import logging

from rfi.backend.causality import DirectedAcyclicGraph, LinearGaussianNoiseSEM
from rfi.backend.goodness_of_fit import conditional_hellinger_distance, conditional_kl_divergence, conditional_js_divergence
from rfi.backend.utils import flatten_dict
from rfi.utils import search_nonsorted, check_existing_hash
import rfi.explainers.explainer as explainer


logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='config.yaml')
def main(args: DictConfig):

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)

    # Adding default estimator params
    default_names, _, _, default_values, _, _, _ = \
        inspect.getfullargspec(instantiate(args.estimator, context_size=0).__class__.__init__)
    if default_values is not None:
        args.estimator['defaults'] = {
            n: str(v) for (n, v) in zip(default_names[len(default_names) - len(default_values):], default_values)
        }
        args.estimator['defaults'].pop('cat_context')
    logger.info(OmegaConf.to_yaml(args, resolve=True))

    # Data generator init
    dag = DirectedAcyclicGraph.random_dag(**args.data_generator.dag)
    # if 'interpolation_switch' in args.data_generator.sem:
    #     args.data_generator.sem.interpolation_switch = args.data.n_train + args.data.n_test
    sem = instantiate(args.data_generator.sem, dag=dag)

    # Experiment tracking
    mlflow.set_tracking_uri(args.exp.mlflow_uri)
    mlflow.set_experiment(args.data_generator.sem_type)

    # Checking if run exist
    if check_existing_hash(args, args.data_generator.sem_type):
        logger.info('Skipping existing run.')
        return
    else:
        logger.info('No runs found - perfoming one.')

    mlflow.start_run()
    mlflow.log_params(flatten_dict(args))

    # Generating Train-test dataframes
    train_df = pd.DataFrame(sem.sample(size=args.data.n_train, seed=args.data.train_seed).numpy(), columns=dag.var_names)
    test_df = pd.DataFrame(sem.sample(size=args.data.n_test, seed=args.data.test_seed).numpy(), columns=dag.var_names)

    # Saving artifacts
    train_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/train.csv'), index=False)
    test_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/test.csv'), index=False)
    sem.dag.plot_dag()
    plt.savefig(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/dag.png'))
    if len(dag.var_names) <= 20:
        df = pd.concat([train_df, test_df], keys=['train', 'test']).reset_index().drop(columns=['level_1'])
        g = sns.pairplot(df, plot_kws={'alpha': 0.25}, hue='level_0')
        g.fig.suptitle(sem.__class__.__name__)
        plt.savefig(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/data.png'))

    metrics = {}

    for var_ind, target_var in enumerate(dag.var_names):

        var_results = {}

        # Considering all the variables for input
        input_vars = [var for var in dag.var_names if var != target_var]
        y_train, X_train = train_df.loc[:, target_var].values, train_df.loc[:, input_vars].values
        y_test, X_test = test_df.loc[:, target_var].values, test_df.loc[:, input_vars].values

        # Initialising risks
        risks = {}
        for risk in args.predictors.risks:
            risks[risk] = getattr(importlib.import_module('sklearn.metrics'), risk)

        # Fitting predictive model
        models = {}
        for pred_model in args.predictors.pred_models:
            logger.info(f'Fitting {pred_model._target_} for target = {target_var} and inputs {input_vars}')
            model = instantiate(pred_model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            models[pred_model._target_] = model
            for risk, risk_func in risks.items():
                var_results[f'test_{risk}_{pred_model._target_}'] = risk_func(y_test, y_pred)

        sampler = instantiate(args.estimator.sampler, X_train=X_train,
                              fit_method=args.estimator.fit_method, fit_params=args.estimator.fit_params)

        # =================== Relative feature importance ===================
        # 1. G = MB(target_var), FoI = input_vars / MB(target_var)
        G_vars_1 = list(sem.get_markov_blanket(target_var))
        fsoi_vars_1 = [var for var in input_vars if var not in list(sem.get_markov_blanket(target_var))]
        prefix_1 = 'mb'

        # 2. G = input_vars / MB(target_var), FoI = MB(target_var)
        fsoi_vars_2 = list(sem.get_markov_blanket(target_var))
        G_vars_2 = [var for var in input_vars if var not in list(sem.get_markov_blanket(target_var))]
        prefix_2 = 'non_mb'

        for (G_vars, fsoi_vars, prefix) in zip([G_vars_1, G_vars_2], [fsoi_vars_1, fsoi_vars_2], [prefix_1, prefix_2]):
            G = search_nonsorted(input_vars, G_vars)
            fsoi = search_nonsorted(input_vars, fsoi_vars)

            rfi_gof_metrics = {}
            for f, f_var in zip(fsoi, fsoi_vars):
                estimator = sampler.train([f], G)

                # GoF diagnostics
                rfi_gof_results = {}
                if estimator is not None:

                    rfi_gof_results[f'rfi/gof/{prefix}_mean_log_lik'] = \
                        estimator.log_prob(inputs=X_test[:, f], context=X_test[:, G]).mean()

                    # Advanced conditional GoF metrics
                    if sem.get_markov_blanket(f_var).issubset(set(G_vars)):
                        cond_mode = 'all'
                    if isinstance(sem, LinearGaussianNoiseSEM):
                        cond_mode = 'arbitrary'

                    if sem.get_markov_blanket(f_var).issubset(set(G_vars)) or isinstance(sem, LinearGaussianNoiseSEM):
                        rfi_gof_results[f'rfi/gof/{prefix}_kld'] = \
                            conditional_kl_divergence(estimator, sem, f_var, G_vars, args.exp, cond_mode, test_df)
                        rfi_gof_results[f'rfi/gof/{prefix}_hd'] = \
                            conditional_hellinger_distance(estimator, sem, f_var, G_vars, args.exp, cond_mode, test_df)
                        rfi_gof_results[f'rfi/gof/{prefix}_jsd'] = \
                            conditional_js_divergence(estimator, sem, f_var, G_vars, args.exp, cond_mode, test_df)

                rfi_gof_metrics = {k: rfi_gof_metrics.get(k, []) + [rfi_gof_results.get(k, np.nan)]
                                   for k in set(list(rfi_gof_metrics.keys()) + list(rfi_gof_results.keys()))}

            # Feature importance
            if len(fsoi) > 0:
                var_results[f'rfi/{prefix}_cond_size'] = len(G_vars)

                for model_name, model in models.items():
                    for risk, risk_func in risks.items():

                        rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=risk_func,
                                                            fs_names=input_vars)
                        mb_explanation = rfi_explainer.rfi(X_test, y_test, G, nr_runs=args.exp.rfi.nr_runs)
                        var_results[f'rfi/{prefix}_mean_rfi_{risk}_{model_name}'] = \
                            np.abs(mb_explanation.fi_vals(return_np=True)).mean()

                var_results = {**var_results,
                               **{k: np.nanmean(v) if len(G_vars) > 0 else np.nan for (k, v) in rfi_gof_metrics.items()}}

        # TODO  =================== Global SAGE ===================

        mlflow.log_metrics(var_results, step=var_ind)

        metrics = {k: metrics.get(k, []) + [var_results.get(k, np.nan)]
                   for k in set(list(metrics.keys()) + list(var_results.keys()))}

    # Logging mean statistics
    mlflow.log_metrics({k: np.nanmean(v) for (k, v) in metrics.items()}, step=len(dag.var_names))
    mlflow.end_run()


if __name__ == "__main__":
    main()
