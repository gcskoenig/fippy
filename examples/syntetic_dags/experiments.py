import mlflow
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

from rfi.backend.causality import DirectedAcyclicGraph
from rfi.backend.goodness_of_fit import *
from rfi.backend.utils import flatten_dict
from rfi.samplers.cnflow import CNFSampler
from rfi.samplers.gaussian import GaussianSampler
from rfi.utils import search_nonsorted, calculate_hash
import rfi.explainers.explainer as explainer


logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='config.yaml')
def main(args: DictConfig):

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info(OmegaConf.to_yaml(args, resolve=True))

    # Data generator
    dag = DirectedAcyclicGraph.random_dag(**args.data_generator.dag)
    if 'interpolation_switch' in args.data_generator.sem:
        args.data_generator.sem.interpolation_switch = args.data.n_train + args.data.n_test
    sem = instantiate(args.data_generator.sem, dag=dag)

    train_df = pd.DataFrame(sem.sample(size=args.data.n_train, seed=args.data.train_seed).numpy(), columns=dag.var_names)
    test_df = pd.DataFrame(sem.sample(size=args.data.n_test, seed=args.data.test_seed).numpy(), columns=dag.var_names)

    # Experiment tracking
    mlflow.set_tracking_uri(args.exp.mlflow_uri)
    mlflow.set_experiment(args.data_generator.sem_type)

    # Checking if run exist
    if args.exp.check_exisisting_hash:
        args.hash = calculate_hash(args)

        existing_runs = mlflow.search_runs(filter_string=f"params.hash = '{args.hash}'",
                                           run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY,
                                           experiment_ids=mlflow.get_experiment_by_name(args.data_generator.sem_type).experiment_id)
        if len(existing_runs) > 0:
            logger.info('Skipping existing run.')
            return
        else:
            logger.info('No runs found - perfoming one.')

    mlflow.start_run()
    mlflow.log_params(flatten_dict(args))

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
        if args.exp.conditioning_mode == 'true_parents':
            context_vars = sem.model[target_var]['parents']
        elif args.exp.conditioning_mode == 'all':
            context_vars = [var for var in dag.var_names if var != target_var]
        elif args.exp.conditioning_mode == 'true_markov_blanket':
            context_vars = list(sem.get_markov_blanket(target_var))
        else:
            raise NotImplementedError()

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
                'gof/kld': conditional_kl_divergence(estimator, sem, target_var, context_vars, args.exp, test_df),
                'gof/hd': conditional_hellinger_distance(estimator, sem, target_var, context_vars, args.exp, test_df),
                'gof/jsd': conditional_js_divergence(estimator, sem, target_var, context_vars, args.exp, test_df),
                'gof/log_lik': estimator.log_prob(inputs=test_df.loc[:, target_var].values,
                                                  context=test_df.loc[:, context_vars].values).mean(),
                'gof/context_size': len(context_vars)
            }
            mlflow.log_metrics(gof_results, step=var_ind)

        # Relative feature importance
        sampler = instantiate(args.estimator.sampler, X_train=X_train, X_val=X_test)

        # 1. G = MB(target_var), FoI = input_vars / MB(target_var)
        G_vars = list(sem.get_markov_blanket(target_var))
        fsoi_vars = [var for var in input_vars if var not in list(sem.get_markov_blanket(target_var))]
        G = search_nonsorted(input_vars, G_vars)
        fsoi = search_nonsorted(input_vars, fsoi_vars)
        if len(fsoi) > 0:
            test_log_probs = sampler.train(fsoi, G)
            rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=risk_func, fs_names=input_vars)
            mb_explanation = rfi_explainer.rfi(X_test, y_test, G, nr_runs=args.exp.rfi.nr_runs)
            rfi_results['rfi/mb_cond_size'] = len(G_vars)
            rfi_results['rfi/mb_mean_rfi'] = mb_explanation.rfi_means().mean()
            rfi_results['rfi/mb_mean_log_lik'] = np.mean(test_log_probs) if len(G_vars) > 0 else np.nan

        # 2. G = input_vars / MB(target_var), FoI = MB(target_var)
        fsoi_vars = list(sem.get_markov_blanket(target_var))
        G_vars = [var for var in input_vars if var not in list(sem.get_markov_blanket(target_var))]
        G = search_nonsorted(input_vars, G_vars)
        fsoi = search_nonsorted(input_vars, fsoi_vars)
        if len(fsoi) > 0:
            test_log_probs = sampler.train(fsoi, G)
            rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=risk_func, fs_names=input_vars)
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
