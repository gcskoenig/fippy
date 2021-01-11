import mlflow
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

from rfi.backend.causality import DirectedAcyclicGraph
from rfi.backend.goodness_of_fit import *
from rfi.backend.utils import flatten_dict
from rfi.samplers.cnflow import CNFSampler
from rfi.samplers.gaussian import GaussianSampler
import rfi.explainers.explainer as explainer


logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='config.yaml')
def main(args: DictConfig):

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
    mlflow.start_run()
    mlflow.log_params(flatten_dict(args))

    # Saving artifacts
    train_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/train.csv'), index=False)
    test_df.to_csv(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/test.csv'), index=False)
    sem.dag.plot_dag()
    plt.savefig(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/dag.png'))
    df = pd.concat([train_df, test_df], keys=['train', 'test']).reset_index().drop(columns=['level_1'])
    g = sns.pairplot(df, plot_kws={'alpha': 0.25}, hue='level_0')
    g.fig.suptitle(sem.__class__.__name__)
    plt.savefig(hydra.utils.to_absolute_path(f'{mlflow.get_artifact_uri()}/data.png'))

    metrics = {}

    for var_ind, target_var in enumerate(dag.var_names):

        if args.exp.conditioning_mode == 'true_parents':
            context_vars = sem.model[target_var]['parents']
        elif args.exp.conditioning_mode == 'all':
            context_vars = [var for var in dag.var_names if var != target_var]
        elif args.exp.conditioning_mode == 'true_markov_blanket':
            context_vars = list(sem.get_markov_blanket(target_var))
        else:
            raise NotImplementedError()

        if len(context_vars) == 0:
            continue  # Not evaluating unconditional estimators

        # Estimator
        estimator = instantiate(args.estimator, context_size=len(context_vars))
        if args.estimator.fit_params is None:
            getattr(estimator, args.estimator.fit_method)(train_inputs=train_df.loc[:, target_var].values,
                                                          train_context=train_df.loc[:, context_vars].values)
        else:
            getattr(estimator, args.estimator.fit_method)(train_inputs=train_df.loc[:, target_var].values,
                                                          train_context=train_df.loc[:, context_vars].values,
                                                          **args.estimator.fit_params)


        # Computing metrics
        results = {
            'gof/kld': conditional_kl_divergence(estimator, sem, target_var, context_vars, args.exp, test_df),
            'gof/hd': conditional_hellinger_distance(estimator, sem, target_var, context_vars, args.exp, test_df),
            'gof/jsd': conditional_js_divergence(estimator, sem, target_var, context_vars, args.exp, test_df),
            'gof/log_lik': estimator.log_prob(inputs=test_df.loc[:, target_var].values,
                                              context=test_df.loc[:, context_vars].values).mean()
        }
        mlflow.log_metrics(results, step=var_ind)
        metrics = {k: metrics.get(k, []) + [results[k]] for k in set(list(metrics.keys()) + list(results.keys()))}

    # Logging mean statistics
    mlflow.log_metrics({k: np.mean(v) for (k, v) in metrics.items()}, step=len(dag.var_names))

    mlflow.end_run()


if __name__ == "__main__":
    main()
