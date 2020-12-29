import mlflow
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rfi.examples import SyntheticExample
from rfi.backend.causality import DirectedAcyclicGraph
from rfi.backend.goodness_of_fit import *
from rfi.backend.gaussian.gaussian_estimator import GaussianConditionalEstimator


logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='config.yaml')
def main(args: DictConfig):

    logger.info(OmegaConf.to_yaml(args, resolve=True))

    # Data generator
    dag = DirectedAcyclicGraph.random_dag(**args.data_generator.dag)
    sem = instantiate(args.data_generator.sem, dag=dag)
    data_generator = SyntheticExample(sem=sem)
    # sem.dag.plot_dag()
    # plt.show()

    train_df = pd.DataFrame(sem.sample(size=args.data.n_train, seed=args.data.train_seed).numpy(), columns=dag.var_names)
    test_df = pd.DataFrame(sem.sample(size=args.data.n_test, seed=args.data.test_seed).numpy(), columns=dag.var_names)

    for target_var in dag.var_names:

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
        print(conditional_kl_divergence(estimator, sem, target_var, context_vars, args.exp.conditioning_mode, test_df))
        print(conditional_hellinger_distance(estimator, sem, target_var, context_vars, args.exp.conditioning_mode, test_df))
        print(conditional_js_divergence(estimator, sem, target_var, context_vars, args.exp.conditioning_mode, test_df))



if __name__ == "__main__":
    main()
