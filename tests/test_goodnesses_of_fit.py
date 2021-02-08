import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from rfi.backend.goodness_of_fit import conditional_js_divergence, conditional_kl_divergence, conditional_hellinger_distance
from rfi.backend.causality import DirectedAcyclicGraph, PostNonLinearLaplaceSEM, PostNonLinearMultiplicativeHalfNormalSEM, \
    LinearGaussianNoiseSEM, RandomGPGaussianNoiseSEM
from rfi.examples import SyntheticExample
from rfi.backend.gaussian import GaussianConditionalEstimator
from rfi.backend.cnf import NormalisingFlowEstimator

logging.basicConfig(level=logging.INFO)

SEED = 4242
ASSERT_DECIMAL = 2
SAMPLE_SIZE = 50
DAG_N = 5
DAG_P = 0.5
DAG = DirectedAcyclicGraph.random_dag(DAG_N, DAG_P, seed=SEED)
TARGET_VAR = 'x2'
CONTEXT_VARS = DAG.get_markov_blanket(TARGET_VAR)
SEMS = [
    LinearGaussianNoiseSEM(DAG, seed=SEED),
    RandomGPGaussianNoiseSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE),
    PostNonLinearLaplaceSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE),
    PostNonLinearMultiplicativeHalfNormalSEM(DAG, seed=SEED)
]
ESTIMATORS = [GaussianConditionalEstimator(), NormalisingFlowEstimator(context_size=len(CONTEXT_VARS))]
GOF_ARGS = {
    'metrics': {'epsabs': 0.01},
    'mb_dist': {
        'method': 'mc',
        'mc_size': 10000
    }
}


def test_goodnesses_of_fit():

    for sem in tqdm(SEMS):
        # Generating train /test data
        train_df = pd.DataFrame(sem.sample(SAMPLE_SIZE, seed=SEED).numpy(), columns=sem.dag.var_names)
        test_df = pd.DataFrame(sem.sample(SAMPLE_SIZE, seed=2 * SEED).numpy(), columns=sem.dag.var_names)

        for estimator in ESTIMATORS:
            # Fitting estimator
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)

            # Calculating metrics twice and checking equality (mb_cond_distributions could be different)
            np.testing.assert_almost_equal(
                conditional_kl_divergence(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket', test_df),
                conditional_kl_divergence(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket', test_df),
                ASSERT_DECIMAL
            )
            np.testing.assert_almost_equal(
                conditional_js_divergence(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket', test_df),
                conditional_js_divergence(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket', test_df),
                ASSERT_DECIMAL
            )
            np.testing.assert_almost_equal(
                conditional_hellinger_distance(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket',
                                               test_df),
                conditional_hellinger_distance(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket',
                                               test_df),
                ASSERT_DECIMAL
            )
