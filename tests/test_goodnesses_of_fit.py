import logging
import pandas as pd
import numpy as np
from typing import Union

from rfi.backend.goodness_of_fit import conditional_js_divergence, conditional_kl_divergence, conditional_hellinger_distance
from rfi.backend.causality import DirectedAcyclicGraph, PostNonLinearLaplaceSEM, PostNonLinearMultiplicativeHalfNormalSEM, \
    LinearGaussianNoiseSEM, RandomGPGaussianNoiseSEM, StructuralEquationModel
from rfi.backend.gaussian import GaussianConditionalEstimator
from rfi.backend.cnf import NormalisingFlowEstimator
from rfi.backend.mdn import MixtureDensityNetworkEstimator

logging.basicConfig(level=logging.INFO)

SEED = 4242
ASSERT_DECIMAL = 1
SAMPLE_SIZE = 200
DAG_N = 5
DAG_P = 0.5
DAG = DirectedAcyclicGraph.random_dag(DAG_N, DAG_P, seed=SEED)
TARGET_VAR = 'x2'
CONTEXT_VARS = DAG.get_markov_blanket(TARGET_VAR)
ESTIMATORS = [
    GaussianConditionalEstimator(),
    NormalisingFlowEstimator(context_size=len(CONTEXT_VARS)),
    # MixtureDensityNetworkEstimator(context_size=len(CONTEXT_VARS))
]
GOF_ARGS = {
    'metrics': {'epsabs': 0.05},
    'mb_dist': {
        'method': 'mc',
        'mc_size': 10000
    }
}


class TestConditionalGoF:

    @staticmethod
    def get_train_test_dfs(sem: StructuralEquationModel):
        # Generating train /test data
        train_df = pd.DataFrame(sem.sample(SAMPLE_SIZE, seed=SEED).numpy(), columns=sem.dag.var_names)
        test_df2 = pd.DataFrame(sem.sample(SAMPLE_SIZE, seed=2 * SEED).numpy(), columns=sem.dag.var_names)
        test_df1 = pd.DataFrame(sem.sample(SAMPLE_SIZE, seed=4 * SEED).numpy(), columns=sem.dag.var_names)
        return train_df, test_df1, test_df2

    @staticmethod
    def assert_metrics(test_df1: pd.DataFrame, test_df2: pd.DataFrame, sem: StructuralEquationModel,
                       estimator: Union[GaussianConditionalEstimator, NormalisingFlowEstimator], metric_func: callable,
                       assert_decimals: int):
        # Calculating metrics twice and checking equality (mb_cond_distributions could be different)
        np.testing.assert_almost_equal(
            metric_func(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket', test_df1),
            metric_func(estimator, sem, TARGET_VAR, CONTEXT_VARS, GOF_ARGS, 'true_markov_blanket', test_df2),
            assert_decimals
        )

    def test_linear_gaussian_noise_sem_conditional_kl_divergence(self):
        sem = LinearGaussianNoiseSEM(DAG, seed=SEED)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)
        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_kl_divergence, ASSERT_DECIMAL - 1)

    def test_linear_gaussian_noise_sem_conditional_js_divergence(self):
        sem = LinearGaussianNoiseSEM(DAG, seed=SEED)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)
        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_js_divergence, ASSERT_DECIMAL)

    def test_linear_gaussian_noise_sem_conditional_hellinger_distance(self):
        sem = LinearGaussianNoiseSEM(DAG, seed=SEED)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)
        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_hellinger_distance, ASSERT_DECIMAL)

    def test_randomgp_gaussian_noise_sem_conditional_kl_divergence(self):
        sem = RandomGPGaussianNoiseSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_kl_divergence, ASSERT_DECIMAL - 1)

    def test_randomgp_gaussian_noise_sem_conditional_js_divergence(self):
        sem = RandomGPGaussianNoiseSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_js_divergence, ASSERT_DECIMAL)

    def test_randomgp_gaussian_noise_sem_conditional_hellinger_distance(self):
        sem = RandomGPGaussianNoiseSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_hellinger_distance, ASSERT_DECIMAL)

    def test_post_nonlinear_laplace_sem_conditional_kl_divergence(self):
        sem = PostNonLinearLaplaceSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_kl_divergence, ASSERT_DECIMAL - 1)

    def test_post_nonlinear_laplace_sem_conditional_js_divergence(self):
        sem = PostNonLinearLaplaceSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_js_divergence, ASSERT_DECIMAL)

    def test_post_nonlinear_laplace_sem_conditional_hellinger_distance(self):
        sem = PostNonLinearLaplaceSEM(DAG, seed=SEED, interpolation_switch=2 * SAMPLE_SIZE)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_hellinger_distance, ASSERT_DECIMAL)

    def test_post_nonlinear_multiplicative_half_normal_sem_conditional_kl_divergence(self):
        sem = PostNonLinearMultiplicativeHalfNormalSEM(DAG, seed=SEED)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_kl_divergence, ASSERT_DECIMAL - 1)

    def test_post_nonlinear_multiplicative_half_normal_sem_conditional_js_divergence(self):
        sem = PostNonLinearMultiplicativeHalfNormalSEM(DAG, seed=SEED)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_js_divergence, ASSERT_DECIMAL)

    def test_post_nonlinear_multiplicative_half_normal_sem_conditional_hellinger_distance(self):
        sem = PostNonLinearMultiplicativeHalfNormalSEM(DAG, seed=SEED)
        train_df, test_df1, test_df2 = self.get_train_test_dfs(sem)

        for estimator in ESTIMATORS:
            estimator.fit(train_inputs=train_df[[TARGET_VAR]].values, train_context=train_df[CONTEXT_VARS].values)
            self.assert_metrics(test_df1, test_df2, sem, estimator, conditional_hellinger_distance, ASSERT_DECIMAL)
