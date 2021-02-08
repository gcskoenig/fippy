import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import quad_vec
import logging

import torch

from rfi.backend.causality import DirectedAcyclicGraph, PostNonLinearLaplaceSEM, PostNonLinearMultiplicativeHalfNormalSEM, \
    LinearGaussianNoiseSEM, RandomGPGaussianNoiseSEM
from rfi.examples import SyntheticExample


logging.basicConfig(level=logging.INFO)
EPSABS = 0.005
SEED = 4242
ASSERT_DECIMAL = 1
ASSERT_DECIMAL_DATA = 4
SAMPLE_SIZE = 50
DAG_N = 5
DAG_P = 0.5
MC_SIZE = 1000
DAG = DirectedAcyclicGraph.random_dag(DAG_N, DAG_P, seed=SEED)
INTERPOLATION_SWITCH = 2 * SAMPLE_SIZE


class TestSEMs:

    @staticmethod
    def _plot_and_test_dag_and_data(synt_ex: SyntheticExample):
        # Checking reproducability
        np.testing.assert_array_equal(DirectedAcyclicGraph.random_dag(DAG_N, DAG_P, seed=SEED).adjacency_matrix,
                                      DirectedAcyclicGraph.random_dag(DAG_N, DAG_P, seed=SEED).adjacency_matrix)
        np.testing.assert_array_almost_equal(synt_ex.sem.sample(SAMPLE_SIZE, seed=SEED),
                                             synt_ex.sem.sample(SAMPLE_SIZE, seed=SEED), decimal=ASSERT_DECIMAL_DATA)

        synt_ex.sem.dag.plot_dag()
        plt.show(block=False)

        sample = synt_ex.sem.sample(SAMPLE_SIZE, seed=SEED)
        g = sns.pairplot(pd.DataFrame(sample.numpy(), columns=synt_ex.sem.dag.var_names), plot_kws={'alpha': 0.5})
        g.fig.suptitle(synt_ex.sem.__class__.__name__)
        plt.show(block=False)

    @staticmethod
    def _plot_and_test_mb_cond_dist(synt_ex: SyntheticExample, var_ind, method):
        logging.info(f'Calculating MB conditional distribution, using method {method}')
        var = synt_ex.var_names[var_ind]
        sample = synt_ex.sem.sample(SAMPLE_SIZE, seed=SEED)
        global_context = {node: sample[:, node_ind] for (node_ind, node) in enumerate(synt_ex.var_names) if node != var}
        mc_prob = synt_ex.sem.mb_conditional_log_prob(var, global_context=global_context, method=method, quad_epsabs=EPSABS,
                                                      mc_size=MC_SIZE)

        # Checking integration to zero
        integrand = lambda val: mc_prob(torch.tensor(val).repeat(1, len(sample))).exp().numpy()
        int_res = quad_vec(integrand, *synt_ex.sem.support_bounds, epsabs=EPSABS)[0]
        logging.info(f'Mean integral of cond density: {int_res.mean()}')
        if method == 'quad':
            np.testing.assert_array_almost_equal(int_res.flatten(), np.ones(len(sample)), ASSERT_DECIMAL)

        # Plotting conditional distributions
        values = torch.linspace(sample[:, var_ind].min() - 1.0, sample[:, var_ind].max() + 1.0, 200)
        mc_prob_vals = mc_prob(values.unsqueeze(-1).repeat(1, len(sample))).exp().T

        fig, ax = plt.subplots()
        for i, mc_prob_val in enumerate(mc_prob_vals):
            ax.plot(values, mc_prob_val, alpha=0.1, c='blue')
        plt.title(f'Density of {var} / {[node for node in synt_ex.var_names if node != var]}')
        plt.show(block=False)

    @staticmethod
    def _plot_sliders(synt_ex: SyntheticExample, var_ind, cont_ind, method):
        var = synt_ex.var_names[var_ind]
        sample = synt_ex.sem.sample(SAMPLE_SIZE, seed=SEED)
        global_context = {node: sample[cont_ind:cont_ind + 1, node_ind]
                          for (node_ind, node) in enumerate(synt_ex.var_names) if node != var}

        # Calculating MB cond distribution for one context
        mc_prob = synt_ex.sem.mb_conditional_log_prob(var, global_context=global_context, method=method, mc_size=MC_SIZE,
                                                      quad_epsabs=EPSABS)

        # Plotting initial plot
        fig, ax = plt.subplots()
        values = torch.linspace(sample[:, var_ind].min() - 1.0, sample[:, var_ind].max() + 1.0, 200)
        mc_prob_vals = mc_prob(values.unsqueeze(-1)).exp()

        mc_plot, = plt.plot(values, mc_prob_vals, c='blue')
        plt.title(f'Density of {var} / {[node for node in synt_ex.var_names if node != var]}')

        # Slider
        sliders = {}
        for i, node in enumerate(synt_ex.var_names):
            if node != var:
                sliders[node] = Slider(plt.axes([0.25, .22 - i * 0.04, 0.50, 0.02]), node, sample[:, i].min(), sample[:, i].max(),
                                       valinit=global_context[node])

        mc_size_slider = Slider(plt.axes([0.25, .28, 0.50, 0.02]), 'MC size', 10, 5000, valinit=MC_SIZE)

        def update(val):
            # amp is the current value of the slider
            for node, slider in sliders.items():
                global_context[node] = torch.tensor([slider.val])

            mc_prob = synt_ex.sem.mb_conditional_log_prob(var, global_context=global_context, method=method, quad_epsabs=EPSABS,
                                                          mc_size=int(mc_size_slider.val))
            mc_plot.set_ydata(mc_prob(values).exp())

            # update curve
            fig.canvas.draw_idle()

        # call update function on slider value change
        for slider in sliders.values():
            slider.on_changed(update)
        mc_size_slider.on_changed(update)

        fig.subplots_adjust(bottom=0.4)
        plt.show()

    def test_linear_gaussian_noise_sem(self):
        gauss_anm = SyntheticExample(sem=LinearGaussianNoiseSEM(dag=DAG, seed=SEED))

        self._plot_and_test_dag_and_data(gauss_anm)
        self._plot_and_test_mb_cond_dist(gauss_anm, 2, 'mc')
        self._plot_and_test_mb_cond_dist(gauss_anm, 2, 'quad')
        self._plot_sliders(gauss_anm, 2, SAMPLE_SIZE - 1, 'quad')

    def test_post_non_linear_laplace_sem(self):
        laplace_sem = SyntheticExample(sem=PostNonLinearLaplaceSEM(dag=DAG, seed=SEED, interpolation_switch=INTERPOLATION_SWITCH))

        self._plot_and_test_dag_and_data(laplace_sem)
        self._plot_and_test_mb_cond_dist(laplace_sem, 2, 'mc')
        self._plot_and_test_mb_cond_dist(laplace_sem, 2, 'quad')
        self._plot_sliders(laplace_sem, 2, SAMPLE_SIZE - 1, 'quad')

    def test_randomgp_gaussian_noise_sem(self):
        gp_anm = SyntheticExample(sem=RandomGPGaussianNoiseSEM(dag=DAG, seed=SEED, interpolation_switch=INTERPOLATION_SWITCH))

        self._plot_and_test_dag_and_data(gp_anm)
        self._plot_and_test_mb_cond_dist(gp_anm, 2, 'mc')
        self._plot_and_test_mb_cond_dist(gp_anm, 2, 'quad')
        self._plot_sliders(gp_anm, 2, SAMPLE_SIZE - 1, 'quad')

    def test_post_nonlinear_multiplicative_half_normal_sem(self):
        post_half_norm_sem = SyntheticExample(sem=PostNonLinearMultiplicativeHalfNormalSEM(dag=DAG, seed=SEED))

        self._plot_and_test_dag_and_data(post_half_norm_sem)
        self._plot_and_test_mb_cond_dist(post_half_norm_sem, 2, 'mc')
        self._plot_and_test_mb_cond_dist(post_half_norm_sem, 2, 'quad')
        self._plot_sliders(post_half_norm_sem, 2, SAMPLE_SIZE - 1, 'quad')
