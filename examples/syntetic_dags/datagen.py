import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy.integrate import quad_vec
import logging

import torch

from rfi.backend.causality import DirectedAcyclicGraph, PostNonLinearLaplaceSEM, PostNonLinearMultiplicativeHalfNormalSEM, \
    LinearGaussianNoiseSEM, RandomGPGaussianNoiseSEM
from rfi.backend.goodness_of_fit import *
from rfi.examples import SyntheticExample


gauss_anm = SyntheticExample(sem=RandomGPGaussianNoiseSEM(dag=DirectedAcyclicGraph.random_dag(6, 0.5, seed=42), seed=42,
                                                          ))

gauss_anm.sem.dag.plot_dag()
plt.show(block=False)


sample = gauss_anm.sem.sample(300, seed=300)
g = sns.pairplot(pd.DataFrame(sample.numpy(), columns=gauss_anm.sem.dag.var_names), plot_kws={'alpha': 0.5})
g.fig.suptitle(gauss_anm.sem.__class__.__name__)
plt.show(block=False)

var_ind = 2
var = gauss_anm.var_names[var_ind]

value = torch.linspace(sample[:, var_ind].min() - 1.0, sample[:, var_ind].max() + 1.0, 200)

global_context = {node: sample[:, node_ind] for (node_ind, node) in enumerate(gauss_anm.var_names) if node != var}
mc_prob = gauss_anm.sem.mb_conditional_log_prob(var, global_context=global_context, method='mc', quad_epsabs=1e-5, mc_size=2000)
# real_prob = linear_gauss.sem.conditional_distribution(var, global_context=global_context)[0].log_prob(value).exp()
fig, ax = plt.subplots()

mc_prob_vals = mc_prob(value.unsqueeze(-1).repeat(1, len(sample))).exp().T

for i, mc_prob_val in enumerate(mc_prob_vals):
    # label = {node: '{0:.2f}'.format(global_context[node][i].item()) for node in global_context}
    plt.plot(value, mc_prob_val, alpha=0.1, c='blue')
# real_plot = plt.plot(value, real_prob, label='Real density')[0]
plt.title(f'Density of {var} / {[node for node in gauss_anm.var_names if node != var]}')
plt.legend()
plt.show()

integrand = lambda val: mc_prob(torch.tensor([[val]]).repeat(1, len(sample))).exp().numpy()
print(quad_vec(integrand, *gauss_anm.sem.support_bounds, epsabs=1e-4))

    # Slider
    # sliders = {}
    # for i, node in enumerate(gauss_anm.var_names):
    #     if node != var:
    #         sliders[node] = Slider(plt.axes([0.25, .22 - i * 0.04, 0.50, 0.02]), node, sample[:, i].min(), sample[:, i].max(),
    #                                valinit=global_context[node])
    #
    # mc_size_slider = Slider(plt.axes([0.25, .28, 0.50, 0.02]), 'MC size', 10, 1000, valinit=mc_size)
    #
    # def update(val):
    #     # amp is the current value of the slider
    #     for node, slider in sliders.items():
    #         global_context[node] = torch.tensor([slider.val])
    #
    #     mc_prob = gauss_anm.sem.mb_conditional_log_prob(var, global_context=global_context, mc_size=int(mc_size_slider.val))
    #     mc_plot.set_ydata(mc_prob(value).exp())
    #
    #     # real_prob = linear_gauss.sem.conditional_distribution(var, global_context=global_context)[0]
    #     # real_plot.set_ydata(real_prob.log_prob(value).exp())
    #
    #
    #     # update curve
    #     # redraw canvas while idle
    #     fig.canvas.draw_idle()
    #
    #     # def f1(value):
    #     #     value = torch.tensor([value])
    #     #     return mc_prob(value).exp().numpy()
    #     #
    #     # def f2(value):
    #     #     value = torch.tensor([value])
    #     #     return real_prob.log_prob(value).exp().numpy()
    #     #
    #     # print(np.abs(quad(f1, -np.inf, np.inf, epsabs=1e-7)[0] - quad(f2, -np.inf, np.inf, epsabs=1e-7)[0]))
    #
    #
    # # call update function on slider value change
    # for slider in sliders.values():
    #     slider.on_changed(update)
    # mc_size_slider.on_changed(update)
    #
    # fig.subplots_adjust(bottom=0.4)




# print(gauss_anm.sem.joint_log_prob(sample).exp().mean())



# target_var = 'x3'
# context_vars = linear_gauss.sem.model[target_var]['parents']
# context_vars = ['x4', 'x5']
# X_train, y_train, X_test, y_test = linear_gauss.get_train_test_data(context_vars=context_vars, target_var=target_var, n_train=10 ** 3, n_test=10 ** 3,
#                                                                    mc_seed=300)


# estimator = GaussianConditionalEstimator()
# estimator.fit(train_inputs=y_train, train_context=X_train)
# # estimator = ConditionalNormalisingFlowEstimator(len(context_vars))
# # estimator.fit_by_cv(train_inputs=y_train, train_context=X_train)
# sns.kdeplot(y_test, label='data')
# sns.kdeplot(estimator.sample(X_test).flatten(), label='estimator')
# plt.show()

# model_cond_dist = estimator.conditional_distribution(X_test)
# data_cond_dist = linear_gauss.sem.parents_conditional_distribution(target_var)
# data_cond_dist = linear_gauss.sem.parents_conditional_distribution(target_var, global_context={node: torch.tensor()})
# data_cond_dist = linear_gauss.sem.parents_conditional_distribution(target_var, global_context={par: torch.tensor(X_test[:, par_ind]) for (par_ind, par) in enumerate(context_vars)})

# print(conditional_js_divergence(model_distributions=model_cond_dist, data_distributions=data_cond_dist))
# print(conditional_kl_divergence(model_distributions=model_cond_dist, data_distributions=data_cond_dist))
# print(conditional_hellinger_distance(model_distributions=model_cond_dist, data_distributions=data_cond_dist))
# data_pdf = lambda cont: gauss_anm.sem.parents_conditional_distribution(node=target_var, global_context={par: torch.tensor(cont[par_ind])
#                                                                              for (par_ind, par) in enumerate(context_vars)})
#
# print(conditional_hellinger_distance(model_pdf, data_pdf, X_test))

# for i, cont in (enumerate(X_test)):

# Creating random SEM

# x2 = sem.sample(500, mc_seed=43)
#
# print(sem.parents_conditional_log_prob('x1', value=torch.tensor([0.75, 0.95]), global_context={'x0': torch.tensor([0.1, 0.2]),
#                                                                            'x1': torch.tensor([0.1, 0.3])}))
#
# data = pd.DataFrame(np.concatenate([x1.numpy(), x2.numpy()]), columns=dag.var_names)
# data['sample'] = 500 * ['train'] + 500 * ['test']
#
# # Plotting sampled data
# g = sns.pairplot(data, hue='sample')
# # g.set(ylim=(data[dag.var_names].min().min(), data[dag.var_names].max().max()))
# g.fig.suptitle(sem.__class__.__name__)
# plt.show()
