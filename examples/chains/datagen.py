import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt

from rfi.backend.causality.dags import DirectedAcyclicGraph
from rfi.backend.causality.sem import LinearGaussianNoiseSEM


'''
EXAMPLE 1: CONFOUNDING

Order: C, X_1, X_2, X_3, Y
'''
# name = 'confounding2'
# pyro.set_rng_seed(300)
#
# sigma_low = 0.3
# sigma_medium = .5
# sigma_high = 1
#
#
# def model_confounding():
#     c = pyro.sample('c', pyro.distributions.Normal(0, sigma_high))
#     x1 = pyro.sample('x1', pyro.distributions.Normal(0, sigma_high))
#     x2 = pyro.sample('x2', pyro.distributions.Normal(c, sigma_high))
#     x3 = pyro.sample('x3', pyro.distributions.Normal(c, sigma_medium))
#     y = pyro.sample('y', pyro.distributions.Normal(x1 + x2 + c, sigma_medium))
#     unit = torch.tensor([c, x1, x2, x3, y])
#     return unit
#
#
# N = 10 ** 5
#
# dataset = torch.randn(N, 5)
# for n in range(N):
#     unit = model_confounding()
#     # print(unit)
#     dataset[n] = unit
#
# dataset = dataset.numpy()
# np.savetxt('data/{}.csv'.format(name), dataset)

'''
EXAMPLE 2: CHAIN with multiple paths

Order: X_1, X_2, X_3, X_4, Y
'''
# def model_chain():
#     x1 = pyro.sample('x1', pyro.distributions.Normal(0, sigma_high))
#     x2 = pyro.sample('x2', pyro.distributions.Normal(x1, sigma_high))
#     x3 = pyro.sample('x3', pyro.distributions.Normal(x2, sigma_low))
#     x4 = pyro.sample('x4', pyro.distributions.Normal(x1, sigma_high))
#     y = pyro.sample('y', pyro.distributions.Normal(x3 + x4, sigma_medium))
#     unit = torch.tensor([x1, x2, x3, x4, y])
#     return unit

sigma_low = 0.3
sigma_medium = .5
sigma_high = 1

dag = DirectedAcyclicGraph(np.array([[0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]),
                           var_names=['x1', 'x2', 'x3', 'x4', 'y'])

chain2 = LinearGaussianNoiseSEM(
    dag=dag,
    coeff_dict={'x2': {'x1': 1.0}, 'x3': {'x2': 1.0}, 'x4': {'x1': 1.0}, 'y': {'x3': 1.0, 'x4': 1.0}},
    noise_std_dict={'x1': sigma_high, 'x2': sigma_high, 'x3': sigma_low, 'x4': sigma_high, 'y': sigma_medium}
)

dataset = chain2.sample(10 ** 5, seed=42).numpy()

name = 'chain2'
np.savetxt(f'data/{name}.csv', dataset)

dag.plot_dag()
plt.savefig(f'data/{name}.png')
