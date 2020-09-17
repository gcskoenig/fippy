import pyro
import torch
import numpy as np


'''
EXAMPLE 1: CONFOUNDING

Order: C, X_1, X_2, X_3, Y
'''
name='confounding2'
pyro.set_rng_seed(300)


sigma_low = 0.3
sigma_medium = .5
sigma_high = 1

def model_confounding():
        c = pyro.sample('c', pyro.distributions.Normal(0, sigma_high))
        x1 = pyro.sample('x1', pyro.distributions.Normal(0, sigma_high))
        x2 = pyro.sample('x2', pyro.distributions.Normal(c, sigma_high))
        x3 = pyro.sample('x3', pyro.distributions.Normal(c, sigma_medium))
        y = pyro.sample('y', pyro.distributions.Normal(x1 + x2 + c, sigma_medium))
        unit = torch.tensor([c, x1, x2, x3, y])
        return unit

N=10**5

dataset = torch.randn(N, 5)
for n in range(N):
        unit = model_confounding()
        #print(unit)
        dataset[n] = unit

dataset = dataset.numpy()
np.savetxt('data/{}.csv'.format(name), dataset)


'''
EXAMPLE 2: CHAIN with multiple paths

Order: X_1, X_2, X_3, X_4, Y
'''

name='chain2'
pyro.set_rng_seed(300)


sigma_low = 0.3
sigma_medium = .5
sigma_high = 1

def model_chain():
        x1 = pyro.sample('x1', pyro.distributions.Normal(0, sigma_high))
        x2 = pyro.sample('x2', pyro.distributions.Normal(x1, sigma_high))
        x3 = pyro.sample('x3', pyro.distributions.Normal(x2, sigma_low))
        x4 = pyro.sample('x4', pyro.distributions.Normal(x1, sigma_high))
        y = pyro.sample('y', pyro.distributions.Normal(x3+x4, sigma_medium))
        unit = torch.tensor([x1, x2, x3, x4, y])
        return unit

N=10**5

dataset = torch.randn(N, 5)
for n in range(N):
        unit = model_chain()
        #print(unit)
        dataset[n] = unit

dataset = dataset.numpy()
np.savetxt('data/{}.csv'.format(name), dataset)

