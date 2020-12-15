import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from rfi.backend.causality import DirectedAcyclicGraph, RandomGPGaussianNoiseSEM

# Creating random DAG
dag = DirectedAcyclicGraph.random_dag(3, 0.5, seed=42)
dag.plot_dag()
plt.show()

# Creating random ANM
sem = RandomGPGaussianNoiseSEM(dag, noise_std_dict={'x0': 1.0, 'x1': 1.0}, default_noise_std=0.2)


x1 = sem.sample(500, seed=42)
x2 = sem.sample(500, seed=43)

print(sem.conditional_pdf('x2', value=torch.tensor([-0.75]), context={'x0': torch.tensor([-2.5]), 'x1': torch.tensor([0.0])}))

data = pd.DataFrame(np.concatenate([x1.numpy(), x2.numpy()]), columns=dag.var_names)
data['sample'] = 500 * ['train'] + 500 * ['test']

# Plotting sampled data
sns.pairplot(data, hue='sample')
plt.show()
