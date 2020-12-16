import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from rfi.backend.causality import DirectedAcyclicGraph, PostNonLinearLaplaceSEM, PostNonLinearMultiplicativeHalfNormalSEM, \
    LinearGaussianNoiseSEM, RandomGPGaussianNoiseSEM

# Creating random DAG
dag = DirectedAcyclicGraph.random_dag(6, 0.2, seed=42)
dag.plot_dag()
plt.show()

sem_classes = [RandomGPGaussianNoiseSEM, PostNonLinearLaplaceSEM, PostNonLinearMultiplicativeHalfNormalSEM]

for sem_cls in sem_classes:

    # Creating random SEM
    sem = sem_cls(dag, default_noise_std_bounds=(0.0, 0.5))

    x1 = sem.sample(500, seed=42)
    x2 = sem.sample(500, seed=43)

    print(sem.conditional_pdf('x1', value=torch.tensor([0.75, 0.95]), context={'x0': torch.tensor([0.1, 0.2]),
                                                                               'x1': torch.tensor([0.1, 0.3])}))

    data = pd.DataFrame(np.concatenate([x1.numpy(), x2.numpy()]), columns=dag.var_names)
    data['sample'] = 500 * ['train'] + 500 * ['test']

    # Plotting sampled data
    g = sns.pairplot(data, hue='sample')
    # g.set(ylim=(data[dag.var_names].min().min(), data[dag.var_names].max().max()))
    g.fig.suptitle(sem.__class__.__name__)
    plt.show()
