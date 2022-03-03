from rfi.samplers import SequentialSampler, GaussianSampler, UnivRFSampler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


N = 10**3

cat_data = np.random.randint(0, 5, 3*N).reshape(N, 3)
cont_data = np.random.randn(3*N).reshape(N, 3)
data = np.concatenate((cat_data, cont_data), axis=1)

df = pd.DataFrame(data, columns=['cat1', 'cat2', 'cat3', 'cont1', 'cont2', 'cont3'])
X = df[['cat2', 'cat3', 'cont1', 'cont2', 'cont3']]
y = df[['cat1']]

# model = RandomForestClassifier()
# model.fit(X, y)

# adj_matrix = np.zeros((5, 5))
adj_matrix = np.diag(np.ones((4)), 1)
adj_matrix = pd.DataFrame(adj_matrix, columns=X.columns, index=X.columns)

cont_sampler = GaussianSampler(X)
cat_sampler = UnivRFSampler(X)

sampler = SequentialSampler(X, adj_matrix, ['cat2', 'cat3'], cont_sampler=cont_sampler, cat_sampler=cat_sampler)

sampler.train(['cat2', 'cont2'], ['cont3', 'cat3'])
sample = sampler.sample(X, ['cat2', 'cont2'], ['cont3', 'cat3'], num_samples=5)