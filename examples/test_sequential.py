from rfi.samplers.sequential import SequentialSampler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

N = 10**5

cat_data = np.random.randint(0, 5, 3*N).reshape(N, 3)
cont_data = np.random.randn(3*N).reshape(N, 3)
data = np.concatenate((cat_data, cont_data), axis=1)

df = pd.DataFrame(data, columns=['cat1', 'cat2', 'cat3', 'cont1', 'cont2', 'cont3'])
X = df[['cat2', 'cat3', 'cont1', 'cont2', 'cont3']]
y = df[['cat1']]

model = RandomForestClassifier()
model.fit(X, y)

adj_matrix = np.zeros((5, 5))

sampler = SequentialSampler(X, adj_matrix, ['cat2', 'cat3'])