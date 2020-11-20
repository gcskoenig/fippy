'''
LOADING DATA AND FITTING MODEL
'''

import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#from sklearn.inspection import permutation_importance
import math

name = 'chain2'

N=10**3
dataset = np.loadtxt('data/{}.csv'.format(name), dtype=np.float32)
D = np.arange(0, 4)

splitpoint = math.floor(N*0.9)
ix_train = np.arange(0, splitpoint, 1)
ix_test = np.arange(splitpoint, N, 1)

X_train, y_train = dataset[ix_train, :-1], dataset[ix_train,-1]
X_test, y_test = dataset[ix_test, :-1], dataset[ix_test,-1]

# Linear Model
print('Linear Model')

model = LinearRegression()
model.fit(X_train[:, D], y_train)
names = ['x1', 'x2', 'x3', 'x4', 'y']
print(names[0:-1])
print(model.coef_)
#0.00297259 -0.00854544  1.0052813   1.0001918

y_pred = model.predict(X_test[:, D])
risk = mean_squared_error(y_test, y_pred)



'''
Using the data to practice gaussian sampling
'''

import rfi.samplers.gaussian as gaussian

G = np.array([])
fsoi = np.array([0, 1, 2, 3], dtype=np.int16)
names = np.array(names)

sampler = gaussian.GaussianSampler(X_train)	

sampler.train(fsoi, G)
sampler.train(fsoi, G)

import rfi.explainers.explainer as explainer

rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=mean_squared_error,
									fs_names=names)


G = np.array([])
explanation = rfi_explainer.rfi(X_test, y_test, G)
explanation.barplot()

