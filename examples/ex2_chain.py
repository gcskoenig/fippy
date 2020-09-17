'''
LOADING DATA AND FITTING MODEL
'''

import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import math

name = 'chain2'

N=10**5
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
print(risk)
#0.24502383

r = permutation_importance(model, X_test[:, D], y_test, n_repeats=30, random_state=0)
print(names[0:-1])
print(r['importances_mean'])
# 3.84216251e-06 3.91947910e-05 6.65394070e-01 6.20763378e-01


'''
COMPUTING RFI
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rfi import rfi, cfi, plot_rfis, create_2nd_order_knockoff, paired_t
import pandas as pd
import copy

names = np.array(names)
loss = lambda x, y : np.power(x-y, 2)

D = np.arange(0, 4)

rfis = []
pvals = []
rfinames = []
Dnames = [r'$X_1$', r'$X_2$', r'$X_3$', r'$X_4$'] #,

G = np.array([])
res = rfi(model.predict, loss, G, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append(r'$PFI$')


res = cfi(model.predict, loss, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append(r'$CFI$')


G = np.array([0])
res = rfi(model.predict, loss, G, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append( r"$RFI_j^{X_1}$")

G = np.array([1])
res = rfi(model.predict, loss, G, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append( r"$RFI_j^{X_2}$")

G = np.array([1,0])
res = rfi(model.predict, loss, G, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append( r"$RFI_j^{X_2,X_1}$")


pvals = np.array(pvals)
np.save('results/{}_pvals.npy'.format(name), pvals)
print(pvals.shape)
#(5, 2, 30, 4)

print(pvals[:, :, 0, :])

plot_rfis(rfis, Dnames, rfinames, 'results/{}.pdf'.format(name), figsize=(60, 30), textformat='{:3.1f}')

