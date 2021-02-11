"""
LOADING DATA AND FITTING MODEL
Using the data to practice conditional normalising flow sampler
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from rfi.examples.chains import confounding2, chain2
from rfi.samplers.cnflow import CNFSampler
from rfi.samplers.gaussian import GaussianSampler
import rfi.explainers.explainer as explainer

import logging
logging.basicConfig(level=logging.DEBUG)

# Synthesizing data
X_train, y_train, X_test, y_test = chain2.get_train_test_data(context_vars=('x1', 'x2', 'x3', 'x4'), target_var='y',
                                                              n_train=10 ** 4, n_test=10 ** 3, seed=300)
input_var_names = np.array(chain2.var_names[0:-1])

# Linear Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
risk = mean_squared_error(y_test, y_pred)

logging.info('Linear Model')
logging.info(input_var_names)
logging.info(model.coef_)
logging.debug('This is a debugging message.')


# Relative feature importance
G = np.array([1])
fsoi = np.array([0, 1, 2, 3], dtype=np.int16)

samplers_classes = [CNFSampler, GaussianSampler]

for sampler_class in samplers_classes:

    sampler = sampler_class(X_train)
    sampler.train(fsoi, G)  # Fitting sampler

    rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=mean_squared_error,
                                        fs_names=input_var_names)

    explanation = rfi_explainer.rfi(X_test, y_test, G)
    explanation.barplot()

    plt.title(f'{sampler.__class__.__name__}. G = {G}, N = {len(X_train) + len(X_test)}')
    plt.show()
