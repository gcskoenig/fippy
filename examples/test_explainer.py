import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import rfi.utils as utils
import rfi.examples.ii_paper as ii_paper
from rfi.explainers import Explainer
from rfi.samplers import GaussianSampler
from rfi.decorrelators import NaiveGaussianDecorrelator

import logging

logging.basicConfig(level=logging.INFO)

reg_lin = linear_model.LinearRegression()
savepath = 'paper_results/'
ex_identifier = '_final'

# datasets to use
n_train = 10 ** 6
n_test = 10 ** 3

simulations = [ii_paper.ii_psa]
train_on = ['biomarkers', 'cycling']
simulation_id = 0

ex_name = simulations[simulation_id].name + ex_identifier
xcolumns = simulations[simulation_id].sem.dag.var_names[:-1]
ycolumn = 'y'
data = simulations[simulation_id].get_train_test_data(xcolumns, ycolumn,
                                                      n_train=n_train,
                                                      n_test=n_test,
                                                      as_dataframes=True)
df_train, df_test = data
X_train, y_train = df_train[xcolumns], df_train[ycolumn]
X_test, y_test = df_test[xcolumns], df_test[ycolumn]

reg_lin.fit(X_train[train_on], y_train)


def mod_predict(X):
    return reg_lin.predict(X[train_on])


scoring = [mean_squared_error, r2_score]
names = ['MSE', 'r2_score']
models = [reg_lin]
m_names = ['LinearRegression']

print(mean_squared_error(y_test, mod_predict(X_test)))
print(r2_score(y_test, mod_predict(X_test)))

sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)
fsoi = X_train.columns
ordering = [tuple(fsoi)]
nr_orderings = utils.nr_unique_perm(ordering)

wrk = Explainer(mod_predict, fsoi, X_train,
                loss=mean_squared_error, sampler=sampler,
                decorrelator=decorrelator)

# test basic ordering based importance functions

ex1 = wrk.dis_from_ordering(tuple(fsoi), fsoi, X_test, y_test)
ex1.hbarplot()
plt.show()

ex2 = wrk.dis_from_ordering(('PSA', 'cycling', 'biomarkers'), fsoi, X_test, y_test)
ex2.hbarplot()
plt.show()

ex3 = wrk.ais_via_ordering(('PSA', 'cycling', 'biomarkers'),
                           fsoi, X_test, y_test)
ex3.hbarplot()
plt.show()

ex4 = wrk.ais_via_ordering(('cycling', 'PSA', 'biomarkers'),
                           fsoi, X_test, y_test)
ex4.hbarplot()
plt.show()

# test basic fixed importance functions

ex5 = wrk.dis_from_baselinefunc(fsoi, X_test, y_test, baseline='remainder')
ex5.hbarplot()
plt.show()

ex6 = wrk.ais_via_contextfunc(fsoi, X_test, y_test, context='empty')
ex6.hbarplot()
plt.show()

# test advanced importance methods

partial_ordering = [('cycling', 'PSA', 'biomarkers')]

ex7, orderings = wrk.sage(X_test, y_test, partial_ordering,
                          nr_orderings=10, nr_runs=3, nr_resample_marginalize=5)
ex7.hbarplot()
plt.show()

# test decompositions

ex8, orderings8 = wrk.decomposition('direct', fsoi, partial_ordering, X_test, y_test, approx=lambda x: x)
ex8.decomp_hbarplot()
plt.show()

ex9, orderings9 = wrk.decomposition('associative', fsoi, partial_ordering, X_test, y_test, approx=lambda x: x)
ex9.decomp_hbarplot()
plt.show()

ex10, orderings10 = wrk.decomposition('sage', fsoi, partial_ordering, X_test, y_test, approx=lambda x: x,
                                      sage_partial_ordering=partial_ordering, nr_orderings_sage=5)
ex10.decomp_hbarplot()
plt.show()


# debugging ai_via

ex11 = wrk.ais_via_contextfunc(X_test.columns, X_test, y_test, context='empty')
ex11.hbarplot()
plt.show()

ex11 = wrk.ais_via_contextfunc(['biomarkers'], X_test, y_test, context='empty')
ex11.hbarplot()
plt.show()


ex12 = wrk.ai_via(['PSA'], [], ['cycling'], X_test, y_test)
ex12.hbarplot()
plt.show()

ex12 = wrk.ai_via(['PSA'], [], [], X_test, y_test)
ex12.hbarplot()
plt.show()

ex13 = wrk.dis_from_baselinefunc(fsoi, X_test, y_test)
ex13.hbarplot()
plt.show()
