import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import rfi.examples.chains as chains
from rfi.explainers.explainer import Explainer
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator
from rfi.explanation.decomposition import DecompositionExplanation

import logging

logging.basicConfig(level=logging.INFO)

reg_lin = linear_model.LinearRegression()
#savepath = 'paper_results/'
#ex_identifier = '_0401_sage_decomp_'

# datasets to use
n_train, n_test = 10 ** 6 * 4, 10 ** 3


simulations = [chains.chain_short]

simulation_id = 0

ex_name = simulations[simulation_id].name  #+ ex_identifier
xcolumns = simulations[simulation_id].sem.dag.var_names[:-1]
ycolumn = [simulations[simulation_id].sem.dag.var_names[-1]]
data = simulations[simulation_id].get_train_test_data(xcolumns, ycolumn,
                                                      n_train=n_train,
                                                      n_test=n_test,
                                                      as_dataframes=True)
df_train, df_test = data
X_train, y_train = df_train[xcolumns], df_train[ycolumn]
X_test, y_test = df_test[xcolumns], df_test[ycolumn]

# fit models

reg_lin.fit(X_train, y_train)

scoring = [mean_squared_error, r2_score]
names = ['MSE', 'r2_score']
models = [reg_lin]
m_names = ['LinearRegression']

for kk in range(len(models)):
    model = models[kk]
    print('Model: {}'.format(m_names[kk]))
    for jj in np.arange(len(names)):
        print('{}: {}'.format(names[jj],
                              scoring[jj](y_test, model.predict(X_test))))

# explain model

sampler = GaussianSampler(X_train)
decorrelator = NaiveGaussianDecorrelator(X_train)
fsoi = X_train.columns
ordering = [tuple(fsoi)]

wrk = Explainer(reg_lin.predict, fsoi, X_train,
                loss=mean_squared_error, sampler=sampler,
                decorrelator=decorrelator)

# actually do a decomposition

tupl = wrk.decomposition('sage', fsoi, ordering, X_test, y_test,
                         sage_partial_ordering=ordering,
                         nr_orderings_sage=10,
                         nr_orderings=10)

expl_sage, orderings = tupl
expl_sage.to_csv(savepath=savepath, filename=ex_name + 'sage_decmp.csv')
expl_sage.decomp_wbarplots(col_wrap=4)
plt.savefig(savepath + ex_name + 'sage_decomp.pdf')

expl_sage = DecompositionExplanation.from_csv(path=savepath, filename=ex_name + 'sage_decmp.csv')
expl_sage.decomp_wbarplots(col_wrap=4, fs=['total', 'age', 'race', 'sex'])

tupl = wrk.decomposition('tdi', fsoi, ordering, X_test, y_test,
                         nr_orderings=10)
expl_tdi_decomp, orderings = tupl
expl_tdi_decomp.to_csv(savepath=savepath, filename=ex_name + 'tdi_decmp.csv')
expl_tdi_decomp.decomp_wbarplots(col_wrap=4)
plt.savefig(savepath + ex_name + 'tdi_decmp.pdf')

expl_tdi_decomp.decomp_wbarplots(col_wrap=4,
                                 fs=['total', 'education num', 'workclass',
                                     'occupation'])
