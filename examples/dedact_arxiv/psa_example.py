import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import rfi.utils as utils
import rfi.examples.ii_paper as ii_paper
from rfi.explainers.explainer import Explainer
from rfi.samplers.gaussian import GaussianSampler
from rfi.decorrelators.gaussian import NaiveGaussianDecorrelator

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

for simulation_id in range(len(simulations)):

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

    ex_d_ar_via = wrk.viafrom('ar_via', X_test.columns, X_test, y_test,
                               target='Y', nr_runs=20, show_pbar=True)
    ex_d_ar_via.to_csv(savepath=savepath, filename=ex_name + 'viafrom_ar.csv')
    ex_d_ar_via.decomp_wbarplots(col_wrap=None, fs=['total', 'PSA'])
    plt.savefig(savepath + ex_name + 'viafrom_ar_marg.pdf')

    ex_d_dr_from = wrk.viafrom('dr_from', X_test.columns, X_test, y_test,
                               target='Y', nr_runs=20, show_pbar=True)
    ex_d_dr_from.to_csv(savepath=savepath, filename=ex_name + 'viafrom_dr.csv')
    ex_d_dr_from.decomp_wbarplots(col_wrap=None, fs=['total', 'cycling'])
    plt.savefig(savepath + ex_name + 'viafrom_dr.pdf')

    ex_d_sage = wrk.viafrom('sage', X_test.columns, X_test, y_test,
                            target='Y', nr_runs=10, show_pbar=True,
                            nr_orderings=15, nr_resample_marginalize=30)
    ex_d_sage.to_csv(savepath=savepath, filename=ex_name + 'vaifrom_sage.csv')
    ex_d_sage.decomp_wbarplots(col_wrap=None, fs=['total', 'PSA'])
    plt.savefig(savepath + ex_name + 'viafrom_sage.pdf')

    ex_tdi_decomp = wrk.decomposition('tdi', X_test.columns, ordering, X_test, y_test, nr_orderings=50, nr_runs=5)
    ex_tdi_decomp.decomp_wbarplots(col_wrap=None, fs=['total', 'cycling'])

    # check whether cycling is completely explained by psa
    # ex_cycling_total = wrk.dr_from(['cycling'], ['PSA', 'biomarkers'], X_test.columns,
    #                                X_test, y_test)
    # ex_cycling_total.hbarplot()
    # ex_cycling_w_PSA = wrk.dr_from(['cycling'], ['PSA', 'biomarkers'], ['PSA'],
    #                                 X_test, y_test)

    # df = pd.DataFrame(index=ex_cycling_total.scores.index)
    # df['Total DR'] = ex_cycling_total.scores.values
    # df['DR from PSA'] = ex_cycling_w_PSA.scores.values

    # df = df.mean(level='sample')
    # sns.barplot(data=df)
    # plt.savefig(savepath + ex_name + 'dr_from_total_vs_psa.pdf')
