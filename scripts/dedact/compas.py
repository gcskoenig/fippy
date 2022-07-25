import pandas as pd

df = pd.read_csv('~/university/phd/2020/research/software/rfi/scripts/dedact/data/compas-scores-raw.csv')
# df = pd.read_csv('data/compas-scores-raw.csv')

timevars = ['Screening_Date', 'DateOfBirth']

def datetime_to_int(series):
    return pd.to_datetime(series).astype(int) / 10 ** 9

for var in timevars:
    df.loc[:, var] = datetime_to_int(df.loc[:, var]).copy()


target_cols = ['RawScore', 'DisplayText']
rm_cols = ['IsCompleted', 'IsDeleted', 'DecileScore', 'ScoreText']
ignore_cols_feasbility = ['FirstName', 'MiddleName', 'LastName']
rm_cols = rm_cols + ignore_cols_feasbility
remainder = list(set(df.columns) - set(target_cols) - set(rm_cols))

protected = ['Ethnic_Code_Text', 'Sex_Code_Text']
proxies = ['Language']

remainder = list(set(remainder) - set(protected))

categoricalvars = ['Agency_Text', 'AssessmentType', 'MiddleName', 'FirstName', 'ScaleSet',
                   'CustodyStatus', 'MaritalStatus', 'LegalStatus', 'RecSupervisionLevelText',
                   'Language', 'AssessmentReason', 'LastName']

remainder_cont = list(set(remainder) - set(categoricalvars))
X = df[remainder_cont]

cat_vars_enc = []

for varname in list(set(remainder) - set(remainder_cont)):
    print(f'converting {varname}')
    var_enc = pd.get_dummies(df[varname])
    cat_vars_enc = cat_vars_enc + list(var_enc.columns)
    X.loc[:, var_enc.columns] = var_enc.values

covars = remainder_cont + cat_vars_enc

print('number of covars', len(covars))

y = df[target_cols]
y = y['RawScore']
X = X

print('saving X, y')
X.to_csv('X.csv')
y.to_csv('y.csv')

print('training model')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestRegressor()

rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
print(mean_squared_error(y_test, rf.predict(X_test)))

import joblib

joblib.dump(rf, 'random_forest.joblib')
joblib.load('random_forest.joblib')

print('Loading rf successful.')


from rfi.explainers import Explainer
from rfi.samplers import GaussianSampler, UnivRFSampler, SequentialSampler

cat_sampler = UnivRFSampler(X_train)
cont_sampler = GaussianSampler(X_train)

sampler = SequentialSampler(X_train, cat_vars_enc, cat_sampler=cat_sampler, cont_sampler=cont_sampler)

wrk = Explainer(rf, X_train.columns, X_train, sampler=sampler, loss=mean_squared_error)

ex = wrk.ais_via_contextfunc(X_train.columns, X_test, y_test, X_train.columns)

import matplotlib.pyplot as plt

ex.hbarplot()
plt.savefig('ais_via.pdf')
plt.show()