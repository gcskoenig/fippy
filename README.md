# fippy: A Python Library for Feature Importance 🦦

## Disclaimer

The package is still under development and in early testing stages. Therefore, we do not guarantee stability. The package was previously called `rfi`and accompagnies our paper on Relative Feature Importance. [[arXiv]](https://arxiv.org/abs/2007.08283)


## Functionality

In this library we offer an implementation of various feature importance techniques including

- Permutation Feature Importance (PFI)
- Conditional Feature Importance (CFI)
- Relative Feature Importance (RFI)
- marginal and conditional SAGE

For the conditional-sampling-based techniques, the package includes a range of different samplers.

- Random forest based categorical sampling (univariate)
- Random forest based continuous sampling (univariate)
- Sequential samplers that allow to combine univariate samplers for sampling from multivariate conditional densities
- Gaussian samplers (both univariate and multivariate conditional densities)
- Mixtures of Gaussians
- Conditional Normalizing Flows


## Installation

The package assumes a current version of python, i.e. `>= 3.9.7`. Create a virtual environment and manually install the following dependencies: `torch` (following the installation instructions on the pytorch website), `ray` including tuning functionality (e.g. `pip install -U "ray[tune]"`), `scikit-learn` (following the instructions on their website). Then install the `requirements.txt` using `pip install -r [path-to-rfi-folder]/requirements.txt`. Then you can install the rfi package using `pip install -e [path-to-fippy-folder]`. 


## Usage

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import category_encoders as ce
import logging

logging.basicConfig(level=logging.INFO)

## to specify by user
savepath = '~/Downloads/'

## load and prepare data
sns.get_dataset_names()
df = sns.load_dataset('mpg')
df.dtypes
df.drop('name', axis=1, inplace=True)
if df.isna().any().any():
    df.dropna(inplace=True)

X, y = df.drop('mpg', axis=1), df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## create and fit pipeline
ohe = ce.OneHotEncoder()
rf = RandomForestRegressor()
pipe = make_pipeline(ohe, rf)

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

## setup fippy explainer to explain the model
from fippy.explainers import Explainer
from fippy.samplers import UnivRFSampler, ContUnivRFSampler, SequentialSampler

# create univariate categorical and continuous samplers and compile them to sequential sampler
cat_fs = X_train.select_dtypes(include='object').columns # specify categorical features
cat_sampler = UnivRFSampler(X_train, cat_inputs=cat_fs)
cont_sampler = ContUnivRFSampler(X_train, cat_inputs=cat_fs)
sampler = SequentialSampler(X_train, categorical_fs=cat_fs,
                            cont_sampler=cont_sampler, cat_sampler=cat_sampler)

# create explainer
wrk = Explainer(pipe.predict, X.columns, X_train, sampler, mean_squared_error)


## compute PFI
ex_pfi = wrk.dis_from_baselinefunc(X.columns, X_test, y_test, X.columns, baseline='remainder')
ex_pfi.hbarplot()
plt.show()

# mean feature importance for each feature (and respective standard deviation)
ex_pfi.fi_means_stds()

# save explanation to csv 
ex_pfi.to_csv(savepath=savepath, filename='pfi.csv')

# load explanation from csv again
from fippy.explanation import Explanation
ex_pfi = Explanation.from_csv(savepath + 'pfi.csv')


## compute CFI

ex_cfi = wrk.ais_via_contextfunc(X.columns, X_test, y_test, context='remainder')
ex_cfi.hbarplot()
plt.show()

ex_cfi.fi_means_stds()
ex_cfi.to_csv(savepath=savepath, filename='cfi.csv')

## compute conditional SAGE

ordering = [tuple(X.columns)]
ex_sage, sage_orderings = wrk.sage(X_test, y_test, ordering, method='associative', nr_orderings=20, nr_runs=3)
ex_sage.hbarplot()
plt.show()

ex_sage.fi_means_stds()
ex_sage.to_csv(savepath=savepath, filename='sage.csv')
```
