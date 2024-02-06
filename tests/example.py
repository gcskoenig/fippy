import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import category_encoders as ce


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
cat_sampler = UnivRFSampler(X_train)
cont_sampler = ContUnivRFSampler(X_train)
sampler = SequentialSampler(X_train, categorical_fs=X_train.select_dtypes(include='object').columns,
                            cont_sampler=cont_sampler, cat_sampler=cat_sampler)

# create explainer
wrk = Explainer(pipe.predict, X.columns, X_train, sampler)

wrk.dis_from_baselinefunc(X.columns, X_test, y_test, X.columns)