# A Python Library for Relative Feature Importance (RFI)

In this library we offer an implementation of Relative Feature Importance including:

- a variety of conditional sampling techniques
- visualization
- significance testing

The library is accompagnied by our ICPR paper [[arXiv]](https://arxiv.org/abs/2007.08283)

## Sampling techniques

- Gaussian Sampling
- Conditional Normalizing Flows

## Installation

```
python setup.py install --user
```

## Use Case

```
# load data
# fit model

import rfi.samplers.gaussian as gaussian
import rfi.explainers.explainer as explainer
import matplotlib.pyplot as plt

fsoi = np.array([0, 1, 2, 3], dtype=np.int16)
names = np.array(names)

sampler = gaussian.GaussianSampler(X_train)
rfi_explainer = explainer.Explainer(model.predict, fsoi, X_train, sampler=sampler, loss=mean_squared_error,
									fs_names=names)
                  
G = np.array([1])
for f in fsoi:
    sampler.train([f], G)
ex = rfi_explainer.rfi(X_test, y_test, G)

print(ex.fsoi_names)
print(ex.fi_means())

ex.barplot()
plt.show()

```
