# fippy: A Python Library for Feature Importance

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

TODO