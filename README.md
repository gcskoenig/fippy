# A Python Library for Feature Importance

![Python package](https://github.com/gcskoenig/rfi/workflows/Python%20package/badge.svg?branch=master)

## Disclaimer

The package is still under development and in early testing stages. Therefore, we do not guarantee stability. 

## Functionality

In this library we offer an implementation of Relative Feature Importance including:

- a variety of conditional sampling techniques
- visualization
- significance testing

The library is accompagnied by our ICPR paper [[arXiv]](https://arxiv.org/abs/2007.08283)

## Sampling techniques

- Gaussian Sampling
- Conditional Normalizing Flows

## Installation

The package assumes a current version of python, i.e. `>= 3.9.7`. Create a virtual environment and manually install the following dependencies: `torch` (following the installation instructions on the pytorch website), `ray` including tuning functionality (e.g. `pip install -U "ray[tune]"`), `scikit-learn` (following the instructions on their website). Then install the `requirements.txt` using `pip install -r [path-to-rfi-folder]/requirements.txt`. Then you can install the rfi package using `pip install -e [path-to-rfi-folder]`. 
