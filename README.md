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

```
python setup.py install --user
```

or alternatively

```
python setup.py develop
```

## Use Case

See `examples/dedact_arxiv`for two examples that are explained in the dedact paper (https://arxiv.org/abs/2106.08086)
