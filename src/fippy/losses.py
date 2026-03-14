"""Loss functions for feature importance computation.

All loss functions are observation-wise: they return an array of the same
length as y_true, with one loss value per observation.
"""
import numpy as np


def squared_error(y_true, y_pred):
    """Observation-wise squared error: (y - yhat)^2."""
    return (np.asarray(y_true) - np.asarray(y_pred)) ** 2


def absolute_error(y_true, y_pred):
    """Observation-wise absolute error: |y - yhat|."""
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))
