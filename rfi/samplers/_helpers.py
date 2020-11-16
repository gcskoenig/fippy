import numpy as np


def sample_id(X_test, ixs):
	''' Simple sampler that returns identical data for specified indexes

	Args:
		X_test: data
		ixs: ixs for columns to be "sampled"
	'''
	return X_test[:, ixs]
