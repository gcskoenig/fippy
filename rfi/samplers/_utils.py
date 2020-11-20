import numpy as np


def sample_id(ix):
	''' Simple sampler that returns sample function that simply
	copies the original data

	Args:
		X_test: data
		ixs: ixs for columns to be "sampled"
	'''
	def sample(X_test):
		return X_test[:, ix]
	return sample

def sample_perm(ix):
	''' Simple sampler that permutes the values

	Args:
		X_test: data
		ixs: ixs for columns to be "sampled"
	'''
	def sample(X_test):
		xs = np.array(X_test[:, ix])
		xs = np.random.permutation(xs)
		return xs
	return sample