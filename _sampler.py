"""Sampler can be used sample perturbed versions of a variable conditional on a set G.

More details can be found in the class description
"""

class Sampler():
	"""Can be used to resample perturbed versions of a variable conditional on
	any set of variables G.

	More details here.

	Attributes:
		X_train: reference to training data.
		mask: features of interest.
	"""
	def __init__(self, X_train, mask):
		"""Initialize Sampler with X_train and mask."""
		self.X_train = X_train
		self.mask = mask

	def train(G):
		"""Trains sampler using the training dataset to resample
		relative to any variable set G.

		Args:
			G: arbitrary set of variables.

		Returns:
			Nothing. Now the sample function can be used
			to resample on seen or unseen data.
		"""
		pass


	def sample(X_eval):
		"""Sample unmasked variables using trained resampler.

		Args:
			X_eval: Data for which sampling shall be performed.

		Returns:
			Resampled data for the unmasked variables.
			np.array with shape (X_eval.shape[0], # of unmasked features)
		"""
		pass