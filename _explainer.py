"""Explainers compute RFI relative to any set of features G.

Different sampling algorithms and loss functions can be used.
More details in the docstring for the class Explainer.
"""


class Explainer():
	"""Uses Relative Feature Importance to compute the importance of features
	relative to any set of features G.

	Default samplers or loss function can be defined.
	Masks allows to specify for which features importance
	shall be computed.

	Attributes:
		model: Model or predict function.
		mask: Features to be explained.
		sampler: default sampler.
		loss: default loss.
	"""
	def __init__(self, model, mask, sampler=None, loss=None):
		"""Inits Explainer with model, mask and potentially sampler and loss"""
		self.model = model
		self.mask = mask
		self.sampler = sampler
		self.loss = loss


	def rfi(self, X_test, y_test, G, sampler=None, loss=None):
		"""Computes Relative Feature importance

		Args:


		Returns:

		"""
		pass