#!/usr/bin/env python

"""
Nonconformity functions.
"""

# Authors: Henrik Linusson

from __future__ import division

import abc
import numpy as np
import sklearn.base
from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.base import OobClassifierAdapter, OobRegressorAdapter

# -----------------------------------------------------------------------------
# Error functions
# -----------------------------------------------------------------------------

class ClassificationErrFunc(object):
	"""Base class for classification model error functions.
	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(ClassificationErrFunc, self).__init__()

	@abc.abstractmethod
	def apply(self, prediction, y):
		"""Apply the nonconformity function.

		Parameters
		----------
		prediction : numpy array of shape [n_samples, n_classes]
			Class probability estimates for each sample.

		y : numpy array of shape [n_samples]
			True output labels of each sample.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of the samples.
		"""
		pass


class RegressionErrFunc(object):
	"""Base class for regression model error functions.
	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(RegressionErrFunc, self).__init__()

	@abc.abstractmethod
	def apply(self, prediction, y):#, norm=None, beta=0):
		"""Apply the nonconformity function.

		Parameters
		----------
		prediction : numpy array of shape [n_samples, n_classes]
			Class probability estimates for each sample.

		y : numpy array of shape [n_samples]
			True output labels of each sample.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of the samples.
		"""
		pass

	@abc.abstractmethod
	def apply_inverse(self, nc, significance):#, norm=None, beta=0):
		"""Apply the inverse of the nonconformity function (i.e.,
		calculate prediction interval).

		Parameters
		----------
		nc : numpy array of shape [n_calibration_samples]
			Nonconformity scores obtained for conformal predictor.

		significance : float
			Significance level (0, 1).

		Returns
		-------
		interval : numpy array of shape [n_samples, 2]
			Minimum and maximum interval boundaries for each prediction.
		"""
		pass


class InverseProbabilityErrFunc(ClassificationErrFunc):
	"""Calculates the probability of not predicting the correct class.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		1 - \hat{P}(y_i | x) \, .
	"""

	def __init__(self):
		super(InverseProbabilityErrFunc, self).__init__()

	def apply(self, prediction, y):
		prob = np.zeros(y.size, dtype=np.float32)
		for i, y_ in enumerate(y):
			if y_ >= prediction.shape[1]:
				prob[i] = 0
			else:
				prob[i] = prediction[i, int(y_)]
		return 1 - prob


class MarginErrFunc(ClassificationErrFunc):
	"""
	Calculates the margin error.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		0.5 - \dfrac{\hat{P}(y_i | x) - max_{y \, != \, y_i} \hat{P}(y | x)}{2}
	"""

	def __init__(self):
		super(MarginErrFunc, self).__init__()

	def apply(self, prediction, y):
		prob = np.zeros(y.size, dtype=np.float32)
		for i, y_ in enumerate(y):
			if y_ >= prediction.shape[1]:
				prob[i] = 0
			else:
				prob[i] = prediction[i, int(y_)]
				prediction[i, int(y_)] = -np.inf
		return 0.5 - ((prob - prediction.max(axis=1)) / 2)


class AbsErrorErrFunc(RegressionErrFunc):
	"""Calculates absolute error nonconformity for regression problems.

		For each correct output in ``y``, nonconformity is defined as

		.. math::
			| y_i - \hat{y}_i |
	"""

	def __init__(self):
		super(AbsErrorErrFunc, self).__init__()

	def apply(self, prediction, y):
		return np.abs(prediction - y)

	def apply_inverse(self, nc, significance):
		nc = np.sort(nc)[::-1]
		border = int(np.floor(significance * (nc.size + 1))) - 1
		# TODO: should probably warn against too few calibration examples
		border = min(max(border, 0), nc.size - 1)
		return np.vstack([nc[border], nc[border]])


class SignErrorErrFunc(RegressionErrFunc):
	"""Calculates signed error nonconformity for regression problems.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		y_i - \hat{y}_i

	References
	----------
	.. [1] Linusson, Henrik, Ulf Johansson, and Tuve Lofstrom.
		Signed-error conformal regression. Pacific-Asia Conference on Knowledge
		Discovery and Data Mining. Springer International Publishing, 2014.
	"""

	def __init__(self):
		super(SignErrorErrFunc, self).__init__()

	def apply(self, prediction, y):
		return (prediction - y)

	def apply_inverse(self, nc, significance):
		nc = np.sort(nc)[::-1]
		upper = int(np.floor((significance / 2) * (nc.size + 1)))
		lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
		# TODO: should probably warn against too few calibration examples
		upper = min(max(upper, 0), nc.size - 1)
		lower = max(min(lower, nc.size - 1), 0)
		return np.vstack([-nc[lower], nc[upper]])


# -----------------------------------------------------------------------------
# Base nonconformity scorer
# -----------------------------------------------------------------------------
class BaseScorer(sklearn.base.BaseEstimator):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(BaseScorer, self).__init__()

	@abc.abstractmethod
	def fit(self, x, y):
		pass

	@abc.abstractmethod
	def score(self, x, y=None):
		pass


class RegressorNormalizer(BaseScorer):
	def __init__(self, base_model, normalizer_model, err_func):
		super(RegressorNormalizer, self).__init__()
		self.base_model = base_model
		self.normalizer_model = normalizer_model
		self.err_func = err_func

	def fit(self, x, y, y_hat=None):
		if self.base_model is None:
			try:
				if y_hat is None:
					raise ValueError('y_hat cannot be None when base_model is None')
			except ValueError:
				raise
			residual_prediction=y_hat
		else:
			residual_prediction = self.base_model.predict(x)
		residual_error = np.abs(self.err_func.apply(residual_prediction, y))
		residual_error += 0.00001 # Add small term to avoid log(0)
		log_err = np.log(residual_error)
		self.normalizer_model.fit(x, log_err)

	def score(self, x, y=None):
		norm = np.exp(self.normalizer_model.predict(x))
		return norm


class NcFactory(object):
	@staticmethod
	def create_nc(model, err_func=None, normalizer_model=None, oob=False):
		if normalizer_model is not None:
			normalizer_adapter = RegressorAdapter(normalizer_model)
		else:
			normalizer_adapter = None

		if model is None:
			err_func = AbsErrorErrFunc() if err_func is None else err_func
			adapter = None
			if normalizer_adapter is not None:
				normalizer = RegressorNormalizer(adapter,
												 normalizer_adapter,
												 err_func)
				return RegressorNc(adapter, err_func, normalizer)
			else:
				return RegressorNc(adapter, err_func)
		elif isinstance(model, sklearn.base.ClassifierMixin):
			err_func = MarginErrFunc() if err_func is None else err_func
			if oob:
				c = sklearn.base.clone(model)
				c.fit([[0], [1]], [0, 1])
				if hasattr(c, 'oob_decision_function_'):
					adapter = OobClassifierAdapter(model)
				else:
					raise AttributeError('Cannot use out-of-bag '
					                      'calibration with {}'.format(
						model.__class__.__name__
					))
			else:
				adapter = ClassifierAdapter(model)

			if normalizer_adapter is not None:
				normalizer = RegressorNormalizer(adapter,
				                                 normalizer_adapter,
				                                 err_func)
				return ClassifierNc(adapter, err_func, normalizer)
			else:
				return ClassifierNc(adapter, err_func)

		elif isinstance(model, sklearn.base.RegressorMixin):
			err_func = AbsErrorErrFunc() if err_func is None else err_func
			if oob:
				c = sklearn.base.clone(model)
				c.fit([[0], [1]], [0, 1])
				if hasattr(c, 'oob_prediction_'):
					adapter = OobRegressorAdapter(model)
				else:
					raise AttributeError('Cannot use out-of-bag '
					                     'calibration with {}'.format(
						model.__class__.__name__
					))
			else:
				adapter = RegressorAdapter(model)

			if normalizer_adapter is not None:
				normalizer = RegressorNormalizer(adapter,
				                                 normalizer_adapter,
				                                 err_func)
				return RegressorNc(adapter, err_func, normalizer)
			else:
				return RegressorNc(adapter, err_func)


class BaseModelNc(BaseScorer):
	"""Base class for nonconformity scorers based on an underlying model.

	Parameters
	----------
	model : ClassifierAdapter or RegressorAdapter
		Underlying classification model used for calculating nonconformity
		scores.

	err_func : ClassificationErrFunc or RegressionErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.
	"""
	def __init__(self, model, err_func, normalizer=None, beta=0):
		super(BaseModelNc, self).__init__()
		self.err_func = err_func
		self.model = model
		self.normalizer = normalizer
		self.beta = beta

		# If we use sklearn.base.clone (e.g., during cross-validation),
		# object references get jumbled, so we need to make sure that the
		# normalizer has a reference to the proper model adapter, if applicable.
		if (self.normalizer is not None and
			hasattr(self.normalizer, 'base_model')):
			self.normalizer.base_model = self.model

		self.last_x, self.last_y = None, None
		self.last_prediction = None
		self.clean = False

	def fit(self, x, y, y_hat=None):
		"""Fits the underlying model of the nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the underlying model.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the underlying model.

		y_hat : numpy array of shape [n_samples]
			Outputs of examples for which to calculate a nonconformity score when bfase model is not available.

		Returns
		-------
		None
		"""
		if self.model is not None:
			self.model.fit(x, y)
		else:
			try:
				if y_hat is None:
					raise ValueError('y_hat cannot be None when base_model is None')
			except ValueError:
				raise
		if self.normalizer is not None:
			self.normalizer.fit(x, y)
		self.clean = False

	def score(self, x, y=None, y_hat=None):
		"""Calculates the nonconformity score of a set of samples.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for which to calculate a nonconformity score.

		y : numpy array of shape [n_samples]
			Outputs of examples for which to calculate a nonconformity score.

		y_hat : numpy array of shape [n_samples]
			Outputs of examples for which to calculate a nonconformity score when base model is not available.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of samples.
		"""
		if self.model is None:
			try:
				if y_hat is None:
					raise ValueError('y_hat cannot be None when base_model is None')
			except ValueError:
				raise
			prediction = y_hat

		else:
			prediction = self.model.predict(x)
		n_test = x.shape[0]
		if self.normalizer is not None:
			norm = self.normalizer.score(x) + self.beta
		else:
			norm = np.ones(n_test)

		return self.err_func.apply(prediction, y) / norm


# -----------------------------------------------------------------------------
# Classification nonconformity scorers
# -----------------------------------------------------------------------------
class ClassifierNc(BaseModelNc):
	"""Nonconformity scorer using an underlying class probability estimating
	model.

	Parameters
	----------
	model : ClassifierAdapter
		Underlying classification model used for calculating nonconformity
		scores.

	err_func : ClassificationErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.

	Attributes
	----------
	model : ClassifierAdapter
		Underlying model object.

	err_func : ClassificationErrFunc
		Scorer function used to calculate nonconformity scores.

	See also
	--------
	RegressorNc, NormalizedRegressorNc
	"""
	def __init__(self,
	             model,
	             err_func=MarginErrFunc(),
	             normalizer=None,
	             beta=0):
		super(ClassifierNc, self).__init__(model,
		                                   err_func,
		                                   normalizer,
		                                   beta)


# -----------------------------------------------------------------------------
# Regression nonconformity scorers
# -----------------------------------------------------------------------------
class RegressorNc(BaseModelNc):
	"""Nonconformity scorer using an underlying regression model.

	Parameters
	----------
	model : RegressorAdapter
		Underlying regression model used for calculating nonconformity scores.

	err_func : RegressionErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.

	Attributes
	----------
	model : RegressorAdapter
		Underlying model object.

	err_func : RegressionErrFunc
		Scorer function used to calculate nonconformity scores.

	See also
	--------
	ProbEstClassifierNc, NormalizedRegressorNc
	"""
	def __init__(self,
	             model,
	             err_func=AbsErrorErrFunc(),
	             normalizer=None,
	             beta=0):
		super(RegressorNc, self).__init__(model,
		                                  err_func,
		                                  normalizer,
		                                  beta)

	def predict(self, x, nc, y_hat=None,significance=None):
		"""Constructs prediction intervals for a set of test examples.

		Predicts the output of each test pattern using the underlying model,
		and applies the (partial) inverse nonconformity function to each
		prediction, resulting in a prediction interval for each test pattern.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		y_hat: numpy array of shape [n_samples]
			Outputs of examples from the fitted underlying model.

		significance : float
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then intervals for
			all significance levels (0.01, 0.02, ..., 0.99) are output in a
			3d-matrix.

		Returns
		-------
		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
			If significance is ``None``, then p contains the interval (minimum
			and maximum boundaries) for each test pattern, and each significance
			level (0.01, 0.02, ..., 0.99). If significance is a float between
			0 and 1, then p contains the prediction intervals (minimum and
			maximum	boundaries) for the set of test patterns at the chosen
			significance level.
		"""
		n_test = x.shape[0]

		if self.model is None:
			try:
				if y_hat is None:
					raise ValueError('y_hat cannot be None when base_model is None')
			except ValueError:
				raise
			prediction = y_hat

		else:
			prediction = self.model.predict(x)

		if self.normalizer is not None:
			norm = self.normalizer.score(x) + self.beta
		else:
			norm = np.ones(n_test)

		if significance:
			intervals = np.zeros((x.shape[0], 2))
			err_dist = self.err_func.apply_inverse(nc, significance)
			err_dist = np.hstack([err_dist] * n_test)
			err_dist *= norm

			intervals[:, 0] = prediction - err_dist[0, :]
			intervals[:, 1] = prediction + err_dist[1, :]

			return intervals
		else:
			significance = np.arange(0.01, 1.0, 0.01)
			intervals = np.zeros((x.shape[0], 2, significance.size))

			for i, s in enumerate(significance):
				err_dist = self.err_func.apply_inverse(nc, s)
				err_dist = np.hstack([err_dist] * n_test)
				err_dist *= norm

				intervals[:, 0, i] = prediction - err_dist[0, :]
				intervals[:, 1, i] = prediction + err_dist[0, :]

			return intervals
