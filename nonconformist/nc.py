#!/usr/bin/env python

"""
Nonconformity functions.
"""

# Authors: Henrik Linusson

from __future__ import division

import abc
import numpy as np
from scipy.stats import pearsonr
from sklearn.base import clone
from nonconformist.base import ClassifierAdapter, RegressorAdapter


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
	def apply(self, prediction, y, norm=None, beta=0):
		"""Apply the nonconformity function.

		Parameters
		----------
		prediction : numpy array of shape [n_samples, n_classes]
			Class probability estimates for each sample.

		y : numpy array of shape [n_samples]
			True output labels of each sample.

		norm : numpy array of shape [n_samples]
			Normalization values for normalized nonconformity.

		beta : float
			Beta parameter for normalized nonconformity. Larger beta reduces
			influence of normalization model.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of the samples.
		"""
		pass

	@abc.abstractmethod
	def apply_inverse(self, prediction, nc, significance, norm=None, beta=0):
		"""Apply the inverse of the nonconformity function (i.e.,
		calculate prediction interval).

		Parameters
		----------
		prediction : numpy array of shape [n_samples]
			Point (regression) predictions of a test examples.

		nc : numpy array of shape [n_calibration_samples]
			Nonconformity scores obtained for conformal predictor.

		significance : float
			Significance level (0, 1).

		norm : numpy array of shape [n_samples]
			Normalization values for normalized nonconformity.

		beta : float
			Beta parameter for normalized nonconformity. Larger beta reduces
			influence of normalization model.

		Returns
		-------
		interval : numpy array of shape [n_samples, 2]
			Minimum and maximum interval boundaries for each prediction.
		"""
		pass

	@staticmethod
	def _check_norm_beta(norm, beta):
		if norm is None:
			norm = 1
			beta = 0
		elif beta is None:
			beta = 0

		return norm, beta


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

	def apply(self, prediction, y, norm=None, beta=0):
		norm, beta = self._check_norm_beta(norm, beta)
		return np.abs(prediction - y) / (norm + beta)

	def apply_inverse(self, prediction, nc, significance, norm=None, beta=0):
		norm, beta = self._check_norm_beta(norm, beta)
		nc = np.sort(nc)[::-1]
		border = int(np.floor(significance * (nc.size + 1))) - 1
		# TODO: should probably warn against too few calibration examples
		border = min(max(border, 0), nc.size - 1)
		return np.vstack([prediction - (nc[border] * (norm + beta)),
		                  prediction + (nc[border] * (norm + beta))]).T


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

	def apply(self, prediction, y, norm=None, beta=0):
		norm, beta = self._check_norm_beta(norm, beta)
		return (prediction - y) / (norm + beta)

	def apply_inverse(self, prediction, nc, significance, norm=None, beta=0):
		norm, beta = self._check_norm_beta(norm, beta)
		nc = np.sort(nc)[::-1]
		upper = int(np.floor((significance / 2) * (nc.size + 1)))
		lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
		# TODO: should probably warn against too few calibration examples
		upper = min(max(upper, 0), nc.size - 1)
		lower = max(min(lower, nc.size - 1), 0)
		return np.vstack([prediction + (nc[lower] * (norm + beta)),
		                  prediction + (nc[upper] * (norm + beta))]).T


# -----------------------------------------------------------------------------
# Base nonconformity scorer
# -----------------------------------------------------------------------------
class BaseModelNc(object):
	"""Base class for nonconformity scorers based on an underlying model.

	Parameters
	----------
	model : ClassifierAdapter or RegressorAdapter
		Underlying classification model used for calculating nonconformity
		scores.

	err_func : ClassificationErrFunc or RegressionErrFunc
		Error function object.
	"""
	def __init__(self, model, err_func):
		self.err_func = err_func
		self.model = model

		self.last_x, self.last_y = None, None
		self.last_prediction = None
		self.clean = False

	def fit(self, x, y):
		"""Fits the underlying model of the nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the underlying model.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the underlying model.

		Returns
		-------
		None
		"""
		self.model.fit(x, y)
		self.clean = False

	def calc_nc(self, x, y):
		"""Calculates the nonconformity score of a set of samples.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for which to calculate a nonconformity score.

		y : numpy array of shape [n_samples]
			Outputs of examples for which to calculate a nonconformity score.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of samples.
		"""
		prediction = self.model.predict(x)
		return self.err_func.apply(prediction, y)

	def get_params(self, deep=False):
		if deep:
			return {
				'model': clone(self.model),
				'err_func': self.err_func
			}
		else:
			return {
				'model': self.model,
				'err_func': self.err_func
			}


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
	             err_func=MarginErrFunc()):
		super(ClassifierNc, self).__init__(model,
		                                   err_func)


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
	             err_func=AbsErrorErrFunc()):
		super(RegressorNc, self).__init__(model,
		                                  err_func)

	def predict(self, x, nc, significance=None):
		"""Constructs prediction intervals for a set of test examples.

		Predicts the output of each test pattern using the underlying model,
		and applies the (partial) inverse nonconformity function to each
		prediction, resulting in a prediction interval for each test pattern.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then intervals for
			all significance levels (0.01, 0.02, ..., 0.99) are output in a
			3d-matrix.

		Returns
		-------
		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99}
			If significance is ``None``, then p contains the interval (minimum
			and maximum boundaries) for each test pattern, and each significance
			level (0.01, 0.02, ..., 0.99). If significance is a float between
			0 and 1, then p contains the prediction intervals (minimum and
			maximum	boundaries) for the set of test patterns at the chosen
			significance level.
		"""
		prediction = self.model.predict(x)
		if significance:
			return self.err_func.apply_inverse(prediction, nc, significance)
		else:
			significance = np.arange(0.01, 1.0, 0.01)
			return np.dstack([self.err_func.apply_inverse(prediction, nc, s)
			                  for s in significance])


class NormalizedRegressorNc(RegressorNc):
	"""Nonconformity scorer using an underlying regression model together
	with a normalization model.

	Parameters
	----------
	model : RegressorAdapter
		Underlying regression model used for calculating nonconformity scores.

	normalizer_model : RegressorAdapter
		Normalizer regression model used for calculating nonconformity scores.

	err_func : RegressionErrFunc
		Scorer callable object with signature ``score(estimator, x, y)``.

	beta : float
		Parameter for normalization weighting. A larger beta results in the
		normalization model having a smaller impact on the final prediction
		interval size.

	Attributes
	----------
	model : RegressorAdapter
		Underlying model object.

	normalizer_model : RegressorAdapter
		Underlying normalizer object.

	err_func : RegressionErrFunc
		Scorer function used to calculate nonconformity scores.

	beta : float
		Normalization weight.

	See also
	--------
	RegressorNc, ProbEstClassifierNc
	"""
	def __init__(self,
	             model,
	             normalizer_model,
	             err_func=AbsErrorErrFunc(),
	             beta='auto'):
		super(NormalizedRegressorNc, self).__init__(model,
		                                            err_func)
		self.normalizer_model = normalizer_model
		self.beta = beta
		self.beta_ = None

	def fit(self, x, y):
		super(NormalizedRegressorNc, self).fit(x, y)
		err = np.abs(self.model.predict(x) - y)
		err += 0.00001  # Add a small error to each sample to avoid log(0)
		log_err = np.log(err)
		self.normalizer_model.fit(x, log_err)

	def calc_nc(self, x, y):
		norm = np.exp(self.normalizer_model.predict(x))
		prediction = self.model.predict(x)

		if self.beta == 'auto':
			r = pearsonr(norm, prediction)[0]
			if r < 0:
				self.beta_ = 0
			elif r < 0.3:
				self.beta_ = 1
			else:
				self.beta_ = 10
		else:
			self.beta_ = self.beta

		return self.err_func.apply(prediction, y, norm, self.beta_)

	def predict(self, x, nc, significance=None):
		prediction = self.model.predict(x)
		norm = np.exp(self.normalizer_model.predict(x))
		if significance:
			return self.err_func.apply_inverse(prediction,
			                                   nc,
			                                   significance,
			                                   norm,
			                                   self.beta_)
		else:
			significance = np.arange(0.01, 1.0, 0.01)
			return np.dstack([self.err_func.apply_inverse(prediction,
			                                              nc,
			                                              s,
			                                              norm,
			                                              self.beta_)
			                  for s in significance])

	def get_params(self, deep=False):
		params = super(RegressorNc, self).get_params()
		params['beta'] = self.beta
		if deep:
			params['normalizer_model'] = clone(self.normalizer_model)
		else:
			params['normalizer_model'] = self.normalizer_model
		return params
