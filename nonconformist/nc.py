#!/usr/bin/env python

"""
Nonconformity functions.
"""

# Authors: Henrik Linusson

from __future__ import division

import numpy as np
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# Classification error functions
# -----------------------------------------------------------------------------
def inverse_probability(prediction, y):
	"""Calculates the probability of not predicting the correct class.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		1 - \hat{P}(y_i | x) \, .

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
	prob = np.zeros(y.size, dtype=np.float32)
	for i, y_ in enumerate(y):
		prob[i] = prediction[i, int(y[i])]
	return 1 - prob

def margin(prediction, y):
	"""Calculates the margin error.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		0.5 - \hat{P}(y_i | x) - max_{y \, != \, y_i} \hat{P}(y | x) \, .

	Parameters
	----------
	prediction : numpy array of shape [n_samples, n_classes]
		Class probability estimates for each sample.

	y : numpy array of shape [n_samples]
		True output labels of each sample.

	Returns
	-------
	nc : numpy array of shape [n_samples, n_classes]
		Nonconformity scores for each sample and each class.
	"""
	prob = np.zeros(y.size, dtype=np.float32)
	for i, y_ in enumerate(y):
		prob[i] = prediction[i, int(y[i])]
		prediction[i, int(y[i])] = -np.inf
	return 0.5 - ((prob - prediction.max(axis=1)) / 2)

# -----------------------------------------------------------------------------
# Regression error functions
# -----------------------------------------------------------------------------
def abs_error(prediction, y, norm=None, beta=0):
	"""Calculates absolute error nonconformity for regression problems.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		| y_i - \hat{y}_i |

	Parameters
	----------
	prediction : numpy array of shape [n_samples]
		Regression prediction for each sample.

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
		Nonconformity score of each sample.
	"""
	norm = np.array([1] * prediction.shape[0]) if norm is None else norm
	beta = 0 if beta is None else beta
	return np.abs(prediction - y) / (norm + beta)

def abs_error_inv(prediction, nc, significance, norm=None, beta=0):
	"""Calculates a prediction from an absolute-error nonconformity function.

	Calculates the partial inverse of the ``absolute_error`` nonconformity
	function, i.e., the minimum and maximum boundaries of the prediction
	interval for each point prediction, given a significance level.

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
	norm = np.array([1] * prediction.shape[0]) if norm is None else norm
	beta = 0 if beta is None else beta

	nc = np.sort(nc)[::-1]
	border = int(np.floor(significance * (nc.size + 1))) - 1
	# TODO: should probably warn against too few calibration examples
	border = min(max(border, 0), nc.size - 1)
	return np.vstack([prediction - (nc[border] * (norm + beta)),
	                  prediction + (nc[border] * (norm + beta))]).T

def sign_error(prediction, y, norm=None, beta=None):
	"""Calculates signed error nonconformity for regression problems.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		y_i - \hat{y}_i

	Parameters
	----------
	prediction : numpy array of shape [n_samples]
		Regression prediction for each sample.

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
		Nonconformity score of each sample.
	"""
	norm = np.array([1] * prediction.shape[0]) if norm is None else norm
	beta = 0 if beta is None else beta

	return (prediction - y) / (norm + beta)

def sign_error_inv(prediction, nc, significance, norm=None, beta=None):
	"""Calculates a prediction from a signed-error nonconformity function.

	Calculates the partial inverse of the ``signed_error`` nonconformity
	function, i.e., the minimum and maximum boundaries of the prediction
	interval for each point prediction, given a significance level.

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
	norm = np.array([1] * prediction.shape[0]) if norm is None else norm
	beta = 0 if beta is None else beta

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
class BaseNc(object):
	"""Base class for nonconformity scorers based on an underlying model.
	"""
	def __init__(self, model_class, err_func, model_params=None):
		self.err_func = err_func
		self.model_class = model_class
		self.model_params = model_params if model_params else {}
		self.model = self.model_class(**self.model_params)

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

	def __get_prediction(self, x):
		if (not self.clean or
			self.last_x is None or
		    not np.array_equal(self.last_x, x)):

			self.last_x = x
			self.last_prediction = self._underlying_predict(x)
			self.clean = True

		return self.last_prediction.copy()

	def calc_nc(self, x, y):
		"""Calculates the nonconformity score of a set of samples.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for which to calculate a nonconformity score.

		y : numpy array of shape [n_samples, n_features]
			Outputs of examples for which to calculate a nonconformity score.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of samples.
		"""
		prediction = self.__get_prediction(x)
		return self.err_func(prediction, y)

# -----------------------------------------------------------------------------
# Classification nonconformity scorers
# -----------------------------------------------------------------------------
class ProbEstClassifierNc(BaseNc):
	"""Nonconformity scorer using an underlying class probability estimating
	model.

	Parameters
	----------
	model_class : class
		The model_class should be implement the ``fit(x, y)`` and
		``predict_proba(x)`` methods, as used by the classification models
		present in the scikit-learn library.

	err_func : callable
		Scorer callable object with signature ``score(estimator, x, y)``.

	model_params : dict, optional
		Dict containing keyword parameters to pass to model_class upon
		initialization.

	Attributes
	----------
	model_class : class
		Class used to construct the underlying model.

	err_func : callable
		Scorer function used to calculate nonconformity scores.

	model_params : dict
		Parameters sent to underlying model.

	model : object
		Underlying model object.

	See also
	--------
	RegressorNc
	"""
	def __init__(self,
	             model_class,
	             err_func=margin,
	             model_params=None):
		super(ProbEstClassifierNc, self).__init__(model_class,
		                                          err_func,
		                                          model_params)

	def _underlying_predict(self, x):
		return self.model.predict_proba(x)

# -----------------------------------------------------------------------------
# Regression nonconformity scorers
# -----------------------------------------------------------------------------
class RegressorNc(BaseNc):
	"""Nonconformity scorer using an underlying regression model.

	Parameters
	----------
	model_class : class
		The model_class should be implement the ``fit(x, y)`` and
		``predict(x)`` methods, as used by the regression models
		present in the scikit-learn library.

	err_func : callable
		Scorer callable object with signature ``score(estimator, x, y)``.

	inverse_error_func : callable
		Inverse (or partial inverse) of err_func.

	model_params : dict, optional
		Dict containing keyword parameters to pass to model_class upon
		initialization.

	Attributes
	----------
	model_class : class
		Class used to construct the underlying model.

	err_func : callable
		Scorer function used to calculate nonconformity scores.

	inverse_err_func : callable
		Inverse function (partial) of nonconformity function.

	model_params : dict
		Parameters sent to underlying model.

	model : object
		Underlying model object.

	See also
	--------
	ProbEstClassifierNc
	"""
	def __init__(self,
	             model_class,
	             err_func=abs_error,
	             inverse_err_func=abs_error_inv,
	             model_params=None):
		super(RegressorNc, self).__init__(model_class,
		                                  err_func,
		                                  model_params)

		self.inverse_err_func = inverse_err_func

	def _underlying_predict(self, x):
		return self.model.predict(x)

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
		prediction = self._underlying_predict(x)
		if significance:
			return self.inverse_err_func(prediction, nc, significance)
		else:
			significance = np.arange(0.01, 1.0, 0.01)
			return np.dstack([self.inverse_err_func(prediction, nc, s)
			                  for s in significance])

class NormalizedRegressorNc(RegressorNc):
	def __init__(self,
	             model_class,
	             normalizer_class,
	             err_func=abs_error,
	             inverse_err_func=abs_error_inv,
	             model_params=None,
	             normalizer_params=None,
	             beta='auto'):
		super(NormalizedRegressorNc, self).__init__(model_class,
		                                            err_func,
		                                            inverse_err_func,
		                                            model_params)
		self.normalizer_params = normalizer_params if normalizer_params else {}
		self.normalizer_class = normalizer_class
		self.normalizer = self.normalizer_class(**self.normalizer_params)
		self.beta = beta
		self.beta_ = None

	def fit(self, x, y):
		super(NormalizedRegressorNc, self).fit(x, y)
		err = np.abs(self._underlying_predict(x) - y)
		err += 0.00001 # Add a small error to each sample to avoid log(0)
		log_err = np.log(err)
		self.normalizer.fit(x, log_err)

	def calc_nc(self, x, y):
		norm = np.exp(self.normalizer.predict(x))
		prediction = self._underlying_predict(x)

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

		return self.err_func(prediction, y, norm, self.beta_)

	def predict(self, x, nc, significance=None):
		prediction = self._underlying_predict(x)
		norm = np.exp(self.normalizer.predict(x))
		if significance:
			return self.inverse_err_func(prediction,
			                             nc,
			                             significance,
			                             norm,
			                             self.beta_)
		else:
			significance = np.arange(0.01, 1.0, 0.01)
			return np.dstack([self.inverse_err_func(prediction,
			                                        nc,
			                                        s,
			                                        norm,
			                                        self.beta_)
			                  for s in significance])
