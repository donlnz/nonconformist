#!/usr/bin/env python

"""
Inductive conformal predictors.
"""

# Authors: Henrik Linusson

from __future__ import division

import numpy as np

from nonconformist.base import RegressorMixin, ClassifierMixin

# -----------------------------------------------------------------------------
# Base inductive conformal predictor
# -----------------------------------------------------------------------------
class BaseIcp(object):
	"""Base class for inductive conformal predictors.
	"""
	def __init__(self, nc_function):
		self.cal_x, self.cal_y = None, None
		self.nc_function = nc_function

	def fit(self, x, y):
		"""Fit underlying nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the nonconformity scorer.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the nonconformity scorer.

		Returns
		-------
		None
		"""
		#TODO: incremental?
		self.nc_function.fit(x, y)

	def calibrate(self, x, y, increment=False):
		"""Calibrate conformal predictor based on underlying nonconformity
		scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for calibrating the conformal predictor.

		y : numpy array of shape [n_samples, n_features]
			Outputs of examples for calibrating the conformal predictor.

		increment : boolean
			If ``True``, performs an incremental recalibration of the conformal
			predictor. The supplied ``x`` and ``y`` are added to the set of
			previously existing calibration examples, and the conformal
			predictor is then calibrated on both the old and new calibration
			examples.

		Returns
		-------
		None
		"""
		# TODO: conditional
		self._calibrate_hook(x, y, increment)
		self._update_calibration_set(x, y, increment)
		self.cal_scores = self.nc_function.calc_nc(self.cal_x, self.cal_y)

	def _calibrate_hook(self, x, y, increment):
		pass

	def _update_calibration_set(self, x, y, increment):
		if increment and self.cal_x is not None and self.cal_y is not None:
			self.cal_x = np.vstack([self.cal_x, x])
			self.cal_y = np.hstack([self.cal_y, y])
		else:
			self.cal_x, self.cal_y = x, y

	def get_params(self, deep=False):
		return {'nc_function': self.nc_function}

# -----------------------------------------------------------------------------
# Inductive conformal classifier
# -----------------------------------------------------------------------------
class IcpClassifier(BaseIcp, ClassifierMixin):
	"""Inductive conformal classifier.

	Parameters
	----------
	nc_function : object
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``
		and ``calc_nc(x, y)``.

	smoothing : boolean
		Decides whether to use stochastic smoothing of p-values.

	Attributes
	----------
	cal_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of calibration set.

	cal_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : object
		Nonconformity scorer object used to calculate nonconformity scores.

	See also
	--------
	IcpRegressor

	References
	----------
	.. [1] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
		intervals with regression neural networks. Neural Networks, 24(8),
		842-851.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_iris
	>>> from sklearn.tree import DecisionTreeClassifier
	>>> from nonconformist.icp import IcpClassifier
	>>> from nonconformist.nc import ProbEstClassifierNc, margin
	>>> iris = load_iris()
	>>> idx = np.random.permutation(iris.target.size)
	>>> train = idx[:int(idx.size / 3)]
	>>> cal = idx[int(idx.size / 3):int(2 * idx.size / 3)]
	>>> test = idx[int(2 * idx.size / 3):]
	>>> nc = ProbEstClassifierNc(DecisionTreeClassifier, margin)
	>>> icp = IcpClassifier(nc)
	>>> icp.fit(iris.data[train, :], iris.target[train])
	>>> icp.calibrate(iris.data[cal, :], iris.target[cal])
	>>> icp.predict(iris.data[test, :], significance=0.10)
	...             # doctest: +SKIP
	array([[ True, False, False],
		[False,  True, False],
		...,
		[False,  True, False],
		[False,  True, False]], dtype=bool)
	"""
	def __init__(self, nc_function, smoothing=True):
		super(IcpClassifier, self).__init__(nc_function)
		self.classes = None
		self.last_p = None
		self.smoothing = smoothing

	def _calibrate_hook(self, x, y, increment=False):
		self._update_classes(y, increment)

	def _update_classes(self, y, increment):
		if self.classes is None or not increment:
			self.classes = np.unique(y)
		else:
			self.classes = np.unique(np.hstack([self.classes, y]))

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float or None
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then the p-values
			are output rather than the predictions.

		Returns
		-------
		p : numpy array of shape [n_samples, n_classes]
			If significance is ``None``, then p contains the p-values for each
			sample-class pair; if significance is a float between 0 and 1, then
			p is a boolean array denoting which labels are included in the
			prediction sets.
		"""
		# TODO: if x == self.last_x ...
		n_test_objects = x.shape[0]
		p = np.zeros((n_test_objects, self.classes.size))
		for i, c in enumerate(self.classes):
			test_class = np.zeros(x.shape[0])
			test_class.fill(c)

			# TODO: maybe calculate p-values using cython or similar
			# TODO: interpolated p-values

			test_nc_scores = self.nc_function.calc_nc(x, test_class)
			n_cal = self.cal_scores.size
			for j, nc in enumerate(test_nc_scores):
				n_ge = np.sum(self.cal_scores >= nc)
				p[j, i] = n_ge / (n_cal + 1)

			if self.smoothing:
				p[:, i] += np.random.uniform(0, 1, n_test_objects) / (n_cal + 1)
			else:
				p[:, i] += 1 / (n_cal + 1)

		if significance:
			return p > significance
		else:
			return p

# -----------------------------------------------------------------------------
# Inductive conformal regressor
# -----------------------------------------------------------------------------
class IcpRegressor(BaseIcp, RegressorMixin):
	"""Inductive conformal regressor.

	Parameters
	----------
	nc_function : object
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``,
		``calc_nc(x, y)`` and ``predict(x, nc_scores, significance)``.

	Attributes
	----------
	cal_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of calibration set.

	cal_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : object
		Nonconformity scorer object used to calculate nonconformity scores.

	See also
	--------
	IcpClassifier

	References
	----------
	.. [1] Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002).
		Inductive confidence machines for regression. In Machine Learning: ECML
		2002 (pp. 345-356). Springer Berlin Heidelberg.

	.. [2] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
		intervals with regression neural networks. Neural Networks, 24(8),
		842-851.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_boston
	>>> from sklearn.tree import DecisionTreeRegressor
	>>> from nonconformist.icp import IcpRegressor
	>>> from nonconformist.nc import RegressorNc, abs_error, abs_error_inv
	>>> boston = load_boston()
	>>> idx = np.random.permutation(boston.target.size)
	>>> train = idx[:int(idx.size / 3)]
	>>> cal = idx[int(idx.size / 3):int(2 * idx.size / 3)]
	>>> test = idx[int(2 * idx.size / 3):]
	>>> nc = RegressorNc(DecisionTreeRegressor, abs_error, abs_error_inv)
	>>> icp = IcpRegressor(nc)
	>>> icp.fit(boston.data[train, :], boston.target[train])
	>>> icp.calibrate(boston.data[cal, :], boston.target[cal])
	>>> icp.predict(boston.data[test, :], significance=0.10)
	...     # doctest: +SKIP
	array([[  5. ,  20.6],
		[ 15.5,  31.1],
		...,
		[ 14.2,  29.8],
		[ 11.6,  27.2]])
	"""
	def __init__(self, nc_function):
		super(IcpRegressor, self).__init__(nc_function)

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

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
		# TODO: interpolated p-values
		return self.nc_function.predict(x, self.cal_scores, significance)