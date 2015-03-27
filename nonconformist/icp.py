#!/usr/bin/env python

"""
Inductive conformal predictors.
"""

# Authors: Henrik Linusson

from __future__ import division

import numpy as np

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
		"""Fit model.
		"""
		#TODO: incremental?
		self.nc_function.fit(x, y)

	def calibrate(self, x, y, increment=False):
		"""Calibrate model.
		"""
		# TODO: conditional
		self._update_calibration_set(x, y, increment)
		self.cal_scores = self.nc_function.calc_nc(self.cal_x, self.cal_y)

	def _update_calibration_set(self, x, y, increment):
		if increment and self.cal_x is not None and self.cal_y is not None:
			self.cal_x = np.vstack([self.cal_x, x])
			self.cal_y = np.hstack([self.cal_y, y])
		else:
			self.cal_x, self.cal_y = x, y

# -----------------------------------------------------------------------------
# Inductive conformal classifier
# -----------------------------------------------------------------------------
class IcpClassifier(BaseIcp):
	"""Inductive conformal classifier.

	Parameters
	----------
	nc_function:
		asd

	smoothing:
		asd

	Attributes
	----------
	cal_x:
		pass

	cal_y:
		pass

	See also
	--------
	nonconformist.regression.IcpRegressor

	References
	----------

	Examples
	--------
	"""

	@classmethod
	def get_problem_type(cls):
		return 'classification'

	def __init__(self, nc_function, smoothing=True):
		super(IcpClassifier, self).__init__(nc_function)
		self.classes = None
		self.last_p = None
		self.smoothing = smoothing

	def calibrate(self, x, y, increment=False):
		super(IcpClassifier, self).calibrate(x, y, increment)
		self._update_classes(y, increment)

	def _update_classes(self, y, increment):
		if self.classes is None or not increment:
			self.classes = np.unique(y)
		else:
			self.classes = np.unique(np.hstack([self.classes, y]))

	def predict(self, x, significance=None):
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
class IcpRegressor(BaseIcp):
	"""
	Inductive conformal regressor.
	"""

	@classmethod
	def get_problem_type(cls):
		return 'regression'

	def __init__(self, nc_function):
		super(IcpRegressor, self).__init__(nc_function)

	def predict(self, x, significance):
		# TODO: interpolated p-values
		return self.nc_function.predict(x, self.cal_scores, significance)