#!/usr/bin/env python

"""
Conformal regression.
"""

# Authors: Henrik Linusson

from __future__ import division

import numpy as np

# TODO: normalized nonconformity scores

# -----------------------------------------------------------------------------
# Error functions
# -----------------------------------------------------------------------------

def absolute_error(prediction, y):
	return np.abs(prediction - y)

def absolute_error_inverse(prediction, nc, significance):
	nc = np.sort(nc)[::-1]
	border = int(np.floor(significance * (nc.size + 1))) - 1
	# TODO: should probably warn against too few calibration examples
	border = max(border, 0)
	return np.vstack([prediction - nc[border], prediction + nc[border]]).T

def signed_error(prediction, y):
	return prediction - y

def signed_error_inverse(prediction, nc, significance):
	nc = np.sort(nc)[::-1]
	upper = int(np.floor((significance / 2) * (nc.size + 1)))
	lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
	# TODO: should probably warn against too few calibration examples
	upper = max(upper, 0)
	lower = min(lower, nc.size - 1)
	return np.vstack([prediction + nc[lower], prediction + nc[upper]]).T


# -----------------------------------------------------------------------------
# Nonconformity functions
# -----------------------------------------------------------------------------
class RegressorNc(object):
	'''
	Nonconformity function based on a simple regression model.
	'''
	def __init__(self,
	             model_class,
	             err_func,
	             inverse_err_func,
	             model_params=None):
		self.last_x, self.last_y = None, None
		self.last_prediction = None
		self.clean = False
		self.err_func = err_func
		self.inverse_err_func = inverse_err_func

		self.model_class = model_class
		self.model_params = model_params if model_params else {}

		self.model = self.model_class(**self.model_params)

	def fit(self, x, y, increment=False):
		# TODO: incremental
		self.model.fit(x, y)
		self.clean = False

	def underlying_predict(self, x):
		if (not self.clean or
			self.last_x is None or
		    not np.array_equal(self.last_x, x)):

			self.last_x = x
			self.last_prediction = self.model.predict(x)
			self.clean = True

		return self.last_prediction.copy()

	def calc_nc(self, x, y):
		prediction = self.underlying_predict(x)
		return self.err_func(prediction, y)

	def predict(self, x, nc, significance):
		prediction = self.underlying_predict(x)
		return self.inverse_err_func(prediction, nc, significance)

# -----------------------------------------------------------------------------
# Conformal predictors
# -----------------------------------------------------------------------------
class IcpRegressor(object):
	'''
	Inductive conformal regressor.
	'''
	def __init__(self, nc_function):
		self.cal_x, self.cal_y = None, None
		self.nc_function = nc_function

	def fit(self, x, y, increment=False):
		self.nc_function.fit(x, y, increment)

	def calibrate(self, x, y, condition=None, increment=False):
		# TODO: conditional
		if increment and self.cal_x is not None and self.cal_y is not None:
			self.cal_x = np.vstack([self.cal_x, x])
			self.cal_x = np.hstack([self.cal_y, y])
		else:
			self.cal_x, self.cal_y = x, y

		self.cal_scores = self.nc_function.calc_nc(self.cal_x, self.cal_y)

	def predict(self, x, significance):
		return self.nc_function.predict(x, self.cal_scores, significance)