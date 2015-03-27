#!/usr/bin/env python

"""
Base classes for conformal predictors.
"""

# Authors: Henrik Linusson

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