#!/usr/bin/env python

"""
Conformal regression.
"""

# Authors: Henrik Linusson

from __future__ import division

import numpy as np
from sklearn.cross_validation import train_test_split

class ConformalRegressor(object):
	def __init__(self,
				 nc_function,
				 calibration_portion=0.3,
				 condition=None,
				 n_cal=None):
		self.nc_func = nc_function
		self.cal_port = calibration_portion
		self.condition = condition
		self.n_cal = n_cal

		if self.n_cal and self.cal_port:
			# should throw an error
			pass

	def fit(self, x, y):
		# TODO: one-sided CPR / signed-error CPR
		# TODO: more flexible calibration set selection (size, order, OOB... ?)

		if self.n_cal:
			self.cal_port = self.n_cal / y.size

		x_tr, x_cal, y_tr, y_cal = train_test_split(x, y, test_size=self.cal_port)
		self.n_cal = y_cal.size
		self.x_tr, self.x_cal, self.y_tr, self.y_cal = x_tr, x_cal, y_tr, y_cal
		self.nc_func.fit(self.x_tr, self.y_tr)

		if self.condition:
			self.cal_nc = {}
			for category in np.unique(self.condition(x_cal, None)):
				idx = self.condition(x_cal, None) == category
				self.cal_nc[category] = self.nc_func.calc_nc(self.x_cal[idx, :], self.y_cal[idx])
				self.cal_nc[category] = np.sort(self.cal_nc[category])[::-1]
		else:
			self.cal_nc = self.nc_func.calc_nc(x_cal, y_cal)
			self.cal_nc = np.sort(self.cal_nc)[::-1]

	def predict(self, x, significance, modified_p_val=False, interpolate=False):
		if self.condition:
			pred = np.zeros((x.shape[0], 2))
			for i in range(x.shape[0]):
				nc_scores = self.cal_nc[self.condition(x[i, :], None)]
				if nc_scores.size < (1 / significance):
					nc = nc_scores[0]
				else:
					idx_real = significance * (nc_scores.size + 1) - 1
					idx_discrete_mod = int(np.floor(nc_scores.size - (1 - significance) * nc_scores.size))
					idx_discrete_original = int(np.floor(idx_real))
					if modified_p_val or interpolate:
						idx_discrete = idx_discrete_mod
					else:
						idx_discrete = idx_discrete_original
					if idx_discrete >= nc_scores.size - 1:
						nc = nc_scores[-1]
					elif interpolate and idx_discrete_mod != idx_discrete_original:

						enum = (nc_scores[idx_discrete_original] - nc_scores[idx_discrete_mod])
						denom = (idx_discrete_original + 1) / (nc_scores.size - 1) - (idx_discrete_mod + 1) / (nc_scores.size - 1)
						mult = (idx_real + 1) / (nc_scores.size - 1) - (idx_discrete_mod + 1) / (nc_scores.size - 1)

						nc = nc_scores[idx_discrete_mod] + (enum / denom) * mult
					else:
						nc = nc_scores[idx_discrete]
				pred[i, :] = self.nc_func.predict(x[i, :], nc)
		else:
			nc_scores = self.cal_nc
			if nc_scores.size < (1 / significance):
				nc = nc_scores[0]
			else:
				idx_real = significance * (nc_scores.size + 1) - 1
				idx_discrete_mod = int(np.floor(nc_scores.size - (1 - significance) * nc_scores.size))
				idx_discrete_original = int(np.floor(idx_real))
				if modified_p_val or interpolate:
					idx_discrete = idx_discrete_mod
				else:
					idx_discrete = idx_discrete_original
				if idx_discrete >= nc_scores.size - 1:
					nc = nc_scores[-1]
				elif interpolate and idx_discrete_mod != idx_discrete_original:

					enum = (nc_scores[idx_discrete_original] - nc_scores[idx_discrete_mod])
					denom = (idx_discrete_original + 1) / (nc_scores.size - 1) - (idx_discrete_mod + 1) / (nc_scores.size - 1)
					mult = (idx_real + 1) / (nc_scores.size - 1) - (idx_discrete_mod + 1) / (nc_scores.size - 1)

					nc = nc_scores[idx_discrete_mod] + (enum / denom) * mult
				else:
					nc = nc_scores[idx_discrete]
			pred = self.nc_func.predict(x, nc)

		return pred