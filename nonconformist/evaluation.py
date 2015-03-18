#!/usr/bin/env python

"""
docstring
"""

# Authors: Henrik Linusson

# TODO: should possibly (re)implement a cross-validator suited for CP

from __future__ import division

import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit

class IcpCrossValidationHelper(object):
	def __init__(self, icp, calibration_portion=0.25, significance=0.1):
		self.icp = icp
		self.cal_port = calibration_portion
		self.significance = significance

	def fit(self, x, y):
		split = StratifiedShuffleSplit(y, n_iter=1, test_size=self.cal_port)
		for train, cal in split:
			self.icp.fit(x[train, :], y[train])
			self.icp.calibrate(x[cal, :], y[cal])

	def predict(self, x, significance=True):
		if significance:
			return self.icp.predict(x, significance=self.significance)
		else:
			return self.icp.predict(x, significance=None)

	def get_params(self, deep=False):
		return {'icp': self.icp,
		        'calibration_portion': self.cal_port,
		        'significance': self.significance}

# -----------------------------------------------------------------------------
# Validity measures
# -----------------------------------------------------------------------------
def reg_n_correct(model, x, y):
	prediction = model.predict(x)
	low = y >= prediction[:,0]
	high = y <= prediction[:,1]
	correct = low * high

	return y[correct].size

def reg_mean_errors(model, x, y):
	return 1 - reg_n_correct(model, x, y) / y.size

def class_n_correct(model, x, y):
	prediction = model.predict(x)
	correct = np.zeros((y.size,), dtype=bool)
	for i, y_ in enumerate(y):
		correct[i] = prediction[i, y_]
	return np.sum(correct)

def class_mean_errors(model, x, y):
	return 1 - (class_n_correct(model, x, y) / y.size)

# -----------------------------------------------------------------------------
# Efficiency measures
# -----------------------------------------------------------------------------
def reg_mean_size(x, y, model):
	prediction = model.predict(x)
	interval_size = 0
	for j in range(y.size):
		interval_size += np.abs(prediction[j, 1] - prediction[j, 0])
	return interval_size / y.size

def class_avg_c(model, x, y):
	prediction = model.predict(x)
	return np.sum(prediction) / prediction.shape[0]

def class_mean_p_val(model, x, y):
	prediction = model.predict(x, significance=False)
	return np.mean(prediction)

def one_c(model, x, y):
	prediction = model.predict(x)
	return np.sum(1 for _ in filter(lambda x: np.sum(x) == 1, prediction))