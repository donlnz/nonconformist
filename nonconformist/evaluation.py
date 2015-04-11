#!/usr/bin/env python

"""
Evaluation of conformal predictors.
"""

# Authors: Henrik Linusson

# TODO: should possibly (re)implement a cross-validator suited for CP

from __future__ import division

import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split

class IcpClassCrossValHelper(object):
	"""Helper class for running sklearn's ``cross_val_score`` evaluation
	method on IcpClassifiers.

	See also
	--------
	IcpRegCrossValHelper

	Examples
	--------
	>>> from sklearn.datasets import load_iris
	>>> from sklearn.ensemble import RandomForestClassifier
	>>> from sklearn.cross_validation import cross_val_score
	>>> from nonconformist.icp import IcpClassifier
	>>> from nonconformist.nc import ProbEstClassifierNc, margin
	>>> from nonconformist.evaluation import IcpClassCrossValHelper
	>>> from nonconformist.evaluation import class_mean_errors
	>>> data = load_iris()
	>>> nc = ProbEstClassifierNc(RandomForestClassifier, margin)
	>>> icp = IcpClassifier(nc)
	>>> icp_cv = IcpClassCrossValHelper(icp, significance=0.1)
	>>> cross_val_score(icp_cv,
	...                 data.data,
	...                 data.target,
	...                 scoring=class_mean_errors,
	...                 cv=10)
	...     # doctest: +SKIP
	array([ 0.        ,  0.        ,  0.        ,  0.06666667,  0.33333333,
	0.2       ,  0.        ,  0.13333333,  0.53333333,  0.06666667])
	"""
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

class IcpRegCrossValHelper(object):
	"""Helper class for running sklearn's ``cross_val_score`` evaluation
	method on IcpRegressors.

	See also
	--------
	IcpClassCrossValHelper

	Examples
	--------
	>>> from sklearn.datasets import load_boston
	>>> from sklearn.ensemble import RandomForestRegressor
	>>> from sklearn.cross_validation import cross_val_score
	>>> from nonconformist.icp import IcpRegressor
	>>> from nonconformist.nc import RegressorNc, abs_error, abs_error_inv
	>>> from nonconformist.evaluation import IcpRegCrossValHelper
	>>> from nonconformist.evaluation import reg_mean_errors
	>>> data = load_boston()
	>>> nc = RegressorNc(RandomForestRegressor, abs_error, abs_error_inv)
	>>> icp = IcpRegressor(nc)
	>>> icp_cv = IcpRegCrossValHelper(icp, significance=0.1)
	>>> cross_val_score(icp_cv,
	...                 data.data,
	...                 data.target,
	...                 scoring=reg_mean_errors,
	...                 cv=10)
	...     # doctest: +SKIP
	array([ 0.17647059,  0.03921569,  0.        ,  0.23529412,  0.11764706,
	0.17647059,  0.2       ,  0.36      ,  0.2       ,  0.16      ])
	"""
	def __init__(self, icp, calibration_portion=0.25, significance=0.1):
		self.icp = icp
		self.cal_port = calibration_portion
		self.significance = significance

	def fit(self, x, y):
		split = train_test_split(x, y, test_size=self.cal_port)
		x_tr, x_cal, y_tr, y_cal = split[0], split[1], split[2], split[3]
		self.icp.fit(x_tr, y_tr)
		self.icp.calibrate(x_cal, y_cal)

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
def reg_n_correct(model, x, y, significance=None):
	"""Calculates the number of correct predictions made by a conformal
	regression model.

	Parameters
	----------
	model : object
		Conformal regressor object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	if significance:
		prediction = model.predict(x, significance)
	else:
		prediction = model.predict(x)
	low = y >= prediction[:,0]
	high = y <= prediction[:,1]
	correct = low * high

	return y[correct].size

def reg_mean_errors(model, x, y, significance=None):
	"""Calculates the average error rate of a conformal regression model.

	Parameters
	----------
	model : object
		Conformal regressor object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	return 1 - reg_n_correct(model, x, y, significance) / y.size

def class_n_correct(model, x, y, significance=None):
	"""Calculates the number of correct predictions made by a conformal
	classification model.

	Parameters
	----------
	model : object
		Conformal classifier object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	if significance:
		prediction = model.predict(x, significance)
	else:
		prediction = model.predict(x)
	correct = np.zeros((y.size,), dtype=bool)
	for i, y_ in enumerate(y):
		correct[i] = prediction[i, y_]
	return np.sum(correct)

def class_mean_errors(model, x, y, significance=None):
	"""Calculates the average error rate of a conformal classification model.

	Parameters
	----------
	model : object
		Conformal classifier object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	return 1 - (class_n_correct(model, x, y, significance) / y.size)

# -----------------------------------------------------------------------------
# Efficiency measures
# -----------------------------------------------------------------------------
def reg_mean_size(model, x, y, significance=None):
	"""Calculates the average prediction interval size of a conformal
	regression model.

	Parameters
	----------
	model : object
		Conformal regressor object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	if significance:
		prediction = model.predict(x)
	else:
		prediction = model.predict(x, significance)
	interval_size = 0
	for j in range(y.size):
		interval_size += np.abs(prediction[j, 1] - prediction[j, 0])
	return interval_size / y.size

def class_avg_c(model, x, y, significance=None):
	"""Calculates the average number of classes per prediction of a conformal
	classification model.

	Parameters
	----------
	model : object
		Conformal classifier object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	if significance:
		prediction = model.predict(x, significance)
	else:
		prediction = model.predict(x)
	return np.sum(prediction) / prediction.shape[0]

def class_mean_p_val(model, x, y):
	"""Calculates the mean of the p-values output by a conformal classification
	model.

	Parameters
	----------
	model : object
		Conformal classifier object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	prediction = model.predict(x, significance=None)
	return np.mean(prediction)

def class_one_c(model, x, y, significance=None):
	"""Calculates the rate of singleton predictions (prediction sets containing
	only a single class label) of a conformal classification model.

	Parameters
	----------
	model : object
		Conformal classifier object.

	x : numpy array of shape [n_samples, n_features]
		Inputs of test objects.

	y : numpy array of shape [n_samples]
		Outputs of test objects.
	"""
	if significance:
		prediction = model.predict(x, significance)
	else:
		prediction = model.predict(x)
	return np.sum(1 for _ in filter(lambda x: np.sum(x) == 1, prediction))