#!/usr/bin/env python

"""
Evaluation of conformal predictors.
"""

# Authors: Henrik Linusson

# TODO: cross_val_score/run_experiment should possibly allow multiple to be evaluated on identical folding

from __future__ import division

from nonconformist.base import RegressorMixin, ClassifierMixin

import sys
import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.base import clone, BaseEstimator


class BaseIcpCvHelper(BaseEstimator):
	"""Base class for cross validation helpers.
	"""
	def __init__(self, icp, calibration_portion):
		super(BaseIcpCvHelper, self).__init__()
		self.icp = icp
		self.calibration_portion = calibration_portion

	def predict(self, x, significance=None):
			return self.icp.predict(x, significance)


class ClassIcpCvHelper(BaseIcpCvHelper, ClassifierMixin):
	"""Helper class for running the ``cross_val_score`` evaluation
	method on IcpClassifiers.

	See also
	--------
	IcpRegCrossValHelper

	Examples
	--------
	>>> from sklearn.datasets import load_iris
	>>> from sklearn.ensemble import RandomForestClassifier
	>>> from nonconformist.icp import IcpClassifier
	>>> from nonconformist.nc import ClassifierNc, MarginErrFunc
	>>> from nonconformist.evaluation import ClassIcpCvHelper
	>>> from nonconformist.evaluation import class_mean_errors
	>>> from nonconformist.evaluation import cross_val_score
	>>> data = load_iris()
	>>> nc = ProbEstClassifierNc(RandomForestClassifier(), MarginErrFunc())
	>>> icp = IcpClassifier(nc)
	>>> icp_cv = ClassIcpCvHelper(icp)
	>>> cross_val_score(icp_cv,
	...                 data.data,
	...                 data.target,
	...                 iterations=2,
	...                 folds=2,
	...                 scoring_funcs=[class_mean_errors],
	...                 significance_levels=[0.1])
	...     # doctest: +SKIP
	   class_mean_errors  fold  iter  significance
	0           0.013333     0     0           0.1
	1           0.080000     1     0           0.1
	2           0.053333     0     1           0.1
	3           0.080000     1     1           0.1
	"""
	def __init__(self, icp, calibration_portion=0.25):
		super(ClassIcpCvHelper, self).__init__(icp, calibration_portion)

	def fit(self, x, y):
		split = StratifiedShuffleSplit(y, n_iter=1,
		                               test_size=self.calibration_portion)
		for train, cal in split:
			self.icp.fit(x[train, :], y[train])
			self.icp.calibrate(x[cal, :], y[cal])


class RegIcpCvHelper(BaseIcpCvHelper, RegressorMixin):
	"""Helper class for running the ``cross_val_score`` evaluation
	method on IcpRegressors.

	See also
	--------
	IcpClassCrossValHelper

	Examples
	--------
	>>> from sklearn.datasets import load_boston
	>>> from sklearn.ensemble import RandomForestRegressor
	>>> from nonconformist.icp import IcpRegressor
	>>> from nonconformist.nc import RegressorNc, AbsErrorErrFunc
	>>> from nonconformist.evaluation import RegIcpCvHelper
	>>> from nonconformist.evaluation import reg_mean_errors
	>>> from nonconformist.evaluation import cross_val_score
	>>> data = load_boston()
	>>> nc = RegressorNc(RandomForestRegressor(), AbsErrorErrFunc())
	>>> icp = IcpRegressor(nc)
	>>> icp_cv = RegIcpCvHelper(icp)
	>>> cross_val_score(icp_cv,
	...                 data.data,
	...                 data.target,
	...                 iterations=2,
	...                 folds=2,
	...                 scoring_funcs=[reg_mean_errors],
	...                 significance_levels=[0.1])
	...     # doctest: +SKIP
	   fold  iter  reg_mean_errors  significance
	0     0     0         0.185771           0.1
	1     1     0         0.138340           0.1
	2     0     1         0.071146           0.1
	3     1     1         0.043478           0.1
	"""
	def __init__(self, icp, calibration_portion=0.25):
		super(RegIcpCvHelper, self).__init__(icp, calibration_portion)

	def fit(self, x, y):
		split = train_test_split(x, y, test_size=self.calibration_portion)
		x_tr, x_cal, y_tr, y_cal = split[0], split[1], split[2], split[3]
		self.icp.fit(x_tr, y_tr)
		self.icp.calibrate(x_cal, y_cal)


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
def cross_val_score(model,x, y, iterations=10, folds=10, fit_params=None,
					scoring_funcs=None, significance_levels=None,
					verbose=False):
	"""Evaluates a conformal predictor using cross-validation.

	Parameters
	----------
	model : object
		Conformal predictor to evaluate.

	x : numpy array of shape [n_samples, n_features]
		Inputs of data to use for evaluation.

	y : numpy array of shape [n_samples]
		Outputs of data to use for evaluation.

	iterations : int
		Number of iterations to use for evaluation. The data set is randomly
		shuffled before each iteration.

	folds : int
		Number of folds to use for evaluation.

	fit_params : dictionary
		Parameters to supply to the conformal prediction object on training.

	scoring_funcs : iterable
		List of evaluation functions to apply to the conformal predictor in each
		fold. Each evaluation function should have a signature
		``scorer(prediction, y, significance)``.

	significance_levels : iterable
		List of significance levels at which to evaluate the conformal
		predictor.

	verbose : boolean
		Indicates whether to output progress information during evaluation.

	Returns
	-------
	scores : pandas DataFrame
		Tabulated results for each iteration, fold and evaluation function.
	"""

	fit_params = fit_params if fit_params else {}
	significance_levels = (significance_levels if significance_levels
	                       is not None else np.arange(0.01, 1.0, 0.01))

	df = pd.DataFrame()

	columns = ['iter',
			   'fold',
			   'significance',
			   ] + [f.__name__ for f in scoring_funcs]
	for i in range(iterations):
		idx = np.random.permutation(y.size)
		x, y = x[idx, :], y[idx]
		cv = KFold(y.size, folds)
		for j, (train, test) in enumerate(cv):
			if verbose:
				sys.stdout.write('\riter {}/{} fold {}/{}'.format(
					i + 1,
					iterations,
					j + 1,
					folds
				))
			m = clone(model)
			m.fit(x[train, :], y[train], **fit_params)
			prediction = m.predict(x[test, :], significance=None)
			for k, s in enumerate(significance_levels):
				scores = [scoring_func(prediction, y[test], s)
						  for scoring_func in scoring_funcs]
				df_score = pd.DataFrame([[i, j, s] + scores],
											columns=columns)
				df = df.append(df_score, ignore_index=True)

	return df


def run_experiment(models, csv_files, iterations=10, folds=10, fit_params=None,
				   scoring_funcs=None, significance_levels=None,
				   normalize=False, verbose=False, header=0):
	"""Performs a cross-validation evaluation of one or several conformal
	predictors on a	collection of data sets in csv format.

	Parameters
	----------
	models : object or iterable
		Conformal predictor(s) to evaluate.

	csv_files : iterable
		List of file names (with absolute paths) containing csv-data, used to
		evaluate the conformal predictor.

	iterations : int
		Number of iterations to use for evaluation. The data set is randomly
		shuffled before each iteration.

	folds : int
		Number of folds to use for evaluation.

	fit_params : dictionary
		Parameters to supply to the conformal prediction object on training.

	scoring_funcs : iterable
		List of evaluation functions to apply to the conformal predictor in each
		fold. Each evaluation function should have a signature
		``scorer(prediction, y, significance)``.

	significance_levels : iterable
		List of significance levels at which to evaluate the conformal
		predictor.

	verbose : boolean
		Indicates whether to output progress information during evaluation.

	Returns
	-------
	scores : pandas DataFrame
		Tabulated results for each data set, iteration, fold and
		evaluation function.
	"""
	df = pd.DataFrame()
	if not hasattr(models, '__iter__'):
		models = [models]

	for model in models:
		is_regression = model.get_problem_type() == 'regression'

		n_data_sets = len(csv_files)
		for i, csv_file in enumerate(csv_files):
			if verbose:
				print('\n{} ({} / {})'.format(csv_file, i + 1, n_data_sets))
			data = pd.read_csv(csv_file, header=header)
			x, y = data.values[:, :-1], data.values[:, -1]
			x = np.array(x, dtype=np.float64)
			if normalize:
				if is_regression:
					y = y - y.min() / (y.max() - y.min())
				else:
					for j, y_ in enumerate(np.unique(y)):
						y[y == y_] = j

			scores = cross_val_score(model, x, y, iterations, folds,
			                         fit_params, scoring_funcs,
			                         significance_levels, verbose)

			ds_df = pd.DataFrame(scores)
			ds_df['model'] = model.__class__.__name__
			try:
				ds_df['data_set'] = csv_file.split('/')[-1]
			except:
				ds_df['data_set'] = csv_file

			df = df.append(ds_df)

	return df


# -----------------------------------------------------------------------------
# Validity measures
# -----------------------------------------------------------------------------
def reg_n_correct(prediction, y, significance=None):
	"""Calculates the number of correct predictions made by a conformal
	regression model.
	"""
	if significance is not None:
		idx = int(significance * 100 - 1)
		prediction = prediction[:, :, idx]

	low = y >= prediction[:, 0]
	high = y <= prediction[:, 1]
	correct = low * high

	return y[correct].size


def reg_mean_errors(prediction, y, significance):
	"""Calculates the average error rate of a conformal regression model.
	"""
	return 1 - reg_n_correct(prediction, y, significance) / y.size


def class_n_correct(prediction, y, significance):
	"""Calculates the number of correct predictions made by a conformal
	classification model.
	"""
	labels, y = np.unique(y, return_inverse=True)
	prediction = prediction > significance
	correct = np.zeros((y.size,), dtype=bool)
	for i, y_ in enumerate(y):
		correct[i] = prediction[i, int(y_)]
	return np.sum(correct)


def class_mean_errors(prediction, y, significance=None):
	"""Calculates the average error rate of a conformal classification model.
	"""
	return 1 - (class_n_correct(prediction, y, significance) / y.size)


def class_one_err(prediction, y, significance=None):
	"""Calculates the error rate of conformal classifier predictions containing
	 only a single output label.
	"""
	labels, y = np.unique(y, return_inverse=True)
	prediction = prediction > significance
	idx = np.arange(0, y.size, 1)
	idx = filter(lambda x: np.sum(prediction[x, :]) == 1, idx)
	errors = filter(lambda x: not prediction[x, int(y[x])], idx)

	if len(idx) > 0:
		return np.size(errors) / np.size(idx)
	else:
		return 0


def class_mean_errors_one_class(prediction, y, significance, c=0):
	"""Calculates the average error rate of a conformal classification model,
	  considering only test examples belonging to class ``c``. Use
	  ``functools.partial`` in order to test other classes.
	"""
	labels, y = np.unique(y, return_inverse=True)
	prediction = prediction > significance
	idx = np.arange(0, y.size, 1)[y == c]
	errs = np.sum(1 for _ in filter(lambda x: not prediction[x, c], idx))

	if idx.size > 0:
		return errs / idx.size
	else:
		return 0


def class_one_err_one_class(prediction, y, significance, c=0):
	"""Calculates the error rate of conformal classifier predictions containing
	 only a single output label. Considers only test examples belonging to
	 class ``c``. Use ``functools.partial`` in order to test other classes.
	"""
	labels, y = np.unique(y, return_inverse=True)
	prediction = prediction > significance
	idx = np.arange(0, y.size, 1)
	idx = filter(lambda x: prediction[x, c], idx)
	idx = filter(lambda x: np.sum(prediction[x, :]) == 1, idx)
	errors = filter(lambda x: int(y[x]) != c, idx)

	if len(idx) > 0:
		return np.size(errors) / np.size(idx)
	else:
		return 0


# -----------------------------------------------------------------------------
# Efficiency measures
# -----------------------------------------------------------------------------
def _reg_interval_size(prediction, y, significance):
	idx = int(significance * 100 - 1)
	prediction = prediction[:, :, idx]

	return prediction[:, 1] - prediction[:, 0]


def reg_min_size(prediction, y, significance):
	return np.min(_reg_interval_size(prediction, y, significance))


def reg_q1_size(prediction, y, significance):
	return np.percentile(_reg_interval_size(prediction, y, significance), 25)


def reg_median_size(prediction, y, significance):
	return np.median(_reg_interval_size(prediction, y, significance))


def reg_q3_size(prediction, y, significance):
	return np.percentile(_reg_interval_size(prediction, y, significance), 75)


def reg_max_size(prediction, y, significance):
	return np.max(_reg_interval_size(prediction, y, significance))


def reg_mean_size(prediction, y, significance):
	"""Calculates the average prediction interval size of a conformal
	regression model.
	"""
	return np.mean(_reg_interval_size(prediction, y, significance))


def class_avg_c(prediction, y, significance):
	"""Calculates the average number of classes per prediction of a conformal
	classification model.
	"""
	prediction = prediction > significance
	return np.sum(prediction) / prediction.shape[0]


def class_mean_p_val(prediction, y, significance):
	"""Calculates the mean of the p-values output by a conformal classification
	model.
	"""
	return np.mean(prediction)


def class_one_c(prediction, y, significance):
	"""Calculates the rate of singleton predictions (prediction sets containing
	only a single class label) of a conformal classification model.
	"""
	prediction = prediction > significance
	n_singletons = np.sum(1 for _ in filter(lambda x: np.sum(x) == 1,
	                                        prediction))
	return n_singletons / y.size


def class_empty(prediction, y, significance):
	"""Calculates the rate of singleton predictions (prediction sets containing
	only a single class label) of a conformal classification model.
	"""
	prediction = prediction > significance
	n_empty = np.sum(1 for _ in filter(lambda x: np.sum(x) == 0,
	                                        prediction))
	return n_empty / y.size


def n_test(prediction, y, significance):
	"""Provides the number of test patters used in the evaluation.
	"""
	return y.size