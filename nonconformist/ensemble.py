#!/usr/bin/env python

"""
Classes for constructing combined conformal predictors for classification
or regression.
"""

# Authors: Henrik Linusson

import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit

class AggregatedCp(object):
	def __init__(self,
	             cp_class,
	             nc_class,
	             aggregation_func,
	             nc_class_params,
	             n_models):
		self.p_agg_func = aggregation_func
		self.predictors = []
		self.n_models = n_models
		self.cp_class = cp_class
		self.nc_class = nc_class
		self.nc_class_params = nc_class_params

	def _generate_samples(self, x, y):
		yield None

	def fit(self, x, y):
		self.predictors = []
		idx = np.random.permutation(y.size)
		x, y = x[idx, :], y[idx]
		for train, cal in self._generate_samples(x, y):
			predictor = self.cp_class(self.nc_class(**self.nc_class_params))
			predictor.fit(x[train, :], y[train])
			predictor.calibrate(x[cal, :], y[cal])
			self.predictors.append(predictor)

	def predict(self, x, significance=None):
		if self.cp_class.problem_type == 'classification':
			p_values = None
			for predictor in self.predictors:
				if p_values is None:
					p_values = predictor.predict(x)
				else:
					p_values = np.dstack([p_values, predictor.predict(x)])

			p_values = self.p_agg_func(p_values)

			if significance:
				return p_values >= significance
			else:
				return p_values

		if self.cp_class.problem_type == 'regression':
			intervals = None
			for predictor in self.predictors:
				if intervals is None:
					intervals = predictor.predict(x, significance)
				else:
					intervals = np.dstack([intervals,
					                       predictor.predict(x, significance)])

			p_values = self.p_agg_func(intervals)

			return p_values

class BootstrapCp(AggregatedCp):
	def __init__(self,
	             cp_class,
	             nc_class,
	             aggregation_func=lambda x: np.mean(x, axis=2),
	             nc_class_params=None,
	             n_models=10):
		super(BootstrapCp, self).__init__(cp_class,
                                          nc_class,
                                          aggregation_func,
                                          nc_class_params,
                                          n_models)
	def _generate_samples(self, x, y):
		for i in range(self.n_models):
			idx = np.array(range(y.size))
			train = np.random.choice(y.size, y.size, replace=True)
			cal_mask = np.array(np.ones(idx.size), dtype=bool)
			for j in train:
				cal_mask[j] = False
			cal = idx[cal_mask]

			yield train, cal

class CrossCp(AggregatedCp):
	def __init__(self,
	             cp_class,
	             nc_class,
	             aggregation_func=lambda x: np.mean(x, axis=2),
	             nc_class_params=None,
	             n_models=10):
		super(CrossCp, self).__init__(cp_class,
                                      nc_class,
                                      aggregation_func,
                                      nc_class_params,
                                      n_models)

	def _generate_samples(self, x, y):
		if self.cp_class.problem_type == 'classification':
			folds = StratifiedKFold(y, n_folds=self.n_models)
		else:
			folds = KFold(y.size, n_folds=self.n_models)
		for train, cal in folds:
			yield train, cal

class SubSamplingCp(AggregatedCp):
	def __init__(self,
	             cp_class,
	             nc_class,
	             calibration_portion=0.25,
	             aggregation_func=lambda x: np.mean(x, axis=2),
	             nc_class_params=None,
	             n_models=10):
		super(SubSamplingCp, self).__init__(cp_class,
                                            nc_class,
                                            aggregation_func,
                                            nc_class_params,
                                            n_models)

		self.cal_portion = calibration_portion

	def _generate_samples(self, x, y):
		if self.cp_class.problem_type == 'classification':
			splits = StratifiedShuffleSplit(y,
			                                n_iter=self.n_models,
			                                test_size=self.cal_portion)
		else:
			splits = ShuffleSplit(y.size,
			                      n_iter=self.n_models,
			                      test_size=self.cal_portion)

		for train, cal in splits:
			yield train, cal