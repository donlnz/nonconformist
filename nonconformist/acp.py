#!/usr/bin/env python

"""
Aggregated conformal predictors
"""

# Authors: Henrik Linusson

import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit
from sklearn.base import clone
from nonconformist.base import BaseEstimator
from nonconformist.util import calc_p


# -----------------------------------------------------------------------------
# Sampling strategies
# -----------------------------------------------------------------------------
class BootstrapSampler(object):
	"""Bootstrap sampler.

	See also
	--------
	CrossSampler, RandomSubSampler

	Examples
	--------
	"""
	def gen_samples(self, y, n_samples, problem_type):
		for i in range(n_samples):
			idx = np.array(range(y.size))
			train = np.random.choice(y.size, y.size, replace=True)
			cal_mask = np.array(np.ones(idx.size), dtype=bool)
			for j in train:
				cal_mask[j] = False
			cal = idx[cal_mask]

			yield train, cal


class CrossSampler(object):
	"""Cross-fold sampler.

	See also
	--------
	BootstrapSampler, RandomSubSampler

	Examples
	--------
	"""
	def gen_samples(self, y, n_samples, problem_type):
		if problem_type == 'classification':
			folds = StratifiedKFold(y, n_folds=n_samples)
		else:
			folds = KFold(y.size, n_folds=n_samples)
		for train, cal in folds:
			yield train, cal


class RandomSubSampler(object):
	"""Random subsample sampler.

	Parameters
	----------
	calibration_portion : float
		Ratio (0-1) of examples to use for calibration.

	See also
	--------
	BootstrapSampler, CrossSampler

	Examples
	--------
	"""
	def __init__(self, calibration_portion=0.3):
		self.cal_portion = calibration_portion

	def gen_samples(self, y, n_samples, problem_type):
		if problem_type == 'classification':
			splits = StratifiedShuffleSplit(y,
			                                n_iter=n_samples,
			                                test_size=self.cal_portion)
		else:
			splits = ShuffleSplit(y.size,
			                      n_iter=n_samples,
			                      test_size=self.cal_portion)

		for train, cal in splits:
			yield train, cal


# -----------------------------------------------------------------------------
# Conformal ensemble
# -----------------------------------------------------------------------------
class AggregatedCp(BaseEstimator):
	"""Aggregated conformal predictor.

	Combines multiple IcpClassifier or IcpRegressor predictors into an
	aggregated model.

	Parameters
	----------
	predictor : object
		Prototype conformal predictor (e.g. IcpClassifier or IcpRegressor)
		used for defining conformal predictors included in the aggregate model.

	sampler : object
		Sampler object used to generate training and calibration examples
		for the underlying conformal predictors.

	aggregation_func : callable
		Function used to aggregate the predictions of the underlying
		conformal predictors. Defaults to ``numpy.mean``.

	n_models : int
		Number of models to aggregate.

	Attributes
	----------
	predictor : object
		Prototype conformal predictor.

	predictors : list
		List of underlying conformal predictors.

	sampler : object
		Sampler object used to generate training and calibration examples.

	agg_func : callable
		Function used to aggregate the predictions of the underlying
		conformal predictors

	References
	----------
	.. [1] Vovk, V. (2013). Cross-conformal predictors. Annals of Mathematics
		and Artificial Intelligence, 1-20.

	.. [2] Carlsson, L., Eklund, M., & Norinder, U. (2014). Aggregated
		Conformal Prediction. In Artificial Intelligence Applications and
		Innovations (pp. 231-240). Springer Berlin Heidelberg.

	Examples
	--------
	"""
	def __init__(self,
	             predictor,
	             sampler=BootstrapSampler(),
	             aggregation_func=None,
	             n_models=10):
		self.predictors = []
		self.n_models = n_models
		self.predictor = predictor
		self.sampler = sampler

		if aggregation_func is not None:
			self.agg_func = aggregation_func
		else:
			self.agg_func = lambda x: np.mean(x, axis=2)

	def fit(self, x, y):
		"""Fit underlying conformal predictors.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the underlying conformal predictors.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the underlying conformal predictors.

		Returns
		-------
		None
		"""
		self.n_train = y.size
		self.predictors = []
		idx = np.random.permutation(y.size)
		x, y = x[idx, :], y[idx]
		problem_type = self.predictor.__class__.get_problem_type()
		samples = self.sampler.gen_samples(y,
		                                   self.n_models,
		                                   problem_type)
		for train, cal in samples:
			predictor = clone(self.predictor)
			predictor.fit(x[train, :], y[train])
			predictor.calibrate(x[cal, :], y[cal])
			self.predictors.append(predictor)

		if problem_type == 'classification':
			self.classes = self.predictors[0].classes

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float or None
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then the p-values
			are output rather than the predictions. Note: ``significance=None``
			is applicable to classification problems only.

		Returns
		-------
		p : numpy array of shape [n_samples, n_classes] or [n_samples, 2]
			For classification problems: If significance is ``None``, then p
			contains the p-values for each sample-class pair; if significance
			is a float between 0 and 1, then p is a boolean array denoting
			which labels are included in the prediction sets.

			For regression problems: Prediction interval (minimum and maximum
			boundaries) for the set of test patterns.
		"""
		is_regression =\
			self.predictor.__class__.get_problem_type() == 'regression'

		n_examples = x.shape[0]

		if is_regression and significance is None:
			signs = np.arange(0.01, 1.0, 0.01)
			pred = np.zeros((n_examples, 2, signs.size))
			for i, s in enumerate(signs):
				predictions = np.dstack([p.predict(x, s)
				                         for p in self.predictors])
				predictions = self.agg_func(predictions)
				pred[:, :, i] = predictions
			return pred
		else:
			def f(p, x):
				return p.predict(x, significance if is_regression else None)
			predictions = np.dstack([f(p, x) for p in self.predictors])
			predictions = self.agg_func(predictions)

			if significance and not is_regression:
				return predictions >= significance
			else:
				return predictions


class CrossConformalClassifier(AggregatedCp):
	"""Cross-conformal classifier.

	Combines multiple IcpClassifiers into a cross-conformal classifier.

	Parameters
	----------
	predictor : object
		Prototype conformal predictor (e.g. IcpClassifier or IcpRegressor)
		used for defining conformal predictors included in the aggregate model.

	aggregation_func : callable
		Function used to aggregate the predictions of the underlying
		conformal predictors. Defaults to ``numpy.mean``.

	n_models : int
		Number of models to aggregate.

	Attributes
	----------
	predictor : object
		Prototype conformal predictor.

	predictors : list
		List of underlying conformal predictors.

	sampler : object
		Sampler object used to generate training and calibration examples.

	agg_func : callable
		Function used to aggregate the predictions of the underlying
		conformal predictors

	References
	----------
	.. [1] Vovk, V. (2013). Cross-conformal predictors. Annals of Mathematics
		and Artificial Intelligence, 1-20.

	Examples
	--------
	"""
	def __init__(self,
				 predictor,
				 n_models=10):
		super(CrossConformalClassifier, self).__init__(predictor,
													   CrossSampler(),
													   n_models)

	def predict(self, x, significance=None):
		ncal_ngt_neq = np.stack([p._get_stats(x) for p in self.predictors],
		                        axis=3)
		ncal_ngt_neq = ncal_ngt_neq.sum(axis=3)

		p = calc_p(ncal_ngt_neq[:, :, 0],
		           ncal_ngt_neq[:, :, 1],
		           ncal_ngt_neq[:, :, 2],
		           smoothing=self.predictors[0].smoothing)

		if significance:
			return p > significance
		else:
			return p


class BootstrapConformalClassifier(AggregatedCp):
	"""Bootstrap conformal classifier.

	Combines multiple IcpClassifiers into a bootstrap conformal classifier.

	Parameters
	----------
	predictor : object
		Prototype conformal predictor (e.g. IcpClassifier or IcpRegressor)
		used for defining conformal predictors included in the aggregate model.

	aggregation_func : callable
		Function used to aggregate the predictions of the underlying
		conformal predictors. Defaults to ``numpy.mean``.

	n_models : int
		Number of models to aggregate.

	Attributes
	----------
	predictor : object
		Prototype conformal predictor.

	predictors : list
		List of underlying conformal predictors.

	sampler : object
		Sampler object used to generate training and calibration examples.

	agg_func : callable
		Function used to aggregate the predictions of the underlying
		conformal predictors

	References
	----------
	.. [1] Vovk, V. (2013). Cross-conformal predictors. Annals of Mathematics
		and Artificial Intelligence, 1-20.

	Examples
	--------
	"""
	def __init__(self,
				 predictor,
	             n_models=10):
		super(BootstrapConformalClassifier, self).__init__(predictor,
														   BootstrapSampler(),
														   n_models)

	def predict(self, x, significance=None):
		ncal_ngt_neq = np.stack([p._get_stats(x) for p in self.predictors],
		                        axis=3)
		ncal_ngt_neq = ncal_ngt_neq.sum(axis=3)

		p = calc_p(ncal_ngt_neq[:, :, 0] + ncal_ngt_neq[:, :, 0] / self.n_train,
		           ncal_ngt_neq[:, :, 1] + ncal_ngt_neq[:, :, 0] / self.n_train,
		           ncal_ngt_neq[:, :, 2],
		           smoothing=self.predictors[0].smoothing)

		if significance:
			return p > significance
		else:
			return p
