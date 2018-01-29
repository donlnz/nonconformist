from nonconformist.icp import *

# TODO: move contents from nonconformist.icp here

# -----------------------------------------------------------------------------
# TcpClassifier
# -----------------------------------------------------------------------------
class TcpClassifier(BaseEstimator, ClassifierMixin):
	"""Transductive conformal classifier.

	Parameters
	----------
	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``
		and ``calc_nc(x, y)``.

	smoothing : boolean
		Decides whether to use stochastic smoothing of p-values.

	Attributes
	----------
	train_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of training set.

	train_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity scores.

	classes : numpy array of shape [n_classes]
		List of class labels, with indices corresponding to output columns
		 of TcpClassifier.predict()

	See also
	--------
	IcpClassifier

	References
	----------
	.. [1] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning
	in a random world. Springer Science & Business Media.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_iris
	>>> from sklearn.svm import SVC
	>>> from nonconformist.base import ClassifierAdapter
	>>> from nonconformist.cp import TcpClassifier
	>>> from nonconformist.nc import ClassifierNc, MarginErrFunc
	>>> iris = load_iris()
	>>> idx = np.random.permutation(iris.target.size)
	>>> train = idx[:int(idx.size / 2)]
	>>> test = idx[int(idx.size / 2):]
	>>> model = ClassifierAdapter(SVC(probability=True))
	>>> nc = ClassifierNc(model, MarginErrFunc())
	>>> tcp = TcpClassifier(nc)
	>>> tcp.fit(iris.data[train, :], iris.target[train])
	>>> tcp.predict(iris.data[test, :], significance=0.10)
	...             # doctest: +SKIP
	array([[ True, False, False],
		[False,  True, False],
		...,
		[False,  True, False],
		[False,  True, False]], dtype=bool)
	"""

	def __init__(self, nc_function, condition=None, smoothing=True):
		self.train_x, self.train_y = None, None
		self.nc_function = nc_function
		super(TcpClassifier, self).__init__()

		# Check if condition-parameter is the default function (i.e.,
		# lambda x: 0). This is so we can safely clone the object without
		# the clone accidentally having self.conditional = True.
		default_condition = lambda x: 0
		is_default = (callable(condition) and
		              (condition.__code__.co_code ==
		               default_condition.__code__.co_code))

		if is_default:
			self.condition = condition
			self.conditional = False
		elif callable(condition):
			self.condition = condition
			self.conditional = True
		else:
			self.condition = lambda x: 0
			self.conditional = False

		self.smoothing = smoothing

		self.base_icp = IcpClassifier(
			self.nc_function,
			self.condition,
			self.smoothing
		)

		self.classes = None

	def fit(self, x, y):
		self.train_x, self.train_y = x, y
		self.classes = np.unique(y)

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
		n_test = x.shape[0]
		n_train = self.train_x.shape[0]
		p = np.zeros((n_test, self.classes.size))
		for i in range(n_test):
			for j, y in enumerate(self.classes):
				train_x = np.vstack([self.train_x, x[i, :]])
				train_y = np.hstack([self.train_y, y])
				self.base_icp.fit(train_x, train_y)
				scores = self.base_icp.nc_function.score(train_x, train_y)
				ngt = (scores[:-1] > scores[-1]).sum()
				neq = (scores[:-1] == scores[-1]).sum()

				p[i, j] = calc_p(n_train, ngt, neq, self.smoothing)

		if significance is not None:
			return p > significance
		else:
			return p

	def predict_conf(self, x):
		"""Predict the output values for a set of input patterns, using
		the confidence-and-credibility output scheme.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		Returns
		-------
		p : numpy array of shape [n_samples, 3]
			p contains three columns: the first column contains the most
			likely class for each test pattern; the second column contains
			the confidence in the predicted class label, and the third column
			contains the credibility of the prediction.
		"""
		p = self.predict(x, significance=None)
		label = p.argmax(axis=1)
		credibility = p.max(axis=1)
		for i, idx in enumerate(label):
			p[i, idx] = -np.inf
		confidence = 1 - p.max(axis=1)

		return np.array([label, confidence, credibility]).T
