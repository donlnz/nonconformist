#!/usr/bin/env python

"""
Example: inductive conformal classification using DecisionTreeClassifier
"""

# Authors: Henrik Linusson

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.datasets import load_iris

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import TcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.evaluation import class_mean_errors

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
data = load_iris()

idx = np.random.permutation(data.target.size)
train = idx[:int(idx.size / 2)]
test = idx[int(idx.size / 2):]

# -----------------------------------------------------------------------------
# Train and calibrate TCP
# -----------------------------------------------------------------------------
tcp = TcpClassifier(
	ClassifierNc(
		ClassifierAdapter(SVC(probability=True, gamma='scale')),
		MarginErrFunc()
	)
)

tcp.fit(data.data[train, :], data.target[train])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
prediction = tcp.predict(data.data[test, :], significance=0.1)
header = np.array(['c0','c1','c2','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pd.DataFrame(np.vstack([header, table]))
print('TCP')
print('---')
print(df)

error_rate = class_mean_errors(tcp.predict(data.data[test, :]), data.target[test], significance=0.1)
print('Error rate: {}'.format(error_rate))


# -----------------------------------------------------------------------------
# Train and calibrate Mondrian (class-conditional) TCP
# -----------------------------------------------------------------------------
tcp = TcpClassifier(
	ClassifierNc(
		ClassifierAdapter(SVC(probability=True, gamma='scale')),
		MarginErrFunc()
	),
	condition=lambda x: x[1],
)

tcp.fit(data.data[train, :], data.target[train])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
prediction = tcp.predict(data.data[test, :], significance=0.1)
header = np.array(['c0','c1','c2','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pd.DataFrame(np.vstack([header, table]))
print('\nClass-conditional TCP')
print('---------------------')
print(df)

error_rate = class_mean_errors(tcp.predict(data.data[test, :]), data.target[test], significance=0.1)
print('Error rate: {}'.format(error_rate))
