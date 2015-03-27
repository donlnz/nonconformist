#!/usr/bin/env python

"""
Example: inductive conformal classification using DecisionTreeClassifier
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

from nonconformist.icp import IcpClassifier
from nonconformist.nc import ProbEstClassifierNc, margin

data = load_iris()

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
idx = np.random.permutation(data.target.size)
train = idx[:int(idx.size / 3)]
calibrate = idx[int(idx.size / 3):int(2 * idx.size / 3)]
test = idx[int(2 * idx.size / 3):]

# -----------------------------------------------------------------------------
# Train and calibrate
# -----------------------------------------------------------------------------
icp = IcpClassifier(ProbEstClassifierNc(DecisionTreeClassifier, margin))
icp.fit(data.data[train, :], data.target[train])
icp.calibrate(data.data[calibrate, :], data.target[calibrate])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
import pandas

prediction = icp.predict(data.data[test, :], significance=0.1)
header = np.array(['c0','c1','c2','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print(df)