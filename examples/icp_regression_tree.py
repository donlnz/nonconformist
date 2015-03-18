#!/usr/bin/env python

"""
Example: inductive conformal regression using DecisionTreeRegressor
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston

from nonconformist.regression import IcpRegressor, RegressorNc
from nonconformist.regression import absolute_error, absolute_error_inverse

data = load_boston()

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
icp = IcpRegressor(RegressorNc(DecisionTreeRegressor,
                               absolute_error,
                               absolute_error_inverse))
icp.fit(data.data[train, :], data.target[train])
icp.calibrate(data.data[calibrate, :], data.target[calibrate])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
import pandas

prediction = icp.predict(data.data[test, :], significance=0.1)
header = np.array(['min','max','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print(df)