#!/usr/bin/env python

"""
Example: inductive conformal regression using TreeEnsemble (nonconformist.tree)

With this small TreeEnsemble implementation, you can implement regression ICP without importing sklearn library.
This is quite useful if your application is deployed on an embedded Linux distribution which has limited resources.

"""

# Authors: Henrik Linusson

import numpy as np
import pandas as pd

from nonconformist.tree import TreeEnsemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston

from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc, AbsErrorErrFunc, RegressorNormalizer

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
data = load_boston()

idx = np.random.permutation(data.target.size)
train = idx[:int(idx.size / 3)]
calibrate = idx[int(idx.size / 3):int(2 * idx.size / 3)]
test = idx[int(2 * idx.size / 3):]

# -----------------------------------------------------------------------------
# Without normalization
# -----------------------------------------------------------------------------
# Train and calibrate
# -----------------------------------------------------------------------------
underlying_model = RegressorAdapter(TreeEnsemble(n_trees=3, sample_sz=2000))
nc = RegressorNc(underlying_model, AbsErrorErrFunc())
icp = IcpRegressor(nc)
icp.fit(data.data[train, :], data.target[train])
icp.calibrate(data.data[calibrate, :], data.target[calibrate])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
prediction = icp.predict(data.data[test, :], significance=0.1)
header = ['min','max','truth','size']
size = prediction[:, 1] - prediction[:, 0]
table = np.vstack([prediction.T, data.target[test], size.T]).T
df = pd.DataFrame(table, columns=header)
print(df)

# -----------------------------------------------------------------------------
# With normalization
# -----------------------------------------------------------------------------
# Train and calibrate
# -----------------------------------------------------------------------------
underlying_model = RegressorAdapter(TreeEnsemble(n_trees=3, sample_sz=2000))
normalizing_model = RegressorAdapter(KNeighborsRegressor(n_neighbors=1))
normalizer = RegressorNormalizer(underlying_model, normalizing_model, AbsErrorErrFunc())
nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)
icp = IcpRegressor(nc)
icp.fit(data.data[train, :], data.target[train])
icp.calibrate(data.data[calibrate, :], data.target[calibrate])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
prediction = icp.predict(data.data[test, :], significance=0.1)
header = ['min','max','truth','size']
size = prediction[:, 1] - prediction[:, 0]
table = np.vstack([prediction.T, data.target[test], size.T]).T
df = pd.DataFrame(table, columns=header)
print(df)