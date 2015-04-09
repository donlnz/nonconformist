#!/usr/bin/env python

"""
Example: combining multiple inductive conformal regressors
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes

from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc, abs_error, abs_error_inv
from nonconformist.acp import AggregatedCp
from nonconformist.acp import RandomSubSampler, BootstrapSampler, CrossSampler


data = load_diabetes()

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
idx = np.random.permutation(data.target.size)
train = idx[:int(2 * idx.size / 3)]
test = idx[int(2 * idx.size / 3):]

# -----------------------------------------------------------------------------
# Train and calibrate
# -----------------------------------------------------------------------------
nc_class_params = {'model_class': DecisionTreeRegressor,
                   'err_func': abs_error,
                   'inverse_err_func': abs_error_inv}

rscp = AggregatedCp(IcpRegressor,
                    RegressorNc,
                    sampler=RandomSubSampler(),
                    nc_class_params=nc_class_params)
rscp.fit(data.data[train, :], data.target[train])

ccp = AggregatedCp(IcpRegressor,
                   RegressorNc,
                   sampler=CrossSampler(),
                   nc_class_params=nc_class_params)
ccp.fit(data.data[train, :], data.target[train])

bcp = AggregatedCp(IcpRegressor,
                   RegressorNc,
                   sampler=BootstrapSampler(),
                   nc_class_params=nc_class_params)
bcp.fit(data.data[train, :], data.target[train])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
import pandas

prediction = rscp.predict(data.data[test, :], significance=0.1)
header = np.array(['min', 'max','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print('RSCP')
print(df)

prediction = ccp.predict(data.data[test, :], significance=0.1)
header = np.array(['min', 'max','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print('CCP')
print(df)

prediction = bcp.predict(data.data[test, :], significance=0.1)
header = np.array(['min', 'max','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print('BCP')
print(df)