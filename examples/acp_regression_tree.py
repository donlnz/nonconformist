#!/usr/bin/env python

"""
Example: combining multiple inductive conformal regressors
"""

# Authors: Henrik Linusson

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes

from nonconformist.ensemble import *
from nonconformist.regression import IcpRegressor, RegressorNc
from nonconformist.regression import absolute_error, absolute_error_inverse

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
rscp = SubSamplingCp(IcpRegressor,
                     RegressorNc,
                     nc_class_params={'model_class': DecisionTreeRegressor,
                                      'err_func': absolute_error,
                                      'inverse_err_func': absolute_error_inverse})
rscp.fit(data.data[train, :], data.target[train])

ccp = CrossCp(IcpRegressor,
              RegressorNc,
              nc_class_params={'model_class': DecisionTreeRegressor,
                               'err_func': absolute_error,
                               'inverse_err_func': absolute_error_inverse})
ccp.fit(data.data[train, :], data.target[train])

bcp = BootstrapCp(IcpRegressor,
                  RegressorNc,
                  nc_class_params={'model_class': DecisionTreeRegressor,
                                   'err_func': absolute_error,
                                   'inverse_err_func': absolute_error_inverse})
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