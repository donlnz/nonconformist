#!/usr/bin/env python

"""
Example: combining multiple inductive conformal regressors
"""

# Authors: Henrik Linusson

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes

from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc
from nonconformist.acp import AggregatedCp
from nonconformist.acp import RandomSubSampler, BootstrapSampler, CrossSampler
from nonconformist.evaluation import reg_mean_errors

# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------
data = load_diabetes()

idx = np.random.permutation(data.target.size)
train = idx[:int(2 * idx.size / 3)]
test = idx[int(2 * idx.size / 3):]

truth = data.target[test]
columns = ['min', 'max', 'truth']
significance = 0.1

# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------

models = {  'ACP-RandomSubSampler'  : AggregatedCp(
                                    IcpRegressor(
                                        RegressorNc(
                                            DecisionTreeRegressor())),
                                    RandomSubSampler()),
            'ACP-CrossSampler'      : AggregatedCp(
                                        IcpRegressor(
                                            RegressorNc(
                                                DecisionTreeRegressor())),
                                        CrossSampler()),
            'ACP-BootstrapSampler'  : AggregatedCp(
                                        IcpRegressor(
                                            RegressorNc(
                                                DecisionTreeRegressor())),
                                        BootstrapSampler())
      }

# -----------------------------------------------------------------------------
# Train, predict and evaluate
# -----------------------------------------------------------------------------
for name, model in models.iteritems():
    model.fit(data.data[train, :], data.target[train])
    prediction = model.predict(data.data[test, :])
    prediction_sign = model.predict(data.data[test, :],
                                    significance=significance)
    table = np.vstack((prediction_sign.T, truth)).T
    df = pd.DataFrame(table, columns=columns)
    print('\n{}'.format(name))
    print('Error rate: {}'.format(reg_mean_errors(prediction,
                                                  truth,
                                                  significance)))
    print(df)
