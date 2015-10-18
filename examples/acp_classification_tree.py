#!/usr/bin/env python

"""
Example: combining multiple inductive conformal classifiers
"""

# Authors: Henrik Linusson

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

from nonconformist.nc import margin, ProbEstClassifierNc
from nonconformist.icp import IcpClassifier
from nonconformist.acp import AggregatedCp, BootstrapSampler, CrossSampler, RandomSubSampler

data = load_iris()

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
idx = np.random.permutation(data.target.size)
train = idx[:int(2 * idx.size / 3)]
test = idx[int(2 * idx.size / 3):]

# -----------------------------------------------------------------------------
# Train and calibrate
# -----------------------------------------------------------------------------
rscp = AggregatedCp(IcpClassifier,
                     ProbEstClassifierNc,
                     sampler=RandomSubSampler(),
                     nc_class_params={'model_class': DecisionTreeClassifier,
                                      'err_func': margin})
rscp.fit(data.data[train, :], data.target[train])

ccp = AggregatedCp(IcpClassifier,
                  ProbEstClassifierNc,
                  sampler=CrossSampler(),
                  nc_class_params={'model_class': DecisionTreeClassifier,
                                   'err_func': margin})
ccp.fit(data.data[train, :], data.target[train])

bcp = AggregatedCp(IcpClassifier,
                  ProbEstClassifierNc,
                  sampler=BootstrapSampler(),
                  nc_class_params={'model_class': DecisionTreeClassifier,
                                   'err_func': margin})
bcp.fit(data.data[train, :], data.target[train])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------

truth = data.target[test].reshape(-1, 1)
columns = ['c0','c1','c2','truth']
significance = 0.1

prediction = rscp.predict(data.data[test, :], significance=significance)
table = np.hstack((prediction, truth))
df = pd.DataFrame(table, columns=columns)
print('RSCP')
print(df)

prediction = ccp.predict(data.data[test, :], significance=significance)
table = np.hstack((prediction, truth))
df = pd.DataFrame(table, columns=columns)
print('CCP')
print(df)

prediction = bcp.predict(data.data[test, :], significance=significance)
table = np.hstack((prediction, truth))
df = pd.DataFrame(table, columns=columns)
print('BCP')
print(df)
