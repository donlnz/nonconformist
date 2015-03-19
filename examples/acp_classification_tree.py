#!/usr/bin/env python

"""
Example: combining multiple inductive conformal classifiers
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

from nonconformist.ensemble import *
from nonconformist.classification import IcpClassifier, PetClassifierNc
from nonconformist.classification import margin

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
                     PetClassifierNc,
                     sampler=RandomSubSampler(),
                     nc_class_params={'model_class': DecisionTreeClassifier,
                                      'err_func': margin})
rscp.fit(data.data[train, :], data.target[train])

ccp = AggregatedCp(IcpClassifier,
              PetClassifierNc,
              sampler=BootstrapSampler(),
              nc_class_params={'model_class': DecisionTreeClassifier,
                               'err_func': margin})
ccp.fit(data.data[train, :], data.target[train])

bcp = AggregatedCp(IcpClassifier,
                  PetClassifierNc,
                  sampler=BootstrapSampler(),
                  nc_class_params={'model_class': DecisionTreeClassifier,
                                   'err_func': margin})
bcp.fit(data.data[train, :], data.target[train])

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
import pandas

prediction = rscp.predict(data.data[test, :], significance=0.1)
header = np.array(['c0','c1','c2','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print('RSCP')
print(df)

prediction = ccp.predict(data.data[test, :], significance=0.1)
header = np.array(['c0','c1','c2','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print('CCP')
print(df)

prediction = bcp.predict(data.data[test, :], significance=0.1)
header = np.array(['c0','c1','c2','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pandas.DataFrame(np.vstack([header, table]))
print('BCP')
print(df)