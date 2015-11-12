#!/usr/bin/env python

"""
Example: combining multiple inductive conformal classifiers
"""

# Authors: Henrik Linusson

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

from nonconformist.nc import  ProbEstClassifierNc
from nonconformist.icp import IcpClassifier
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier
from nonconformist.evaluation import class_mean_errors

# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------
data = load_iris()

idx = np.random.permutation(data.target.size)
train = idx[:int(2 * idx.size / 3)]
test = idx[int(2 * idx.size / 3):]

truth = data.target[test].reshape(-1, 1)
columns = ['C-{}'.format(i) for i in np.unique(data.target)] + ['truth']
significance = 0.1

# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------

models = {  'ACP-RandomSubSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ProbEstClassifierNc(
                                                DecisionTreeClassifier())),
                                        RandomSubSampler()),
            'ACP-CrossSampler'      : AggregatedCp(
                                        IcpClassifier(
                                            ProbEstClassifierNc(
                                                DecisionTreeClassifier())),
                                        CrossSampler()),
            'ACP-BootstrapSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ProbEstClassifierNc(
                                                DecisionTreeClassifier())),
                                        BootstrapSampler()),
            'CCP'                   : CrossConformalClassifier(
                                        IcpClassifier(
                                            ProbEstClassifierNc(
                                                DecisionTreeClassifier()))),
            'BCP'                   : BootstrapConformalClassifier(
                                        IcpClassifier(
                                            ProbEstClassifierNc(
                                                DecisionTreeClassifier())))
          }

# -----------------------------------------------------------------------------
# Train, predict and evaluate
# -----------------------------------------------------------------------------
for name, model in models.iteritems():
    model.fit(data.data[train, :], data.target[train])
    prediction = model.predict(data.data[test, :], significance=significance)
    table = np.hstack((prediction, truth))
    df = pd.DataFrame(table, columns=columns)
    print('\n{}'.format(name))
    print('Error rate: {}'.format(class_mean_errors(prediction,
                                                    truth,
                                                    significance)))
    print(df)
