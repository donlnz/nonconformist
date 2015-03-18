#!/usr/bin/env python

"""
docstring
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score

from nonconformist.classification import IcpClassifier, PetClassifierNc
from nonconformist.classification import margin
from nonconformist.evaluation import IcpCrossValidationHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors

data = load_iris()

# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------
icp = IcpClassifier(PetClassifierNc(RandomForestClassifier, margin))
icp_cv = IcpCrossValidationHelper(icp, significance=0.20)

errors = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         scoring=class_mean_errors,
                         cv=10)



size = cross_val_score(icp_cv,
                       data.data,
                       data.target,
                       scoring=class_avg_c,
                       cv=10)

from pprint import pprint
print('Errors:')
pprint(errors)
print('Mean: {}\n'.format(np.mean(errors)))
print('AvgC:')
pprint(size)
print('Mean: {}\n'.format(np.mean(size)))