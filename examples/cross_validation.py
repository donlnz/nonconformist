#!/usr/bin/env python

"""
docstring
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.cross_validation import cross_val_score, KFold

from nonconformist.classification import IcpClassifier, PetClassifierNc, margin
from nonconformist.regression import IcpRegressor, RegressorNc
from nonconformist.regression import signed_error, signed_error_inverse
from nonconformist.regression import absolute_error, absolute_error_inverse

from nonconformist.evaluation import IcpClassCrossValHelper, IcpRegCrossValHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.evaluation import reg_mean_size, reg_mean_errors


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
data = load_iris()

icp = IcpClassifier(PetClassifierNc(RandomForestClassifier, margin))
icp_cv = IcpClassCrossValHelper(icp, significance=0.05)

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

print('Classification: iris')
print('Mean errors: {}'.format(np.mean(errors)))
print('Mean avgc: {}\n'.format(np.mean(size)))

# -----------------------------------------------------------------------------
# Regression, absolute error
# -----------------------------------------------------------------------------
data = load_diabetes()

icp = IcpRegressor(RegressorNc(RandomForestRegressor,
                               absolute_error,
                               absolute_error_inverse))
icp_cv = IcpRegCrossValHelper(icp, significance=0.01)

folds = KFold(data.target.size, 10, shuffle=True)
errors = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         scoring=reg_mean_errors,
                         cv=folds)

size = cross_val_score(icp_cv,
                       data.data,
                       data.target,
                       scoring=reg_mean_size,
                       cv=folds)

print('Absolute error regression: boston')
print('Mean errors: {}'.format(np.mean(errors)))
print('Mean interval size: {}\n'.format(np.mean(size)))

# -----------------------------------------------------------------------------
# Regression, signed error
# -----------------------------------------------------------------------------
data = load_diabetes()

icp = IcpRegressor(RegressorNc(RandomForestRegressor,
                               signed_error,
                               signed_error_inverse))
icp_cv = IcpRegCrossValHelper(icp, significance=0.05)

folds = KFold(data.target.size, 10, shuffle=True)
errors = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         scoring=reg_mean_errors,
                         cv=folds)

size = cross_val_score(icp_cv,
                       data.data,
                       data.target,
                       scoring=reg_mean_size,
                       cv=folds)

print('Signed error regression: boston')
print('Mean errors: {}'.format(np.mean(errors)))
print('Mean interval size: {}\n'.format(np.mean(size)))