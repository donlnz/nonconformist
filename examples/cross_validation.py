#!/usr/bin/env python

"""
Example: cross-validation evaluation of inductive conformal classification
and regression models.
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.cross_validation import cross_val_score, KFold

from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import margin
from nonconformist.nc import ProbEstClassifierNc, RegressorNc
from nonconformist.nc import sign_error, sign_error_inv
from nonconformist.nc import abs_error, abs_error_inv

from nonconformist.evaluation import IcpClassCrossValHelper, IcpRegCrossValHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.evaluation import reg_mean_size, reg_mean_errors


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
data = load_iris()

icp = IcpClassifier(ProbEstClassifierNc(RandomForestClassifier, margin))
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
                               abs_error,
                               abs_error_inv))
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

print('Absolute error regression: diabetes')
print('Mean errors: {}'.format(np.mean(errors)))
print('Mean interval size: {}\n'.format(np.mean(size)))

# -----------------------------------------------------------------------------
# Regression, signed error
# -----------------------------------------------------------------------------
data = load_diabetes()

icp = IcpRegressor(RegressorNc(RandomForestRegressor,
                               sign_error,
                               sign_error_inv))
icp_cv = IcpRegCrossValHelper(icp, significance=0.10)

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

print('Signed error regression: diabetes')
print('Mean errors: {}'.format(np.mean(errors)))
print('Mean interval size: {}\n'.format(np.mean(size)))