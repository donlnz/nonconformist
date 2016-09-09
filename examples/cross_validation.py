#!/usr/bin/env python

"""
Example: cross-validation evaluation of inductive conformal classification
and regression models.
"""

# Authors: Henrik Linusson

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import MarginErrFunc
from nonconformist.nc import ClassifierNc, RegressorNc
from nonconformist.nc import NormalizedRegressorNc
from nonconformist.nc import AbsErrorErrFunc, SignErrorErrFunc

from nonconformist.evaluation import cross_val_score
from nonconformist.evaluation import ClassIcpCvHelper, RegIcpCvHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.evaluation import reg_mean_errors, reg_median_size


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
data = load_iris()

icp = IcpClassifier(ClassifierNc(ClassifierAdapter(RandomForestClassifier(n_estimators=100)),
                                 MarginErrFunc()))
icp_cv = ClassIcpCvHelper(icp)

scores = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         iterations=5,
                         folds=5,
                         scoring_funcs=[class_mean_errors, class_avg_c],
                         significance_levels=[0.05, 0.1, 0.2])

print('Classification: iris')
scores = scores.drop(['fold', 'iter'], axis=1)
print(scores.groupby(['significance']).mean())

# -----------------------------------------------------------------------------
# Regression, absolute error
# -----------------------------------------------------------------------------
data = load_diabetes()

icp = IcpRegressor(RegressorNc(RegressorAdapter(RandomForestRegressor(n_estimators=100)),
                               AbsErrorErrFunc()))
icp_cv = RegIcpCvHelper(icp)

scores = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         iterations=5,
                         folds=5,
                         scoring_funcs=[reg_mean_errors, reg_median_size],
                         significance_levels=[0.05, 0.1, 0.2])


print('Absolute error regression: diabetes')
scores = scores.drop(['fold', 'iter'], axis=1)
print(scores.groupby(['significance']).mean())

# -----------------------------------------------------------------------------
# Regression, absolute error
# -----------------------------------------------------------------------------
data = load_diabetes()

icp = IcpRegressor(NormalizedRegressorNc(RegressorAdapter(RandomForestRegressor(n_estimators=100)),
                                         RegressorAdapter(RandomForestRegressor(n_estimators=100)),
                                         AbsErrorErrFunc()))
icp_cv = RegIcpCvHelper(icp)

scores = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         iterations=5,
                         folds=5,
                         scoring_funcs=[reg_mean_errors, reg_median_size],
                         significance_levels=[0.05, 0.1, 0.2])


print('Normalized absolute error regression: diabetes')
scores = scores.drop(['fold', 'iter'], axis=1)
print(scores.groupby(['significance']).mean())

# -----------------------------------------------------------------------------
# Regression, signed error
# -----------------------------------------------------------------------------
data = load_diabetes()

icp = IcpRegressor(RegressorNc(RegressorAdapter(RandomForestRegressor(n_estimators=100)),
                               SignErrorErrFunc()))
icp_cv = RegIcpCvHelper(icp)

scores = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         iterations=5,
                         folds=5,
                         scoring_funcs=[reg_mean_errors, reg_median_size],
                         significance_levels=[0.05, 0.1, 0.2])

print('Signed error regression: diabetes')
scores = scores.drop(['fold', 'iter'], axis=1)
print(scores.groupby(['significance']).mean())

# -----------------------------------------------------------------------------
# Regression, signed error
# -----------------------------------------------------------------------------
data = load_diabetes()

icp = IcpRegressor(NormalizedRegressorNc(RegressorAdapter(RandomForestRegressor(n_estimators=100)),
                                         RegressorAdapter(RandomForestRegressor(n_estimators=100)),
                                         AbsErrorErrFunc()))
icp_cv = RegIcpCvHelper(icp)

scores = cross_val_score(icp_cv,
                         data.data,
                         data.target,
                         iterations=5,
                         folds=5,
                         scoring_funcs=[reg_mean_errors, reg_median_size],
                         significance_levels=[0.05, 0.1, 0.2])

print('Normalized signed error regression: diabetes')
scores = scores.drop(['fold', 'iter'], axis=1)
print(scores.groupby(['significance']).mean())
