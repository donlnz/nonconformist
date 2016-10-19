#!/usr/bin/env python

"""
Example: cross-validation evaluation of inductive conformal classification
and regression models.
"""

# Authors: Henrik Linusson

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.neighbors import KNeighborsRegressor

from nonconformist.nc import NcFactory

from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.icp import OobCpClassifier, OobCpRegressor
from nonconformist.evaluation import cross_val_score
from nonconformist.evaluation import ClassIcpCvHelper, RegIcpCvHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.evaluation import reg_mean_errors, reg_median_size


def score_model(icp, icp_name, ds, ds_name, scoring_funcs):
	scores = cross_val_score(icp,
	                         ds.data,
	                         ds.target,
	                         iterations=10,
	                         folds=10,
	                         scoring_funcs=scoring_funcs,
	                         significance_levels=[0.05, 0.1, 0.2])

	print('\n{}: {}'.format(icp_name, ds_name))
	scores = scores.drop(['fold', 'iter'], axis=1)
	print(scores.groupby(['significance']).mean())

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
data = load_iris()

nc = NcFactory.create_nc(RandomForestClassifier(n_estimators=100))
icp = IcpClassifier(nc)
icp_cv = ClassIcpCvHelper(icp)
score_model(icp_cv,
            'IcpClassifier',
            data,
            'iris',
            [class_mean_errors, class_avg_c])

# -----------------------------------------------------------------------------
# Classification (normalized)
# -----------------------------------------------------------------------------
data = load_iris()

nc = NcFactory.create_nc(RandomForestClassifier(n_estimators=100),
                         normalizer_model=KNeighborsRegressor())
icp = IcpClassifier(nc)
icp_cv = ClassIcpCvHelper(icp)

score_model(icp_cv,
            'IcpClassifier (normalized)',
            data,
            'iris',
            [class_mean_errors, class_avg_c])

# -----------------------------------------------------------------------------
# Classification OOB
# -----------------------------------------------------------------------------
data = load_iris()

nc = NcFactory.create_nc(RandomForestClassifier(n_estimators=100,
                                                oob_score=True),
                         oob=True)
icp_cv = OobCpClassifier(nc)

score_model(icp_cv,
            'IcpClassifier (OOB)',
            data,
            'iris',
            [class_mean_errors, class_avg_c])

# -----------------------------------------------------------------------------
# Classification OOB normalized
# -----------------------------------------------------------------------------
data = load_iris()

nc = NcFactory.create_nc(RandomForestClassifier(n_estimators=100,
                                                oob_score=True),
                         oob=True,
                         normalizer_model=KNeighborsRegressor())
icp_cv = OobCpClassifier(nc)

score_model(icp_cv,
            'IcpClassifier (OOB, normalized)',
            data,
            'iris',
            [class_mean_errors, class_avg_c])

# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------
data = load_diabetes()

nc = NcFactory.create_nc(RandomForestRegressor(n_estimators=100))
icp = IcpRegressor(nc)
icp_cv = RegIcpCvHelper(icp)

score_model(icp_cv,
            'IcpRegressor',
            data,
            'diabetes',
            [reg_mean_errors, reg_median_size])

# -----------------------------------------------------------------------------
# Regression (normalized)
# -----------------------------------------------------------------------------
data = load_diabetes()

nc = NcFactory.create_nc(RandomForestRegressor(n_estimators=100),
                         normalizer_model=KNeighborsRegressor())
icp = IcpRegressor(nc)
icp_cv = RegIcpCvHelper(icp)

score_model(icp_cv,
            'IcpRegressor (normalized)',
            data,
            'diabetes',
            [reg_mean_errors, reg_median_size])

# -----------------------------------------------------------------------------
# Regression OOB
# -----------------------------------------------------------------------------
data = load_diabetes()

nc = NcFactory.create_nc(RandomForestRegressor(n_estimators=100,
                                               oob_score=True),
                         oob=True)
icp_cv = OobCpRegressor(nc)

score_model(icp_cv,
            'IcpRegressor (OOB)',
            data,
            'diabetes',
            [reg_mean_errors, reg_median_size])

# -----------------------------------------------------------------------------
# Regression OOB normalized
# -----------------------------------------------------------------------------
data = load_diabetes()

nc = NcFactory.create_nc(RandomForestRegressor(n_estimators=100,
                                               oob_score=True),
                         oob=True,
                         normalizer_model=KNeighborsRegressor())
icp_cv = OobCpRegressor(nc)

score_model(icp_cv,
            'IcpRegressor (OOB, normalized)',
            data,
            'diabetes',
            [reg_mean_errors, reg_median_size])