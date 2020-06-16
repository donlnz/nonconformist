import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from nonconformist.base import ClassifierAdapter
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc

data = load_iris()
x, y = data.data, data.target

for i, y_ in enumerate(np.unique(y)):
	y[y == y_] = i

n_instances = y.size
idx = np.random.permutation(n_instances)

train_idx = idx[:int(n_instances / 3)]
cal_idx = idx[int(n_instances / 3):2 * int(n_instances / 3)]
test_idx = idx[2 * int(n_instances / 3):]

nc = ClassifierNc(ClassifierAdapter(RandomForestClassifier(n_estimators=100)))
icp = IcpClassifier(nc)

icp.fit(x[train_idx, :], y[train_idx])
icp.calibrate(x[cal_idx, :], y[cal_idx])


print(pd.DataFrame(icp.predict_conf(x[test_idx, :]),
				   columns=['Label', 'Confidence', 'Credibility']))