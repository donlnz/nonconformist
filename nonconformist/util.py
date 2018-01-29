from __future__ import division
import numpy as np

def calc_p(ncal, ngt, neq, smoothing=False):
	if smoothing:
		return (ngt + (neq + 1) * np.random.uniform(0, 1)) / (ncal + 1)
	else:
		return (ngt + neq + 1) / (ncal + 1)
