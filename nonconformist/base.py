#!/usr/bin/env python

"""
docstring
"""

# Authors: Henrik Linusson

class RegressorMixin(object):
	@classmethod
	def get_problem_type(self):
		return 'regression'

class ClassifierMixin(object):
	@classmethod
	def get_problem_type(self):
		return 'classification'