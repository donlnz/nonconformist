nonconformist API
=================

.. _nonconformity_ref:

:mod:`nc`
---------
.. automodule:: nonconformist.nc
	:no-members:
	:no-inherited-members:

Nonconformity Functions
~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: nonconformist

.. autosummary::
	:template: class.rst
	:toctree: _autosummary/

	nc.BaseModelNc
	nc.ClassifierNc
	nc.RegressorNc

Error Functions
~~~~~~~~~~~~~~~
.. currentmodule:: nonconformist

.. autosummary::
	:template: class.rst
	:toctree: _autosummary/

	nc.ClassificationErrFunc
	nc.RegressionErrFunc
	nc.InverseProbabilityErrFunc
	nc.MarginErrFunc
	nc.AbsErrorErrFunc
	nc.SignErrorErrFunc


.. _icp_ref:

:mod:`icp`
----------
.. automodule:: nonconformist.icp
	:no-members:
	:no-inherited-members:

Classes
~~~~~~~
.. currentmodule:: nonconformist

.. autosummary::
	:template: class.rst
	:toctree: _autosummary/

	icp.IcpClassifier
	icp.IcpRegressor


.. _ensemble_ref:

:mod:`acp`
----------
.. automodule:: nonconformist.acp
	:no-members:
	:no-inherited-members:

Classes
~~~~~~~
.. currentmodule:: nonconformist

.. autosummary::
	:template: class.rst
	:toctree: _autosummary/

	acp.AggregatedCp
	acp.RandomSubSampler
	acp.BootstrapSampler
	acp.CrossSampler


.. _evaluation_ref:

:mod:`evaluation`
-----------------
.. automodule:: nonconformist.evaluation
	:no-members:
	:no-inherited-members:

Classes
~~~~~~~
.. currentmodule:: nonconformist

.. autosummary::
	:template: class.rst
	:toctree: _autosummary/

	evaluation.ClassIcpCvHelper
	evaluation.RegIcpCvHelper

Functions
~~~~~~~~~
.. currentmodule:: nonconformist

.. autosummary::
	:toctree: _autosummary/

	evaluation.cross_val_score
	evaluation.run_experiment
	evaluation.reg_mean_errors
	evaluation.class_mean_errors
	evaluation.reg_mean_size
	evaluation.class_avg_c
	evaluation.class_one_c
	evaluation.class_mean_p_val
