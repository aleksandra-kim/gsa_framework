.. _sensitivity_analysis:

Sensitivity analysis
====================

Method base class
-----------------
.. autoclass:: gsa_framework.sensitivity_analysis.method_base.SensitivityAnalysisMethod
    :members:

Correlation coefficients
------------------------

.. autoclass:: gsa_framework.sensitivity_analysis.correlations.Correlations
    :members:

Sobol indices
-------------

Saltelli estimators
^^^^^^^^^^^^^^^^^^^

.. autoclass:: gsa_framework.sensitivity_analysis.saltelli_sobol.SaltelliSobol
    :members:

Extended Fourier Amplitude Sensitivity Test (eFAST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gsa_framework.sensitivity_analysis.extended_FAST.eFAST
    :members:

Delta moment-independent indices
--------------------------------

.. autoclass:: gsa_framework.sensitivity_analysis.delta.Delta
    :members:

Feature importance with gradient boosting
-----------------------------------------

.. autoclass:: gsa_framework.sensitivity_analysis.gradient_boosting.GradientBoosting
    :members:
