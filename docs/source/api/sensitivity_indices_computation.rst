.. _sensitivity_indices_computation:

Sensitivity indices computation
===============================

Correlation coefficients
------------------------

.. automodule:: gsa_framework.sensitivity_methods.correlations
    :members: correlation_coefficients, get_corrcoef_num_iterations, get_corrcoef_interval_width

Sobol indices
-------------

Saltelli estimators
^^^^^^^^^^^^^^^^^^^

.. automodule:: gsa_framework.sensitivity_methods.saltelli_sobol
    :members: sobol_indices

Extended Fourier Amplitude Sensitivity Test (eFAST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gsa_framework.sensitivity_methods.extended_FAST
    :members: eFAST_indices

Delta moment-independent indices
--------------------------------

.. automodule:: gsa_framework.sensitivity_methods.delta
    :members: delta_indices

Feature importance with gradient boosting
-----------------------------------------

.. automodule:: gsa_framework.sensitivity_methods.gradient_boosting
    :members: xgboost_indices
