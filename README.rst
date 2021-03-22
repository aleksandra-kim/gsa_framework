Welcome to gsa_framework!
=========================

Python package gsa_framework is aimed at providing interface for Global Sensitivity Analysis (GSA). It consists of the following modules:

* sampling
* * random
* * latin hypercube
* * Sobol' quasi-random sequences
* * Saltelli design
* * custom inputs, e.g. obtained from real data / measurements

* sensitivity indices computation
* * Pearson and Spearman correlation coefficients
* * Sobol firt and total order
* * Extended FAST
* * Delta moment-independent indices
* * Feature importances from gradient boosted trees with XGBoost

* models
* * test functions
* * life cycle assessment
* * custom models

* sensitivity analysis that links all of the above for each sensitivity method

* additional components to support reliability of GSA
* * GSA results validation
* * Convergence of sensitivity indices
* * Robustness with bootstrapping

This package is part of the doctoral work of Aleksandra Kim at Paul Scherrer Institute and ETH Zurich.

For detailed API docs, see `the versioned API website <https://gsa-framework.readthedocs.io/>`_.
