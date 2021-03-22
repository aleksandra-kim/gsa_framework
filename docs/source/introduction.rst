.. _introduction:

Python package gsa_framework is aimed at providing interface for Global Sensitivity Analysis (GSA) - the study of how uncertainty in the output of a model (numerical or otherwise) can be apportioned to different sources of uncertainty in the model input :cite:`saltelli2004sensitivity`. For each GSA method, it combines its typical components in a consistent way, while preserving modularity. The components include sampling, model runs and computation of sensitivity indices, as well as optional modules that support reliable and efficient GSA by means of convergence, robustness and validation analyses.

Sampling methods
* random
* latin hypercube
* Sobol' quasi-random sequences
* Saltelli design :cite:`saltelli2010variance`
* custom inputs for the case when input data is independent from sampling and is obtained from measurements

Sensitivity methods

* Pearson and Spearman correlation coefficients
* Sobol firt and total order with Saltelli estimators :cite:`saltelli2010variance`
* Extended FAST :cite:`saltelli1999quantitative`
* Borgonovo delta moment-independent indices :cite:`borgonovo2007new`
* Feature importances from gradient boosted trees with XGBoost :cite:`chen2016xgboost`

Models

* test functions, such as Morris, borehole, wingweight, OTLcircuit, piston, Moon Sobol-Levitan, Sobol G and G star functions
* life cycle assessment model
* custom models

Additional components to support reliability of GSA

* GSA results validation
* Convergence of sensitivity indices
* Robustness with bootstrapping

This package is part of the doctoral work of Aleksandra Kim at Paul Scherrer Institute and ETH Zurich.
