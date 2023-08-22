
Introduction
************

Pytesmo provides a number of tools that can be used to validate satellite
soil moisture (and other climate variables). The pytesmo validation framework
combines these tools and also uses functions from some of our other packages.
See e.g. the Supported Products. for reader packages that work within pytesmo
or the `pygeogrids <https://github.com/TUW-GEO/pygeogrids>`__ python package for nearest
neighbor searching between datasets, calculation of lookup tables, and
reading all grid points of a dataset in the correct order.

Features
========

* easily read data from the Supported Products.
* anomaly calculation based on climatology or using a moving window see
  :mod:`pytesmo.time_series.anomaly`
* easy temporal matching of time series see :mod:`pytesmo.temporal_matching`
* multiple methods for scaling between different observation domains (CDF
  matching, linear regression, min-max matching) see :mod:`pytesmo.scaling`
* calculate standard metrics like correlation coefficients, RMSD, bias, as well
  as more complex ones like :ref:`triple-collocation-example` or MSE as a
  decomposition of the RMSD see :mod:`pytesmo.metrics`

Notebooks
=========
The following documentation is created from ipython notebooks in ``pytesmo/docs/examples``.
The notebooks can be run interactively and the results can be reproduced locally using `jupyter <http://jupyter.org/>`__.
Some of the examples require the packages `ascat` and `ismn`, which can be installed with `pip`.
