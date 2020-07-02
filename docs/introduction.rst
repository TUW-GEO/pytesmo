
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

Prerequisites
=============

Necessary Python packages
-------------------------

In order to enjoy all pytesmo features, a recent Python 3 installtation with the
conda/pip packages listed in :download:`requirements.txt <../requirements.txt>`
should be installed:

Some packages are optional:

* pykdtree: https://github.com/storpipfugl/pykdtree

	which makes Nearest Neighbor search faster

* pyresample: https://github.com/pytroll/pyresample

	for resampling of irregular images onto a regular grid for e.g. plotting

* matplotlib with cartopy/basemap: http://matplotlib.org

  for plotting maps of ISMN stations, maps in general