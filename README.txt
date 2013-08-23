===========
pytesmo - python Toolbox for the Evaluation of Soil Moisture Observations
===========

pytesmo is a package which aims to provide easy reading functions for soil moisture data, as well as 
functions for comparing these datasets against each other.


Features
========

* easily read data from the `Supported Datasets`_
* anomaly calculation based on climatology or using a moving window see :mod:`pytesmo.anomaly`
* easy temporal matching of time series see :mod:`pytesmo.temporal_matching`
* multiple methods for scaling between different observation domains (CDF matching, linear regreesion, min-max matching) see :mod:`pytesmo.scaling`
* calculate standard metrics like correlation coefficients, RMSD, bias, 
  as well as more complex ones like triple collocation or MSE as a decomposition of the RMSD see :mod:`pytesmo.metrics`


Supported Datasets
==================

Soil moisture is observed using different methods and instruments, in this version the following datasets are supported.

Remotely sensed products
------------------------

Right now only soil moisture observations made with the ASCAT sensor on board the METOP satellites are
supported but we are working on support for more products.

ASCAT
~~~~~

* ASCAT SSM(Surface Soil Moisture)
* ASCAT SWI(Soil Water Index)

which can both be downloaded for free after registration at http://rs.geo.tuwien.ac.at/products/

insitu obervations
------------------

Data from the International Soil Moisture Network (ISMN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ISMN data can be downloaded for free after registration from http://ismn.geo.tuwien.ac.at/

Complete Documentation
======================

The Documentation can be found at http://rs.geo.tuwien.ac.at/validation_tool/pytesmo/