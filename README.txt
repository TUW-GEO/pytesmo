Introduction to pytesmo a python Toolbox for the Evaluation of Soil Moisture Observations
*****************************************************************************************

pytesmo is a package which aims it is to provide a standard library that can be used for the comparison and validation
of geospatial time series datasets with a focus on soil moisture.

It contains an expanding collection of readers for different soil moisture datasets (see `Supported Datasets`_) as well as routines for comparing them.
Special classes in the module :mod:`pytesmo.grid` provide easy nearest neighbor searching between datasets as well as 
the calculation of lookup tables of nearest neighbours. They also provide possibilities to easily read all 
grid points of a dataset in the correct order.

It contains the code used for the calculation of metrics by the 
`Satellite Soil Moisture Validation Tool For ASCAT <http://rs.geo.tuwien.ac.at/validation_tool/ascat.html>`_. See :mod:`pytesmo.metrics`.



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

In this version soil moisture observations made with the following instruments are supported:

ERS
~~~

* ERS-1/2 AMI 25km SSM (Surface Soil Moisture)
available from http://rs.geo.tuwien.ac.at/products

ASCAT
~~~~~

* ASCAT SSM(Surface Soil Moisture) Time Series
Available in custom format from http://rs.geo.tuwien.ac.at/products/
Available in netCDF format from http://hsaf.meteoam.it/soil-moisture.php


* ASCAT SWI(Soil Water Index) Time Series
Available in custom format from http://rs.geo.tuwien.ac.at/products/

insitu obervations
------------------

Data from the International Soil Moisture Network (ISMN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ISMN data can be downloaded for free after registration from http://ismn.geo.tuwien.ac.at/