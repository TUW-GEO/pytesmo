pytesmo a Python Toolbox for the Evaluation of Soil Moisture Observations
*************************************************************************

pytesmo is a package which aims it is to provide a standard library that can be
used for the comparison and validation of geospatial time series datasets with a
focus on soil moisture.

It contains an expanding collection of readers for different soil moisture
datasets (see `Supported Datasets`_) as well as routines for comparing them.
Special classes in the module :mod:`pytesmo.grid.grids` provide easy nearest
neighbor searching between datasets as well as the calculation of lookup tables
of nearest neighbours. They also provide possibilities to easily read all grid
points of a dataset in the correct order.

It contains the code used for the calculation of metrics by the `Satellite Soil
Moisture Validation Tool For ASCAT
<http://rs.geo.tuwien.ac.at/validation_tool/ascat.html>`_. See :mod:`pytesmo.metrics`.

Features
========

* easily read data from the `Supported Datasets`_
* anomaly calculation based on climatology or using a moving window see
  :mod:`pytesmo.time_series.anomaly`
* easy temporal matching of time series see :mod:`pytesmo.temporal_matching`
* multiple methods for scaling between different observation domains (CDF
  matching, linear regression, min-max matching) see :mod:`pytesmo.scaling`
* calculate standard metrics like correlation coefficients, RMSD, bias, as well
  as more complex ones like :ref:`triple-collocation-example` or MSE as a
  decomposition of the RMSD see :mod:`pytesmo.metrics`

Supported Datasets
==================

Soil moisture is observed using different methods and instruments, in this
version several satellite datasets as well as in situ data are supported.

ASCAT
-----

Ascat data is supported via the `ascat package
<https://github.com/TUW-GEO/ascat>`_. If you want to use this data then please
follow the `installation instructions
<https://github.com/TUW-GEO/ascat#installation>`_.

H-SAF image products
--------------------

H-SAF data is also supported via the `ascat package
<https://github.com/TUW-GEO/ascat>`_. If you want to use this data then please
follow the `installation instructions
<https://github.com/TUW-GEO/ascat#installation>`_.

SMAP
----

SMAP data is supported via the `smap_io package
<https://github.com/TUW-GEO/smap_io>`_. If you want to use this data then please
follow the `installation instructions
<https://github.com/TUW-GEO/smap_io#installation>`_.

GLDAS Noah
----------

GLDAS Noah data is supported via the `gldas package
<https://github.com/TUW-GEO/gldas>`_. If you want to use this data then please
follow the `installation instructions
<https://github.com/TUW-GEO/gldas#installation>`_.

ERA Interim
-----------

ERA Interim data is supported via the `ecmwf_models package
<https://github.com/TUW-GEO/ecmwf_models>`_. If you want to use this data then please
follow the `installation instructions
<https://github.com/TUW-GEO/ecmwf_models#installation>`_.

ERS
---

* ERS-1/2 AMI 25km SSM (Surface Soil Moisture)

  available from http://rs.geo.tuwien.ac.at/products

To read the ERS please also install the `ascat package
<https://github.com/TUW-GEO/ascat>`_.

Data from the International Soil Moisture Network (ISMN)
--------------------------------------------------------

ISMN data can be downloaded for free after registration from the `ISMN Website
<http://ismn.geo.tuwien.ac.at/>`_

In case of the ISMN, two different formats are provided:

* Variables stored in separate files (CEOP formatted)

	this format is supported 100% and should work with all examples

* Variables stored in separate files (Header+values)

	this format is supported 100% and should work with all examples

If you downloaded ISMN data in one of the supported formats in the past it can
be that station names are not recognized correctly because they contained the
'_' character which is supposed to be the separator. If you experience problems
because of this please download new data from the ISMN since this issue should
be fixed.


Installation
============

Necessary Python packages
-------------------------

In order to enjoy all pytesmo features Python 2.7, 3.3, 3.4 or 3.5 with the following
packages should be installed.

* numpy >= 1.7.0 http://www.numpy.org/
* pandas >= 0.11.0 http://pandas.pydata.org/
* scipy >= 0.12.0 http://www.scipy.org/
* netCDF4 >= 1.0.1 https://pypi.python.org/pypi/netCDF4
* pygeogrids https://pypi.python.org/pypi/pygeogrids
* matplotlib >= 1.2.0 http://matplotlib.org/

optional

* pykdtree https://github.com/storpipfugl/pykdtree

	which makes Nearest Neighbor search faster (Linux only)

* pyresample https://github.com/pytroll/pyresample

	for resampling of irregular images onto a regular grid for e.g. plotting

* matplotlib - basemap >= 1.0.5 http://matplotlib.org/basemap/

  for plotting maps of ISMN stations, maps in general

How to install python packages
------------------------------

If you have no idea of how to install python packages then I'll try to give a
short overview and provide links to resources that can explain the process.

The recommended way of installing python packages is using `pip
<https://pip.pypa.io/en/latest/installing.html>`_ which downloads the package
you want from the `python package repository Pypi <https://pypi.python.org/>`_
and installs it if possible. For more complex packages that depend upon a C or
Fortran library like netCDF4 or pybufr-ecmwf installation instructions are
provided on the package website. Try to install these packages with Anaconda_
whenever possible.


conda
-----

It is easiest to install packages that depend on C or Fortran libraries with
conda. See http://conda.pydata.org/docs/ on how to use it.

The following installation script using ``conda`` should get you started on both
Windows and Linux.

.. code::

   conda create -n pytesmo -c conda-forge python=2.7 numpy scipy pandas netCDF4 cython pytest pip matplotlib pyproj
   source activate test
   pip install pygeogrids
   pip install pyresample
   pip install pytesmo

Windows
-------

.. note::

    If you are using windows and conda is missing a package then always check
    http://www.lfd.uci.edu/~gohlke/pythonlibs/ to see if there is already a
    precompiled .exe or .whl file for you to easily install.


Windows binaries
----------------

pytesmo windows wheels are available for 32 and 64 bit systems from `pypi
<https://pypi.python.org/pypi/pytesmo>`_ so using::

  pip install pytesmo

should generally work on windows if the dependencies are installed.


Linux
-----

If you already have a working python installation with the necessary packages
download and unpack the pytesmo source package which is available from

* Pypi https://pypi.python.org/pypi/pytesmo

just change the active directory to the unpacked pytesmo folder and use
the following command in the command line::

	python setup.py install

or if you'd rather use pip then use the command::

	pip install pytesmo

Contribute
==========

If you would like to help this project by improving the documentation, providing
examples of how you use it or by extending the functionality of pytesmo we would
be very happy.

Please browse the source code which is available at http://github.com/TUW-GEO/pytesmo

Feel free to contact `Christoph Paulik
<http://rs.geo.tuwien.ac.at/our-team/christoph-paulik/>`_ in case of any
questions or requests.
