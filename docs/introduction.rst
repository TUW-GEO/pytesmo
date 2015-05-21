Introduction to pytesmo a python Toolbox for the Evaluation of Soil Moisture Observations
*****************************************************************************************

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

ERS
---

* ERS-1/2 AMI 25km SSM (Surface Soil Moisture)

  available from http://rs.geo.tuwien.ac.at/products

ASCAT
-----

* ASCAT SSM(Surface Soil Moisture) Time Series

  Available in netCDF format from `H-SAF
  <http://hsaf.meteoam.it/soil-moisture.php>`_ (H25 product)


* ASCAT SWI(Soil Water Index) Time Series

  Available in binary format from `TU Wien <http://rs.geo.tuwien.ac.at/products/>`_

H-SAF image products
--------------------

`H-SAF <http://hsaf.meteoam.it/soil-moisture.php>`_ provides three different
image products:

* SM OBS 1 - H07 - Large scale surface soil moisture by radar scatterometer in
  BUFR format over Europe
* SM OBS 2 - H08 - Small scale surface soil moisture by radar scatterometer in
  BUFR format over Europe
* SM DAS 2 - H14 - Profile index in the roots region by scatterometer data
  assimilation in GRIB format, global

They are available after registration from the `H-SAF Website
<http://hsaf.meteoam.it/soil-moisture.php>`_


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

In order to enjoy all pytesmo features Python 2.7, 3.3 or 3.4 with the following
packages should be installed.

* numpy >= 1.7.0 http://www.numpy.org/
* pandas >= 0.11.0 http://pandas.pydata.org/
* scipy >= 0.12.0 http://www.scipy.org/
* statsmodels >= 0.4.3 http://statsmodels.sourceforge.net/
* netCDF4 >= 1.0.1 https://pypi.python.org/pypi/netCDF4
* pygeogrids https://pypi.python.org/pypi/pygeogrids
* matplotlib >= 1.2.0 http://matplotlib.org/

optional

* pybufr-ecmwf https://code.google.com/p/pybufr-ecmwf/

	for reading the H-SAF H07 and H08 products in BUFR Format. As far as I know
	this will only work on Linux or in Cygwin but I have no experience using it on
	Windows. pybufr-ewmwf downloads and installs the BUFR library from the ECMWF
	website.

* pygrib https://code.google.com/p/pygrib/

	for reading the H-SAF H25 product

* pykdtree https://github.com/storpipfugl/pykdtree

	which makes Nearest Neighbor search faster

* pyresample https://code.google.com/p/pyresample/

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
provided on the package website.

.. note::

    If you are using windows always check
    http://www.lfd.uci.edu/~gohlke/pythonlibs/ to see if there is already a
    precompiled .exe or .whl file for you to easily install.

Windows
-------

A relatively easy way to install everything but matplotlib-basemap and netCDF4
is to install `WinPython <https://winpython.github.io/>`_ and then
download basemap, netCDF4 and, if you want to read the H14 product, the pygrib
installer from http://www.lfd.uci.edu/~gohlke/pythonlibs/. Add them to your
winpython installation using the winpython Control Panel.

Just make sure that you download both for the same architecture (32/64 bit) and
the same Python version that you used for WinPython.

After that you can also use the winpython control panel to add the relevant
pytesmo `Windows binaries`_

After that you can open spyder or the Ipython notebook from the winpython
installation directory and start testing pytesmo.

If you want a system installation of python download the following files and
install them in order.

* Python windows installer from http://python.org/download/
* Scipy-stack installer from http://www.lfd.uci.edu/~gohlke/pythonlibs/
* netCDF4 installer from http://www.lfd.uci.edu/~gohlke/pythonlibs/
* pygrib installer from http://www.lfd.uci.edu/~gohlke/pythonlibs/
* pytesmo windows binaries can be installed with pip

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

just change the active directory to the unpacked pytesmo-0.2.0 folder and use
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
