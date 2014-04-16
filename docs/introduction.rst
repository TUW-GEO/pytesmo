Introduction to pytesmo a python Toolbox for the Evaluation of Soil Moisture Observations
*****************************************************************************************

pytesmo is a package which aims it is to provide a standard library that can be used for the comparison and validation
of geospatial time series datasets with a focus on soil moisture.

It contains an expanding collection of readers for different soil moisture datasets (see `Supported Datasets`_) as well as routines for comparing them.
Special classes in the module :mod:`pytesmo.grid.grids` provide easy nearest neighbor searching between datasets as well as 
the calculation of lookup tables of nearest neighbours. They also provide possibilities to easily read all 
grid points of a dataset in the correct order.

It contains the code used for the calculation of metrics by the 
`Satellite Soil Moisture Validation Tool For ASCAT <http://rs.geo.tuwien.ac.at/validation_tool/ascat.html>`_. See :mod:`pytesmo.metrics`.



Features
========

* easily read data from the `Supported Datasets`_
* anomaly calculation based on climatology or using a moving window see :mod:`pytesmo.time_series.anomaly`
* easy temporal matching of time series see :mod:`pytesmo.temporal_matching`
* multiple methods for scaling between different observation domains (CDF matching, linear regreesion, min-max matching) see :mod:`pytesmo.scaling`
* calculate standard metrics like correlation coefficients, RMSD, bias, 
  as well as more complex ones like triple collocation or MSE as a decomposition of the RMSD see :mod:`pytesmo.metrics`


Supported Datasets
==================

Soil moisture is observed using different methods and instruments, in this version several satellite datasets as well as in situ data are supported.

ERS
---

* ERS-1/2 AMI 25km SSM (Surface Soil Moisture)

  available from http://rs.geo.tuwien.ac.at/products

ASCAT
-----

* ASCAT SSM(Surface Soil Moisture) Time Series

  Available in binary format from http://rs.geo.tuwien.ac.at/products/
  
  Available in netCDF format from http://hsaf.meteoam.it/soil-moisture.php (H25 product)


* ASCAT SWI(Soil Water Index) Time Series

  Available in binary format from http://rs.geo.tuwien.ac.at/products/


Data from the International Soil Moisture Network (ISMN)
--------------------------------------------------------

ISMN data can be downloaded for free after registration from http://ismn.geo.tuwien.ac.at/

In case of the ISMN, 3 different formats are provided:

* Variables stored in separate files (CEOP formatted)
	
	this format is supported 100% and should work with all examples
	
* Variables stored in separate files (Header+values)
	
	this format is supported 100% and should work with all examples	
	
* CEOP Reference Data Format

	this format can be read with the readers in pytesmo.io.ismn.readers but
	is not supported for more complex queries since it is missing sensor information
	, only provides soil moisture and soil temperature and contains several depths in 
	one file.
	
If you downloaded ISMN data in one of the supported formats in the past it can be that station
names are not recognized correctly because they contained the '_' character which is supposed to be
the seperator. If you experience problems because of this please download new data from the ISMN since
this issue should be fixed.		


Installation
============

Prerequisites
--------------

In order to enjoy all pytesmo features python version 2.7.5 with the following packages has to be installed

* numpy >= 1.7.0 http://www.numpy.org/
* pandas >= 0.11.0 http://pandas.pydata.org/
* scipy >= 0.12.0 http://www.scipy.org/
* statsmodels >= 0.4.3 http://statsmodels.sourceforge.net/
* matplotlib >= 1.2.0 http://matplotlib.org/
* matplotlib - basemap >= 1.0.5 http://matplotlib.org/basemap/
* netCDF4 >= 1.0.1 https://pypi.python.org/pypi/netCDF4

optional

* pykdtree https://github.com/storpipfugl/pykdtree

	which makes Nearest Neighbor search faster

Windows - new python users
--------------------------

The Anaconda python disribution https://store.continuum.io/cshop/anaconda/ is a good choice since it includes all dependencies needed for pytesmo.
Currently only the 32bit Anaconda python distributin is supported.
After Anaconda is installed open the "Anaconda Command Prompt" and type in the command

pip install pytesmo

This should install pytesmo. If you are behind a proxy server please set the environment variables http_proxy and https_proxy.

Another easy way to install everything but matplotlib-basemap and netCDF4 is to install 
winpython from https://code.google.com/p/winpython/ and then download basemap from 
http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/ 
and netCDF4 from https://code.google.com/p/netcdf4-python/
and add it to your winpython installation using the winpython Control Panel.

Just make sure that you download both for the same architecture (32/64 bit) and the same python version (2.7.x)

After that you can also use the winpython control panel to add the relevant pytesmo `Windows binaries`_

After that you can open spyder or the Ipython notebook from the winpython installation directory and start testing pytesmo.

If you want a system installation of python download the following files and install them in order.

* Python 2.7.x windows installer from http://python.org/download/
* Scipy-stack installer from http://www.lfd.uci.edu/~gohlke/pythonlibs/
* netCDF4 installer from http://www.lfd.uci.edu/~gohlke/pythonlibs/
* pytesmo windows binary

Windows binaries
----------------

pytesmo windows binaries are available for 32 and 64 bit systems:

* 32-bit http://rs.geo.tuwien.ac.at/validation_tool/pytesmo/pytesmo-0.1.2/pytesmo-0.1.2.win32-py2.7.exe
* 64-bit http://rs.geo.tuwien.ac.at/validation_tool/pytesmo/pytesmo-0.1.2/pytesmo-0.1.2.win-amd64-py2.7.exe


Linux
-----

If you already have a working python installation with the necessary packages download and unpack the pytesmo source package which is available from

* Pypi https://pypi.python.org/pypi/pytesmo

just change the active directory to the unpacked pytesmo-0.1.1 folder and use the following command in the command line::
	
	python setup.py install

or if you'd rather use pip then use the command::
	
	pip install pytesmo
	
Contribute
==========

If you would like to help this project by improving the documentation, 
providing examples of how you use it or by extending the functionality of pytesmo we would be very happy.

Please browse the source code which is available at http://github.com/TUW-GEO/pytesmo

Feel free to contact `Christoph Paulik <http://rs.geo.tuwien.ac.at/our-team/christoph-paulik/>`_ in case of any questions or requests.




