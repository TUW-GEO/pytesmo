Introduction to pytesmo a python Toolbox for the Evaluation of Soil Moisture Observations
*****************************************************************************************

pytesmo is a package which aims it is not provide a standard library that can be used for the comparison and validation
of soil moisture datasets from different sources. It contains the code used by the 
`Satellite Soil Moisture Validation Tool For ASCAT <http://rs.geo.tuwien.ac.at/validation_tool/ascat.html>`_.



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

optional

* pykdtree https://github.com/storpipfugl/pykdtree

	which makes Nearest Neighbor search faster

Windows - new python users
--------------------------

For users with little python experience, using Windows, the easiest way to install everything but matplotlib-basemap is to install 
winpython from https://code.google.com/p/winpython/ and then download basemap from http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/
and add it to your winpython installation using the winpython Control Panel.

Just make sure that you download both for the same architecture (32/64 bit) and the same python version (2.7)

You can then add pytesmo-0.1.zip to your winpython installation with the winpython Control Panel

After that you can open spyder from the winpython installation directory and start testing pytesmo.

Windows and Linux
-----------------

If you already have a working python installation with the necessary packages just cd to the unzipped pytesmo-0.1 folder and use::
	
	python setup.py install

or if you'd rather use pip then do::
	
	pip install pytesmo
	





