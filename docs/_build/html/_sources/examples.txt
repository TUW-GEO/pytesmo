.. _examples-page:

Examples
********


Reading and plotting ASCAT data
===============================

This example program reads and plots ASCAT SSM and SWI data with different masking options. 
It can be found in the /bin folder of the pytesmo package under the name plot_ASCAT_data.py.

.. include::
   plot_ascat_data.rst


	
Reading and plotting data from the ISMN
=======================================

This example program chooses a random Network and Station and plots the first variable,depht,sensor
combination. To see how to get data for a variable from all stations see the next example. 
 
It can be found in the /bin folder of the pytesmo package under the name plot_ISMN_data.py.

.. include::
   plot_ISMN.rst

Comparing ASCAT and insitu data from the ISMN
=============================================

This example program loops through all insitu stations that measure soil moisture with a depth between 0 and 0.1m
it then finds the nearest ASCAT grid point and reads the ASCAT data. After temporal matching and scaling using linear CDF matching it computes
several metrics, like the correlation coefficients(Pearson's, Spearman's and Kendall's), Bias, RMSD as well as the Nash–Sutcliffe model efficiency coefficient.

It is stopped after 2 stations to not take to long to run and produce a lot of plots
 
It can be found in the /bin folder of the pytesmo package under the name compare_ISMN_ASCAT.py.

.. include::
   compare_ASCAT_ISMN.rst	
	
	