.. _examples-page:

Examples
********

Reading and plotting data from the ISMN
=======================================

This example program chooses a random Network and Station and plots the first variable,depth,sensor
combination. To see how to get data for a variable from all stations see the next example.

It can be found in the /examples folder of the pytesmo package under the name plot_ISMN_data.py.

.. include::
   plot_ISMN.rst


Calculating anomalies and climatologies
=======================================

This Example script reads and plots ASCAT H25 SSM data. The :mod:`pytesmo.time_series.anomaly` module
is then used to calculate anomalies and climatologies of the time series.
It can be found in the /examples folder of the pytesmo package under the name anomalies.py

.. include::
   anomalies.rst

Calculation of the Soil Water Index
===================================

The Soil Water Index(SWI) which is a method to estimate root zone soil moisture can be calculated from Surface Soil Moisture(SSM) using an exponential Filter. For more details see this publication of `C.Abergel et.al <http://www.hydrol-earth-syst-sci.net/12/1323/2008/>`_. The following example shows how to calculate the SWI for two T values from ASCAT H25 SSM.

.. include::
   swi_calculation/swi_calc.rst


.. include::
   validation_framework.rst


Triple collocation and triple collocation based scaling
=======================================================

This example shows how to use the triple collocation routines in the :mod:`pytesmo.metrics` module.
It also is a crash course to the theory behind triple collocation and links to relevant publications.


.. include::
   Triple collocation.rst

Comparing ASCAT and insitu data from the ISMN without the validation framework
==============================================================================

This example program loops through all insitu stations that measure soil moisture with a depth between 0 and 0.1m
it then finds the nearest ASCAT grid point and reads the ASCAT data. After temporal matching and scaling using linear CDF matching it computes
several metrics, like the correlation coefficients(Pearson's, Spearman's and Kendall's), Bias, RMSD as well as the Nash–Sutcliffe model efficiency coefficient.

It also shows the usage of the :mod:`pytesmo.df_metrics` module.

It is stopped after 2 stations to not take to long to run and produce a lot of plots

It can be found in the /examples folder of the pytesmo package under the name compare_ISMN_ASCAT.py.

.. include::
   compare_ASCAT_ISMN.rst
