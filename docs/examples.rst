.. _examples-page:

Examples
********
All here shown examples are also available as ipython notebooks in ``pytesmo/docs/examples``.

Within the following examples we use two reader packages that should be installed
first:

.. code-block:: bash

    pip install ascat
    pip install ismn

.. include::
   examples/validation_framework.rst

Calculating anomalies and climatologies
=======================================

This Example script reads and plots ASCAT H25 SSM data. The :mod:`pytesmo.time_series.anomaly` module
is then used to calculate anomalies and climatologies of the time series.
It can be found in the /examples folder of the pytesmo package under the name anomalies.py

.. include:: /examples/anomalies.rst

Calculation of the Soil Water Index
===================================

The Soil Water Index(SWI) which is a method to estimate root zone soil moisture can be calculated from Surface Soil Moisture(SSM) using an exponential Filter. For more details see this publication of `C.Abergel et.al <http://www.hydrol-earth-syst-sci.net/12/1323/2008/>`_. The following example shows how to calculate the SWI for two T values from ASCAT H25 SSM.

.. include:: /examples/swi_calc.rst

Triple collocation and triple collocation based scaling
=======================================================

This example shows how to use the triple collocation routines in the :mod:`pytesmo.metrics` module.
It also is a crash course to the theory behind triple collocation and links to relevant publications.


.. include:: /examples/Triple collocation.rst

Comparing ASCAT and insitu data from the ISMN without the validation framework
==============================================================================

This example program loops through all insitu stations that measure soil moisture with a depth between 0 and 0.1m
it then finds the nearest ASCAT grid point and reads the ASCAT data. After temporal matching and scaling using linear CDF matching it computes
several metrics, like the correlation coefficients(Pearson's, Spearman's and Kendall's), Bias, RMSD as well as the Nash–Sutcliffe model efficiency coefficient.

It also shows the usage of the :mod:`pytesmo.df_metrics` module.

It is stopped after 2 stations to not take to long to run and produce a lot of plots

It can be found in the docs/examples folder of the pytesmo package under the name compare_ISMN_ASCAT.py.

.. include:: /examples/compare_ASCAT_ISMN.rst
