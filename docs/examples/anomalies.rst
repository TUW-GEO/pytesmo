.. _anomaly-example-page:

The following example shows how you can use pytesmo to calculate climatology/anomaly
time series. Here we use the test data that is provided within this package.

Import all necessary dependencies:

.. code:: python

    from ascat.read_native.cdr import AscatSsmCdr # install the ascat package first https://github.com/TUW-GEO/ascat
    from pytesmo.time_series import anomaly as ts_anom

    import os
    import matplotlib.pyplot as plt

Set up reading ascat data (from pytesmo test data, make sure to clone it!)

.. code:: python

    testdata_path = os.path.join('..', '..', 'tests', 'test-data')
    ascat_data_folder = os.path.join(testdata_path, 'sat', 'ascat', 'netcdf', '55R22')
    ascat_grid_folder = os.path.join(testdata_path, 'sat', 'ascat', 'netcdf', 'grid')
    static_layers_folder = os.path.join(testdata_path, 'sat', 'h_saf', 'static_layer')
    #init the AscatSsmCdr reader with the paths

    ascat_reader = AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                               grid_filename='TUW_WARP5_grid_info_2_1.nc',
                               static_layer_path=static_layers_folder)

Read ASCAT SSM at one location (nearest neighbour lookup).

.. code:: python

    ascat_ts = ascat_reader.read(11.82935429,45.4731369)
    # plot soil moisture
    ascat_ts.data['sm'].plot()


You should get a time series:

.. image:: /_static/images/anomalies/anomalies_2_1.png

Calculate anomaly based on moving +- 17 day window

.. code:: python

    anomaly = ts_anom.calc_anomaly(ascat_ts.data['sm'], window_size=35)
    anomaly.plot()


You will get an anomaly time series with a moving average used for the seasonality:

.. image:: /_static/images/anomalies/anomalies_3_1.png

Calculate climatology

.. code:: python

    climatology = ts_anom.calc_climatology(ascat_ts.data['sm'])
    climatology.plot()

You will get the climatology time series:

.. image:: /_static/images/anomalies/anomalies_4_1.png

Calculate anomaly based on climatology

.. code:: python

    anomaly_clim = ts_anom.calc_anomaly(ascat_ts.data['sm'], climatology=climatology)
    anomaly_clim.plot()

You will get an anomaly time series which was found using a climatology:

.. image:: /_static/images/anomalies/anomalies_5_1.png
