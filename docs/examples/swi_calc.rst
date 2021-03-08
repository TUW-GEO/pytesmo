.. _ascat-swi-example-page:

This example script shows how we can use the expontential filter from pytesmo
to calculate SWI from ASCAT Soil Moisture.

Import all necessary dependencies:

.. code:: python

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore') # some import warnings are expected and ignored
        # install the ascat package first https://github.com/TUW-GEO/ascat
        from ascat.read_native.cdr import AscatSsmCdr
    from pytesmo.time_series.filters import exp_filter

    import os
    import matplotlib.pyplot as plt

Set up the ascat reader:

.. code:: python

    testdata_path = os.path.join('..', '..', 'tests', 'test-data')
    ascat_data_folder = os.path.join(testdata_path, 'sat', 'ascat', 'netcdf', '55R22')
    ascat_grid_folder = os.path.join(testdata_path, 'sat', 'ascat', 'netcdf', 'grid')
    static_layers_folder = os.path.join(testdata_path, 'sat', 'h_saf', 'static_layer')
    #init the AscatSsmCdr reader with the paths
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore') # some warnings are expected and ignored

        ascat_reader = AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                                   grid_filename='TUW_WARP5_grid_info_2_1.nc',
                                   static_layer_path=static_layers_folder)

Read a point in Northern Italy:

.. code:: python

    # point at (11,45)
    ascat_ts = ascat_reader.read(2302069, mask_ssf=True, mask_frozen_prob=80, mask_snow_prob=20)
    ascat_ts.plot()

.. image:: /_static/images/swi_calculation/output_1_1.png

Apply the exponential filter to calculate SWI from SM:

.. code:: python

    # Drop NA measurements
    ascat_sm_ts = ascat_ts.data[['sm', 'sm_noise']].dropna()

    # Get julian dates of time series
    jd = ascat_sm_ts.index.to_julian_date().values

    # Calculate SWI T=10
    ascat_sm_ts['swi_t10'] = exp_filter(ascat_sm_ts['sm'].values, jd, ctime=10)
    ascat_sm_ts['swi_t50'] = exp_filter(ascat_sm_ts['sm'].values, jd, ctime=50)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ascat_sm_ts['sm'].plot(ax=ax, alpha=0.4, marker='o',color='#00bfff', label='SSM')
    ascat_sm_ts['swi_t10'].plot(ax=ax, lw=2,label='SWI T=10')
    ascat_sm_ts['swi_t50'].plot(ax=ax, lw=2,label='SWI T=50')
    plt.legend()

.. image:: /_static/images/swi_calculation/output_2_1.png
