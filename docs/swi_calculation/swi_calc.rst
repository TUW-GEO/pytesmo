
.. code:: python

    import os

    import matplotlib.pyplot as plt

    from pytesmo.time_series.filters import exp_filter
    import ascat


    ascat_folder = os.path.join('/media', 'sf_R', 'Datapool_processed',
                                'WARP', 'WARP5.5', 'IRMA0_WARP5.5_P2',
                                'R1', '080_ssm', 'netcdf')
    ascat_grid_folder = os.path.join('/media', 'sf_R',
                                     'Datapool_processed', 'WARP',
                                     'ancillary', 'warp5_grid')

    # init the ASCAT_SSM reader with the paths

    # ascat_folder is the path in which the cell files are
    # located e.g. TUW_METOP_ASCAT_WARP55R12_0600.nc
    # ascat_grid_folder is the path in which the file
    # TUW_WARP5_grid_info_2_1.nc is located

    # let's not include the orbit direction since it is saved as 'A'
    # or 'D' it can not be plotted

    # the AscatH25_SSM class automatically detects the version of data
    # that you have in your ascat_folder. Please do not mix files of
    # different versions in one folder

    ascat_SSM_reader = ascat.AscatH25_SSM(ascat_folder, ascat_grid_folder,
                                          include_in_df=['sm', 'sm_noise',
                                                         'ssf', 'proc_flag'])

.. code:: python

    ascat_ts = ascat_SSM_reader.read_ssm(gpi, mask_ssf=True, mask_frozen_prob=10,
                                         mask_snow_prob=10)
    ascat_ts.plot()


.. image:: swi_calculation/output_2_1.png


.. code:: python

    # Drop NA measurements
    ascat_sm_ts = ascat_ts.data[['sm', 'sm_noise']].dropna()

    # Get julian dates of time series
    jd = ascat_sm_ts.index.to_julian_date().get_values()

    # Calculate SWI T=10
    ascat_sm_ts['swi_t10'] = exp_filter(ascat_sm_ts['sm'].values, jd, ctime=10)
    ascat_sm_ts['swi_t50'] = exp_filter(ascat_sm_ts['sm'].values, jd, ctime=50)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ascat_sm_ts['sm'].plot(ax=ax, alpha=0.4, marker='o',color='#00bfff', label='SSM')
    ascat_sm_ts['swi_t10'].plot(ax=ax, lw=2,label='SWI T=10')
    ascat_sm_ts['swi_t50'].plot(ax=ax, lw=2,label='SWI T=50')
    plt.legend()



.. image:: swi_calculation/output_3_1.png
