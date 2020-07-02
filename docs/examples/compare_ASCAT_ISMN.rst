.. _ascat-ismn-example-page:

Here we show an example validation of ASCAT SM against ISMN insitu observations.

Import all necessary functions

.. code:: python

    import ismn.interface as ismn # install the ISMN package first https://github.com/TUW-GEO/ismn
    from ascat.read_native.cdr import AscatSsmCdr # install the ascat package first https://github.com/TUW-GEO/ascat

    import pytesmo.temporal_matching as temp_match
    import pytesmo.scaling as scaling
    import pytesmo.df_metrics as df_metrics
    import pytesmo.metrics as metrics
    
    import os
    import matplotlib.pyplot as plt

    from pytesmo import testdata_path

Create the ascat reader:

.. code:: python

    ascat_data_folder = os.path.join(testdata_path, 'sat', 'ascat', 'netcdf', '55R22')
    ascat_grid_folder = os.path.join(testdata_path, 'sat', 'ascat', 'netcdf', 'grid')
    static_layers_folder = os.path.join(testdata_path, 'sat', 'h_saf', 'static_layer')
    #init the AscatSsmCdr reader with the paths

    ascat_reader = AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                               grid_filename='TUW_WARP5_grid_info_2_1.nc',
                               static_layer_path=static_layers_folder)

Create the ismn reader:

.. code:: python

    #set path to ISMN data
    path_to_ismn_data = os.path.join(testdata_path, 'ismn', 'multinetwork', 'header_values')

    #Initialize reader
    ISMN_reader = ismn.ISMN_Interface(path_to_ismn_data)

Here we create a temporary directory to write some results into. Also we select
new names for the ascat and ismn soil moisture data columns.

.. code:: python

    from tempfile import mkdtemp

    out_path = mkdtemp()
    print('Plots are created in:', out_path)

    label_ascat='sm'
    label_insitu='insitu_sm'

Now we loop through the ISMN stations and at each statione we read the nearest ASCAT pixel.
Temporal matching is then performed and the data is scaled (bias correction).

.. code:: python

    import matplotlib.pyplot as plt

    i = 0

    # this loops through all stations that measure soil moisture
    for station in ISMN_reader.stations_that_measure('soil moisture'):
        
        # this loops through all time series of this station that measure soil moisture
        # between 0 and 0.1 meters
        for t, ISMN_time_series in enumerate(station.data_for_variable('soil moisture',
                                                min_depth=0, max_depth=0.1)):
            
            ascat_time_series = ascat_reader.read(ISMN_time_series.longitude,
                                                  ISMN_time_series.latitude,
                                                  mask_ssf=True,
                                                  mask_frozen_prob = 5,
                                                  mask_snow_prob = 5)
            
            # drop nan values before doing any matching
            ascat_sm = ascat_time_series.data[['sm']].dropna()
            ismn_sm = ISMN_time_series.data[['soil moisture']].dropna()
            
            # rename the soil moisture column in ISMN_time_series.data to insitu_sm
            # to clearly differentiate the time series when they are plotted together
            ismn_sm.rename(columns={'soil moisture':label_insitu}, inplace=True)
            
            # get ISMN data that was observerd within +- 1 hour(1/24. day) of the ASCAT observation
            # do not include those indexes where no observation was found
            matched_data = temp_match.matching(ascat_sm,ismn_sm, window=1/24.)
            # matched ISMN data is now a dataframe with the same datetime index
            # as ascat_time_series.data and the nearest insitu observation
            
            # the plot shows that ISMN and ASCAT are observed in different units
            fig1, ax1 = plt.subplots()
            matched_data.plot(figsize=(15,5),secondary_y=[label_ascat],
                              title='temporally merged data', ax=ax1)
            fig1.show()
            fig1.savefig(os.path.join(out_path, f'compare_ASCAT_ISMN_{i}_{t}_1.png'))

            
            # this takes the matched_data DataFrame and scales all columns to the
            # column with the given reference_index, in this case in situ
            scaled_data = scaling.scale(matched_data, method='lin_cdf_match',
                                        reference_index=1)
            
            # now the scaled ascat data and insitu_sm are in the same space
            fig2, ax2 = plt.subplots()
            scaled_data.plot(figsize=(15,5), title='scaled data', ax=ax2)
            fig2.show()
            fig2.savefig(os.path.join(out_path, f'compare_ASCAT_ISMN_{i}_{t}_2.png'))

            fig3, ax3 = plt.subplots()
            ax3.scatter(scaled_data[label_ascat].values, scaled_data[label_insitu].values)
            ax3.set_xlabel(label_ascat)
            ax3.set_ylabel(label_insitu)
            fig3.show()
            fig3.savefig(os.path.join(out_path, f'compare_ASCAT_ISMN_{i}_{t}_3.png'))
            
            # calculate correlation coefficients, RMSD, bias, Nash Sutcliffe
            x, y = scaled_data[label_ascat].values, scaled_data[label_insitu].values
            
            print("ISMN time series:", ISMN_time_series)
            print("compared to", ascat_time_series)
            print("Results:")
            
            # df_metrics takes a DataFrame as input and automatically
            # calculates the metric on all combinations of columns
            # returns a named tuple for easy printing
            print(df_metrics.pearsonr(scaled_data))
            print("Spearman's (rho,p_value)", metrics.spearmanr(x, y))
            print("Kendalls's (tau,p_value)", metrics.kendalltau(x, y))
            print(df_metrics.kendalltau(scaled_data))
            print(df_metrics.rmsd(scaled_data))
            print("Bias", metrics.bias(x, y))
            print("Nash Sutcliffe", metrics.nash_sutcliffe(x, y))

            plt.close('all')

            print('-----------------------------------------')
            
            
        i += 1
        
        #only show the first 2 stations, otherwise this program would run a long time
        #and produce a lot of plots
        if i >= 2:
            break    


.. image:: /_static/images/compare_ASCAT_ISMN/compare_ASCAT_ISMN_0_0_1.png



.. image:: /_static/images/compare_ASCAT_ISMN/compare_ASCAT_ISMN_0_0_2.png



.. image:: /_static/images/compare_ASCAT_ISMN/compare_ASCAT_ISMN_0_0_3.png


.. parsed-literal::

    ISMN time series: MAQU CST_01 0.05 m - 0.05 m soil moisture measured with ECH20-EC-TM
    compared to GPI: 1814367 Lon: 102.142 Lat: 33.877
    Results:
    (Pearsons_r(sm_and_insitu_sm=0.41146915349727176), p_value(sm_and_insitu_sm=2.1838669056567634e-11))
    Spearman's (rho,p_value) SpearmanrResult(correlation=0.45643054586958337, pvalue=5.856143898211427e-14)
    Kendalls's (tau,p_value) KendalltauResult(correlation=0.3260009747987346, pvalue=2.9245202674608733e-13)
    (Kendall_tau(sm_and_insitu_sm=0.3260009747987346), p_value(sm_and_insitu_sm=2.9245202674608733e-13))
    rmsd(sm_and_insitu_sm=0.07977939728258261)
    Bias 0.001804053923478377
    Nash Sutcliffe -0.1988660324051037


.. image:: /_static/images/compare_ASCAT_ISMN/compare_ASCAT_ISMN_1_0_1.png



.. image:: /_static/images/compare_ASCAT_ISMN/compare_ASCAT_ISMN_1_0_2.png



.. image:: /_static/images/compare_ASCAT_ISMN/compare_ASCAT_ISMN_1_0_3.png


.. parsed-literal::

    ISMN time series: MAQU CST_02 0.05 m - 0.05 m soil moisture measured with ECH20-EC-TM
    compared to GPI: 1803695 Lon: 102.145 Lat: 33.652
    Results:
    (Pearsons_r(sm_and_insitu_sm=0.73829377974113), p_value(sm_and_insitu_sm=9.582827090486536e-48))
    Spearman's (rho,p_value) SpearmanrResult(correlation=0.7088106610178744, pvalue=1.6438422626309885e-42)
    Kendalls's (tau,p_value) KendalltauResult(correlation=0.531613355918225, pvalue=3.009619482130224e-36)
    (Kendall_tau(sm_and_insitu_sm=0.531613355918225), p_value(sm_and_insitu_sm=3.009619482130224e-36))
    rmsd(sm_and_insitu_sm=0.05307874498167096)
    Bias -0.00046688712522047204
    Nash Sutcliffe 0.46408936677107304

    
