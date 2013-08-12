In[1]::

    import pytesmo.io.ismn.interface as ismn
    import pytesmo.io.sat.ascat as ascat
    import pytesmo.temporal_matching as temp_match
    import pytesmo.scaling as scaling
    import pytesmo.metrics as metrics
    
    import os
    import matplotlib.pyplot as plt
	
In[2]::

    #set paths for ASCAT SSM
    path_to_ascat_ssm_data = os.path.join('D:\\','small_projects',
                                          'cpa_2013_07_userformat_reader','data','ASCAT_SSM_25km_ts_WARP5.5_R0.1','data') 
    
    path_to_grid_definition = os.path.join('D:\\','small_projects',
                                          'cpa_2013_07_userformat_reader','data','auxiliary_data','grid_info')
    
    path_to_adv_flags = os.path.join('D:\\','small_projects',
                                          'cpa_2013_07_userformat_reader','data','auxiliary_data','advisory_flags')  
										  
In[3]::

    #set path to ISMN data
    path_to_ismn_data =os.path.join('D:\\','small_projects','cpa_2013_07_ISMN_userformat_reader','header_values_parser_test')
	
In[4]::

    #Initialize readers
    ascat_SSM_reader = ascat.Ascat_SSM(path_to_ascat_ssm_data,path_to_grid_definition,
                                       advisory_flags_path = path_to_adv_flags)
    
    ISMN_reader = ismn.ISMN_Interface(path_to_ismn_data)
In[5]::

    i = 0
    
    label_ascat='SSM'
    label_insitu='insitu_sm'
    
In[6]::

    #this loops through all stations that measure soil moisture
    for station in ISMN_reader.stations_that_measure('soil moisture'):
        
        #this loops through all time series of this station that measure soil moisture
        #between 0 and 0.1 meters
        for ISMN_time_series in station.data_for_variable('soil moisture',min_depth=0,max_depth=0.1):
            
            ascat_time_series = ascat_SSM_reader.read_ssm(ISMN_time_series.longitude,
                                                          ISMN_time_series.latitude,
                                                          mask_ssf=True,
                                                          mask_frozen_prob = 5,
                                                          mask_snow_prob = 5)
            
    
            #drop nan values before doing any matching
            ascat_time_series.data = ascat_time_series.data.dropna()
            ISMN_time_series.data = ISMN_time_series.data.dropna()
            
            #rename the soil moisture column in ISMN_time_series.data to insitu_sm
            #to clearly differentiate the time series when they are plotted together
            ISMN_time_series.data.rename(columns={'soil moisture':label_insitu},inplace=True)
            
            #get ISMN data that was observerd within +- 1 hour(1/24. day) of the ASCAT observation
            #do not include those indexes where no observation was found
            matched_ISMN_data = temp_match.df_match(ascat_time_series.data,ISMN_time_series.data,
                                                    window=1/24.,dropna=True)
            #matched ISMN data is now a dataframe with the same datetime index
            #as ascat_time_series.data and the nearest insitu observation
            
            #temporal matching also includes distance information
            #but we are not interested in it right now so let's drop it
            matched_ISMN_data = matched_ISMN_data.drop(['distance'],axis=1)
            
            #this joins the SSM column of the ASCAT data to the matched ISMN data
            matched_data = matched_ISMN_data.join(ascat_time_series.data[label_ascat])       
            
            #continue only with relevant columns
            matched_data = matched_data[[label_ascat,label_insitu]]
            
            #the plot shows that ISMN and ASCAT are observed in different units
            matched_data.plot(figsize=(15,5),secondary_y=[label_ascat])
            plt.show()
            
            #this takes the matched_data DataFrame and adds a column 
            scaled_data = scaling.add_scaled(matched_data, method='lin_cdf_match',
                                             label_in=label_ascat,label_scale=label_insitu)
            #the label of the scaled data is construced as label_in+'_scaled_'+method
            scaled_ascat_label = label_ascat+'_scaled_'+'lin_cdf_match'
            #now the scaled ascat data and insitu_sm are in the same space    
            scaled_data.plot(figsize=(15,5),secondary_y=[label_ascat])
            plt.show()
            
            plt.scatter(scaled_data[scaled_ascat_label].values,scaled_data[label_insitu].values)
            plt.xlabel(scaled_ascat_label)
            plt.ylabel(label_insitu)
            plt.show()
            
            #calculate correlation coefficients, RMSD, bias, Nash Sutcliffe
            x, y = scaled_data[scaled_ascat_label].values, scaled_data[label_insitu].values
            
            print "ISMN time series:",ISMN_time_series
            print "compared to"
            print ascat_time_series
            print "Results:"
            print "Pearson's (R,p_value)", metrics.pearsonr(x, y)
            print "Spearman's (rho,p_value)", metrics.spearmanr(x, y)
            print "Kendalls's (tau,p_value)", metrics.kendalltau(x, y)
            print "RMSD", metrics.rmsd(x, y)
            print "Bias", metrics.bias(x, y)
            print "Nash Sutcliffe", metrics.nash_sutcliffe(x, y)
            
            
        i += 1
        
        #only show the first 2 stations, otherwise this program would run a long time
        #and produce a lot of plots
        if i >= 2:
            break    




.. image:: compare_ASCAT_ISMN_files/compare_ASCAT_ISMN_5_0.png



.. image:: compare_ASCAT_ISMN_files/compare_ASCAT_ISMN_5_1.png



.. image:: compare_ASCAT_ISMN_files/compare_ASCAT_ISMN_5_2.png


.. parsed-literal::

    ISMN time series: OZNET Alabama 0.00 m - 0.05 m soil moisture measured with Stevens-Hydra-Probe 
    compared to
    ASCAT time series gpi:1884359 lat:-35.342 lon:147.541
    Results:
    Pearson's (R,p_value) (0.59736953256517777, 1.4810058830429653e-60)
    Spearman's (rho,p_value) (0.63684906343988457, 4.8971200217989799e-71)
    Kendalls's (tau,p_value) (0.45994629380576146, 4.6771942474849024e-65)
    RMSD 0.0807313501609
    Bias 0.00258302466701
    Nash Sutcliffe 0.221824420266
    


.. image:: compare_ASCAT_ISMN_files/compare_ASCAT_ISMN_5_4.png



.. image:: compare_ASCAT_ISMN_files/compare_ASCAT_ISMN_5_5.png



.. image:: compare_ASCAT_ISMN_files/compare_ASCAT_ISMN_5_6.png


.. parsed-literal::

    ISMN time series: OZNET Balranald-Bolton_Park 0.00 m - 0.08 m soil moisture measured with CS615 
    compared to
    ASCAT time series gpi:1821003 lat:-33.990 lon:146.381
    Results:
    Pearson's (R,p_value) (0.65811087356086551, 9.1620935528699124e-126)
    Spearman's (rho,p_value) (0.65874491635978671, 4.3707663858540222e-126)
    Kendalls's (tau,p_value) (0.48451720923430946, 4.6613967263363183e-117)
    RMSD 0.0283269899964
    Bias -0.000181669876467
    Nash Sutcliffe 0.314284186192

    
