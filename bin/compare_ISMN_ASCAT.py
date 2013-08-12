#Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, 
#DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Example script that reads ISMN and ASCAT data from a point,
performs temporal matching and scaling and calculates
different metrics.

Created on Aug 9, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''


import pytesmo.io.ismn.interface as ismn
import pytesmo.io.sat.ascat as ascat
import pytesmo.temporal_matching as temp_match
import pytesmo.scaling as scaling
import pytesmo.metrics as metrics

import os
import matplotlib.pyplot as plt


#path to ASCAT SSM zip files from FTP server
#on windows the first string has to be your drive letter
#like 'C:\\'
path_to_ascat_ssm_data = os.path.join('path','to','SSM_data',
                                      'from','FTP Server') 

#path to grid definition file, default name TUW_W54_01_lonlat-ld-land.txt
path_to_grid_definition = os.path.join('path','to','grid_definition',
                                       'from','FTP Server')  

#path to advisory flags from FTP Server
path_to_adv_flags = os.path.join('path','to','advisory_flags',
                                 'from','FTP Server')  


ascat_SSM_reader = ascat.Ascat_SSM(path_to_ascat_ssm_data,path_to_grid_definition,
                                   advisory_flags_path = path_to_adv_flags)


path_to_ismn_data =os.path.join('/media','sf_D','small_projects','cpa_2013_07_ISMN_userformat_reader','ceop_sep_parser_test')

#initialize interface, this can take up to a few minutes the first
#time, since all metadata has to be collected
ISMN_reader = ismn.ISMN_Interface(path_to_ismn_data)

i = 0

label_ascat='SSM'
label_insitu='insitu_sm'

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
        
        #the plot shows that ISMN and ASCAT are observed in different units
        matched_data.plot(secondary_y=[label_ascat])
        plt.show()
        
        #this takes the matched_data DataFrame and adds a column 
        scaled_data = scaling.add_scaled(matched_data, method='lin_cdf_match',
                                         label_in=label_ascat,label_scale=label_insitu)
        #the label of the scaled data is construced as label_in+'_scaled_'+method
        scaled_ascat_label = label_ascat+'_scaled_'+'lin_cdf_match'
        #now the scaled ascat data and insitu_sm are in the same space    
        scaled_data.plot(secondary_y=[label_ascat])
        plt.show()
        
        plt.scatter(matched_data[scaled_ascat_label].values,matched_data[label_insitu].values)
        plt.xlabel(scaled_ascat_label)
        plt.ylabel(label_insitu)
        plt.show()
        
        #calculate correlation coefficients, RMSD, bias, Nash Sutcliffe
        x, y = matched_data[scaled_ascat_label].values, matched_data[label_insitu].values
        
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
    















