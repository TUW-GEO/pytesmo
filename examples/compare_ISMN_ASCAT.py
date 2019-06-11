# -*- coding: utf-8 -*-
# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, 
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
This example program loops through all insitu stations that measure soil moisture with a depth between 0 and 0.1m
it then finds the nearest ASCAT grid point and reads the ASCAT data. After temporal matching and scaling using linear
CDF matching it computes several metrics, like the correlation coefficients(Pearson's, Spearman's and Kendall's), Bias,
RMSD as well as the Nash–Sutcliffe model efficiency coefficient.


Created on Aug 8, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''

import ismn.interface as ismn
import ascat
import pytesmo.temporal_matching as temp_match
import pytesmo.scaling as scaling
import pytesmo.df_metrics as df_metrics
import pytesmo.metrics as metrics

import os
import matplotlib.pyplot as plt

testdata_folder = '/pytesmo/testdata'

ascat_data_folder = os.path.join(testdata_folder,
                                 'sat/ascat/netcdf/55R22')
ascat_grid_folder = os.path.join(testdata_folder,
                                 'sat/ascat/netcdf/grid')
static_layers_folder = os.path.join(testdata_folder,
                                    'sat/h_saf/static_layer')

# init the ASCAT SSM reader with the paths
ascat_SSM_reader = ascat.AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                                     grid_filename='TUW_WARP5_grid_info_2_1.nc',
                                     static_layer_path=static_layers_folder)
ascat_SSM_reader.read_bulk = True


# set path to ISMN data
ismn_data_folder = os.path.join(testdata_folder,
                                 'ismn/multinetwork/header_values')

# Initialize reader
ISMN_reader = ismn.ISMN_Interface(ismn_data_folder)


i = 0

label_ascat = 'sm'
label_insitu = 'insitu_sm'

# this loops through all stations that measure soil moisture
for station in ISMN_reader.stations_that_measure('soil moisture'):
    
    # this loops through all time series of this station that measure soil moisture
    # between 0 and 0.1 meters
    for ISMN_time_series in station.data_for_variable('soil moisture', min_depth=0, max_depth=0.1):

        ascat_time_series = ascat_SSM_reader.read(ISMN_time_series.longitude,
                                                  ISMN_time_series.latitude,
                                                  mask_ssf=True,
                                                  mask_frozen_prob=80,
                                                  mask_snow_prob=80)

        # focus only on the relevant variable
        ascat_time_series.data = ascat_time_series.data[label_ascat]

        # drop nan values before doing any matching
        ascat_time_series.data = ascat_time_series.data.dropna()

        ISMN_time_series.data = ISMN_time_series.data.dropna()
        
        # rename the soil moisture column in ISMN_time_series.data to insitu_sm
        # to clearly differentiate the time series when they are plotted together
        ISMN_time_series.data.rename(columns={'soil moisture':label_insitu}, inplace=True)
        
        # get ISMN data that was observerd within +- 1 hour(1/24. day) of the ASCAT observation
        # do not include those indexes where no observation was found
        matched_data = temp_match.matching(ascat_time_series.data, ISMN_time_series.data,
                                                window=1 / 24.)
        # matched ISMN data is now a dataframe with the same datetime index
        # as ascat_time_series.data and the nearest insitu observation      
        
        # continue only with relevant columns
        matched_data = matched_data[[label_ascat, label_insitu]]
        
        # the plot shows that ISMN and ASCAT are observed in different units
        matched_data.plot(figsize=(15, 5), secondary_y=[label_ascat],
                          title='temporally merged data')
        plt.show()
        
        # this takes the matched_data DataFrame and scales all columns to the 
        # column with the given reference_index, in this case in situ 
        scaled_data = scaling.scale(matched_data, method='lin_cdf_match',
                                         reference_index=1)
        
        # now the scaled ascat data and insitu_sm are in the same space    
        scaled_data.plot(figsize=(15, 5), title='scaled data')
        plt.show()
        
        plt.scatter(scaled_data[label_ascat].values, scaled_data[label_insitu].values)
        plt.xlabel(label_ascat)
        plt.ylabel(label_insitu)
        plt.show()
        
        # calculate correlation coefficients, RMSD, bias, Nash Sutcliffe
        x, y = scaled_data[label_ascat].values, scaled_data[label_insitu].values
        
        print("ISMN time series:", ISMN_time_series)
        print("compared to")
        print(ascat_time_series)
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
        
        
    i += 1

    # only show the first 2 stations, otherwise this program would run a long time
    # and produce a lot of plots
    if i >= 2:
        break


