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
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Example script that reads ISMN metadata and data, then plots 
the available stations on a map and reads and plots a timeseries


Created on Aug 8, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''


import pytesmo.io.ismn.interface as ismn
import os
import matplotlib.pyplot as plt
import random

# path unzipped file downloaded from the ISMN web portal
# on windows the first string has to be your drive letter
# like 'C:\\'
path_to_ismn_data = os.path.join('path', 'to', 'ISMN_data',
                                 'from', 'ISMN website')

path_to_ismn_data = os.path.join('/media', 'sf_D', 'small_projects',
                                 'cpa_2013_07_ISMN_userformat_reader',
                                 'ceop_sep_parser_test')

# initialize interface, this can take up to a few minutes the first
# time, since all metadata has to be collected
ISMN_reader = ismn.ISMN_Interface(path_to_ismn_data)

# plot available station on a map
fig, axes = ISMN_reader.plot_station_locations()
plt.show()

# select random network and station to plot
networks = ISMN_reader.list_networks()
print "Available Networks:"
print networks

network = random.choice(networks)
stations = ISMN_reader.list_stations(network=network)
print "Available Stations in Network %s" % network
print stations

station = random.choice(stations)
station_obj = ISMN_reader.get_station(station)
print "Available Variables at Station %s" % station
# get the variables that this station measures
variables = station_obj.get_variables()
print variables

# to make sure the selected variable is not measured
# by different sensors at the same depths
# we also select the first depth and the first sensor
# even if there is only one
depths_from, depths_to = station_obj.get_depths(variables[0])

sensors = station_obj.get_sensors(variables[0], depths_from[0], depths_to[0])

# read the data of the variable, depth, sensor combination
time_series = station_obj.read_variable(variables[0], depth_from=depths_from[
                                        0], depth_to=depths_to[0], sensor=sensors[0])

# print information about the selected time series
print "Selected time series is:"
print time_series

# plot the data
time_series.plot()
plt.legend()
plt.show()

print "the first 40 valid lines of data: "
# see pandas documentation for more information
print time_series.data.dropna().head(40)
