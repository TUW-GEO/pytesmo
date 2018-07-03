# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import ascat.timeseries as ascat
import pytesmo.time_series.anomaly as ts_anomaly

import os
import matplotlib.pyplot as plt

testdata_folder = '/pytesmo/testdata'

# <codecell>

ascat_data_folder = os.path.join(testdata_folder,
                                 'sat/ascat/netcdf/55R22')
ascat_grid_folder = os.path.join(testdata_folder,
                                 'sat/ascat/netcdf/grid')
static_layers_folder = os.path.join(testdata_folder,
                                    'sat/h_saf/static_layer')

#init the ASCAT_SSM reader with the paths
ascat_SSM_reader = ascat.AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                                     grid_filename='TUW_WARP5_grid_info_2_1.nc',
                                     static_layer_path=static_layers_folder)

# <codecell>

# Read data for location in northern Italy
ascat_ts = ascat_SSM_reader.read(11, 45)
#plot soil moisture
ascat_ts.data['sm'].plot(title='SSM data')
plt.show()

# <codecell>

#calculate anomaly based on moving +- 17 day window
anom = ts_anomaly.calc_anomaly(ascat_ts.data['sm'], window_size=35)
anom.plot(title='Anomaly (35-day window)')
plt.show()

# <codecell>

#calculate climatology
climatology = ts_anomaly.calc_climatology(ascat_ts.data['sm'])
climatology.plot(title='Climatology')
plt.show()

# <codecell>

#calculate anomaly based on climatology
anomaly_clim = ts_anomaly.calc_anomaly(ascat_ts.data['sm'], climatology=climatology)
anomaly_clim.plot(title='Anomaly (climatology)')
plt.show()

# <codecell>
