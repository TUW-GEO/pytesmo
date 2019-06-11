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
Example script reads and plots ASCAT H25 SSM data with different masking options
and also converts the data to absolute values using the included
porosity data.


Created on Nov 8, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''

import matplotlib.pyplot as plt
import ascat.tuw as tuw
import os

# See also: https://github.com/TUW-GEO/ascat/blob/master/docs/read_tuw_ascat.rst

testdata_folder = '/pytesmo/testdata'

ascat_data_folder = os.path.join(testdata_folder,
                                 'sat/ascat/SSM')
ascat_grid_folder = os.path.join(testdata_folder,
                                 'sat/ascat/grid')
advisory_flags_folder = os.path.join(testdata_folder,
                                 'sat/ascat/advisory_flags')

# # init the ASCAT_SSM reader with the paths

ascat_SSM_reader = tuw.Ascat_SSM(ascat_data_folder, ascat_grid_folder,
                                 advisory_flags_path=advisory_flags_folder)

gpi = 2329253

ascat_ts = ascat_SSM_reader.read_ssm(gpi)

ascat_ts.plot()
plt.show()

# the ASCATTimeSeries object also contains metadata

print("Topographic complexity", ascat_ts.topo_complex)
print("Wetland fraction", ascat_ts.wetland_frac)
print("Porosity from GLDAS model", ascat_ts.porosity_gldas)
print("Porosity from Harmonized World Soil Database", ascat_ts.porosity_hwsd)

# It is also possible to automatically convert the soil moisture to absolute values using

ascat_ts_absolute = ascat_SSM_reader.read_ssm(gpi, absolute_values=True)
# this introduces 4 new columns in the returned data
# scaled sm and sm_noise with porosity_gldas
# scaled sm and sm_noise with porosity_hwsd
#print(ascat_ts_absolute.data)

# select relevant columns and plot
# ascat_ts_absolute.data = ascat_ts_absolute.data[['sm_por_gldas', 'sm_por_hwsd']]
ascat_ts_absolute.plot()
plt.show()

# We can also automatically mask the data during reading
# In this example all measurements where the Surface State Flag
# shows frozen or where the frozen or snow probabilities are more
# than 10 percent are removed from the time series
ascat_ts = ascat_SSM_reader.read_ssm(gpi, mask_ssf=True,
                                     mask_frozen_prob=10,
                                     mask_snow_pro=10)

ascat_ts.plot()
plt.show()
