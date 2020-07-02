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
Example script that reads ASCAT Surface Soil Moisture(SSM) 
and ASCAT Soil Water Index(SWI) for a given latitude and
longitude. It also applies different maskings.


Created on Aug 8, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''


from ascat.tuw import Ascat_SWI
from ascat.tuw import Ascat_SSM
from ascat.cgls import SWI_TS
import os
import matplotlib.pyplot as plt


#path to ASCAT SSM zip files from FTP server
#on windows the first string has to be your drive letter
#like 'C:\\'

testdata_folder = '/pytesmo/testdata'

path_to_ascat_ssm_data = os.path.join(testdata_folder,'sat/ascat/SSM') 

path_to_ascat_swi_data = os.path.join(testdata_folder,'sat/ascat/SWI') 

path_to_ascat_swi_ts_data = os.path.join(testdata_folder,'sat/cglops/swi_ts') 


#path to grid definition file, default name TUW_W54_01_lonlat-ld-land.txt
path_to_grid_definition = os.path.join(testdata_folder,'sat/ascat/grid')  

#path to advisory flags from FTP Server
path_to_adv_flags = os.path.join(testdata_folder,'sat/ascat/advisory_flags')   

#init the ASCAT_SSM reader with the paths
ascat_SSM_reader = Ascat_SSM(path_to_ascat_ssm_data,path_to_grid_definition,
                                   advisory_flags_path = path_to_adv_flags)

#lon, lat = 14.284130, 45.698074
lon, lat = 14.284, 45.699

#reads ssm data nearest to this lon,lat coordinates
ssm_data_raw = ascat_SSM_reader.read_ssm(lon,lat)

#plot the data using pandas builtin plot functionality
ssm_data_raw.plot()
plt.show()


#read the same data but mask observations where the SSF shows frozen
#and where frozen and snow probabilty are greater than 20%
ssm_data_masked = ascat_SSM_reader.read_ssm(lon,lat,mask_ssf=True,mask_frozen_prob=20,mask_snow_prob=20)

#plot the data using pandas builtin plot functionality
#this time using a subplot for each variable in the DataFrame
ssm_data_masked.plot(subplots=False)
plt.show()

#plot raw and masked SSM data in one plot to compare them

ssm_data_raw.data['SSM'].plot(label='raw SSM data')
ssm_data_masked.data['SSM'].plot(label='masked SSM data')
plt.legend()
plt.show()


#read SWI data using the SWI_TS reader
ascat_SWI_reader=SWI_TS(path_to_ascat_swi_ts_data)
swi_data_raw = ascat_SWI_reader.read_ts(3002621)

#plot the data using pandas builtin plot functionality
swi_data_raw.plot()
plt.show()



#read SWI data using the Ascat_SWI reader
ascat_SWI_reader = Ascat_SWI(path_to_ascat_swi_data,path_to_grid_definition,advisory_flags_path = path_to_adv_flags)


#reads swi data nearest to this lon,lat coordinates
#without any additional keywords all unmasked T values and
#Quality flags will be read
swi_data_raw = ascat_SWI_reader.read_swi(lon,lat)

#plot the data using pandas builtin plot functionality
swi_data_raw.plot()
plt.show()


#read the same data but this time only SWI with a T value
#of 20 is returned
swi_data_T_20 = ascat_SWI_reader.read_swi(lon,lat,T=20)

#plot the data using pandas builtin plot functionality
#this time using a subplot for each variable in the DataFrame
swi_data_T_20.plot(subplots=True)
plt.show()

#you can also mask manually if you prefer
swi_data_T_20.data = swi_data_T_20.data[swi_data_T_20.data['frozen_prob'] < 10]

swi_data_T_20.plot(subplots=True)
plt.show()












