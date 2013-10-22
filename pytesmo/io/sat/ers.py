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
Created on Oct 22, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

from pytesmo.io.sat.ascat import AscatNetcdf, ASCATTimeSeries



class ERSTimeSeries(ASCATTimeSeries):
    
    def __init__(self, gpi, lon, lat, cell, data):
        super(ERSTimeSeries, self).__init__(gpi, lon, lat, cell, data)
        
    def __repr__(self):
        
        return "ERS time series gpi:%d lat:%2.3f lon:%3.3f" % (self.gpi, self.latitude, self.longitude) 



class ERS_SSM(AscatNetcdf):        
    """
    class for reading ASCAT SSM data. It extends AscatNetcdf and provides the 
    information necessary for reading SSM data
    
    Parameters
    ----------
    path : string
        path to data folder which contains the netCDF files from the FTP server
    grid_path : string
        path to grid_info folder which contains txt files with information about
        grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        name of the grid info txt file in grid_path    
    advisory_flags_path : string, optional
        path to advisory flags .dat files, if not provided they will not be used    
    topo_threshold : int, optional
        if topographic complexity of read grid point is above this 
        threshold a warning is output during reading
    wetland_threshold : int, optional
        if wetland fraction of read grid point is above this 
        threshold a warning is output during reading
    netcdftemplate : string, optional
        string template for the netCDF filename. This specifies where the cell number is
        in the netCDF filename. Standard value is 'TUW_METOP_ASCAT_WARP55R12_%04d.nc' in 
        which %04d will be substituded for the cell number during reading of the data
    
    
    Attributes
    ----------
    include_in_df : list
        list of variables in the netcdf file 
        that should be returned to the user after reading 
        
    Methods
    -------
    read_ssm(*args,**kwargs)
        read surface soil moisture
    """
    def __init__(self, path, grid_path, grid_info_filename='warp5_grid.nc',
                 topo_threshold=50, wetland_threshold=50, netcdftemplate='%04d.nc'):
        
        super(ERS_SSM, self).__init__(path, grid_path, grid_info_filename=grid_info_filename,
                 topo_threshold=topo_threshold, wetland_threshold=wetland_threshold,
                 netcdftemplate=netcdftemplate)
        self.include_in_df = ['sm', 'sm_noise']
        self.to_absolute = ['sm', 'sm_noise']
        
        
    def read_ssm(self, *args, **kwargs):
        """
        function to read SSM takes either 1 or 2 arguments.
        It can be called as read_ssm(gpi,**kwargs) or read_ssm(lon,lat,**kwargs)
        
        Parameters
        ----------
        gpi : int
            grid point index
        lon : float
            longitude of point
        lat : float
            latitude of point
        mask_frozen_prob : int,optional
            if included in kwargs then all observations taken when 
            frozen probability > mask_frozen_prob are removed from the result 
        mask_snow_prob : int,optional
            if included in kwargs then all observations taken when 
            snow probability > mask_snow_prob are removed from the result
        absolute_values : boolean, optional
            if True soil porosities from HWSD and GLDAS will be used to 
            derive absolute values which will be available in the 
            pandas.DataFrame in the columns 
            'sm_por_gldas','sm_noise_por_gldas',
            'sm_por_hwsd','sm_noise_por_hwsd'
             
        Returns
        -------
        df : pandas.DataFrame
            containing all fields in self.include_in_df plus frozen_prob and snow_prob if
            advisory_flags_path was set
        """
        df, gpi, lon, lat, cell = super(ERS_SSM, self)._read_ts(*args, **kwargs)
                
        return ERSTimeSeries(gpi, lon, lat, cell, df)    
