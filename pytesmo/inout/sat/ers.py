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
Module for reading ERS data from TU Wien
'''

import os
from ascat.read_native.cdr import AscatNc


class ERS_SSM(AscatNc):

    """
    Class reading ERS data from TU Wien

    Parameters
    ----------
    path : string
        path to data folder which contains the netCDF files from the FTP server
    grid_path : string
        path to grid_info folder which contains txt files with information about
        grid point index,latitude, longitude and cell
    grid_filename : str
        Grid filename.
    static_layer_path : str
        Path to static layer files from H-SAF
    thresholds : dict, optional
        Thresholds for topographic complexity (default 50) and
        wetland fraction (default 50).
        {'topo_complex': 50, 'wetland_frac': 50}
    """

    def __init__(self, path, grid_path,
                 grid_filename='TUW_WARP5_grid_info_2_1.nc',
                 static_layer_path=None, **kwargs):

        fn_format = 'TUW_ERS_AMI_SSM_WARP55R11_{:04d}'
        grid_filename = os.path.join(grid_path, grid_filename)

        super(ERS_SSM, self).__init__(path, fn_format, grid_filename,
                                      static_layer_path,
                                      ioclass_kws={'loc_ids_name': 'gpi'},
                                      **kwargs)
