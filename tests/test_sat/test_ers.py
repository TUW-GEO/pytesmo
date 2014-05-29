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
Created on Nov 7, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''
import os
import unittest
from pytesmo.io.sat import ers

from datetime import datetime
import numpy as np


class TestERSNetCDF(unittest.TestCase):

    def setUp(self):
        self.ers_folder = os.path.join('/media', 'sf_R', 'Datapool_processed', 'WARP', 'WARP5.5',
                                         'ERS_AMI_WS_WARP5.5_R1.1', '070_ssm', 'netcdf')
        self.ers_grid_folder = os.path.join('/media', 'sf_R', 'Datapool_processed', 'WARP', 'ancillary', 'warp5_grid')
        # init the ERS_SSM reader with the paths
        self.ers_SSM_reader = ers.ERS_SSM(self.ers_folder, self.ers_grid_folder)

    def test_read_ssm(self):

        gpi = 2329253
        result = self.ers_SSM_reader.read_ssm(gpi, absolute_values=True)
        assert result.gpi == gpi
        np.testing.assert_approx_equal(result.longitude, 14.28413, significant=4)
        np.testing.assert_approx_equal(result.latitude, 45.698074, significant=4)
        assert list(result.data.columns) == ['orbit_dir', 'proc_flag',
                                             'sm', 'sm_noise',
                                             'sm_por_gldas', 'sm_noise_por_gldas',
                                             'sm_por_hwsd', 'sm_noise_por_hwsd',
                                             'frozen_prob', 'snow_prob']
        assert len(result.data) == 478
        assert result.data.ix[15].name == datetime(1992, 1, 27, 21, 11, 42)
        assert result.data.ix[15]['sm'] == 57
        assert result.data.ix[15]['sm_noise'] == 7
        assert result.data.ix[15]['frozen_prob'] == 18
        assert result.data.ix[15]['snow_prob'] == 0
        assert result.data.ix[15]['orbit_dir'] == 'A'
        assert result.data.ix[15]['proc_flag'] == 0
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_gldas'], 0.3090667, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_gldas'], 0.03795555, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_hwsd'], 0.2452333, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_hwsd'], 0.03011637, significant=6)
        assert result.topo_complex == 14
        assert result.wetland_frac == 0
        np.testing.assert_approx_equal(result.porosity_gldas, 0.54222, significant=5)
        np.testing.assert_approx_equal(result.porosity_hwsd, 0.430234, significant=5)

    def test_neighbor_search(self):

        self.ers_SSM_reader._load_grid_info()
        gpi, distance = self.ers_SSM_reader.grid.find_nearest_gpi(3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


if __name__ == '__main__':
    unittest.main()