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
Created on Sep 30, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''


import os
import unittest
from pytesmo.io.sat import ascat

from datetime import datetime
import numpy as np


class TestAscat(unittest.TestCase):

    def setUp(self):
        self.ascat_folder = os.path.join(os.path.dirname(__file__), 'test_data', 'ascat', 'SSM')
        self.ascat_adv_folder = os.path.join(os.path.dirname(__file__), 'test_data', 'ascat', 'advisory_flags')
        # grid info file is too big to include on github
        self.ascat_grid_folder = os.path.join('/media', 'sf_D', 'pytesmo', 'test_data', 'ascat', 'grid')
        # init the ASCAT_SSM reader with the paths
        self.ascat_SSM_reader = ascat.Ascat_SSM(self.ascat_folder, self.ascat_grid_folder,
                                   advisory_flags_path=self.ascat_adv_folder)

    def test_read_ssm(self):

        gpi = 2329253
        result = self.ascat_SSM_reader.read_ssm(gpi)
        assert result.gpi == gpi
        assert result.longitude == 14.28413
        assert result.latitude == 45.698074
        assert list(result.data.columns) == ['ERR', 'SSF', 'SSM', 'frozen_prob', 'snow_prob']
        assert len(result.data) == 2058
        assert result.data.ix[15].name == datetime(2007, 1, 15, 19)
        assert result.data.ix[15]['ERR'] == 7
        assert result.data.ix[15]['SSF'] == 1
        assert result.data.ix[15]['SSM'] == 53
        assert result.data.ix[15]['frozen_prob'] == 29
        assert result.data.ix[15]['snow_prob'] == 0

    def test_neighbor_search(self):

        self.ascat_SSM_reader._load_grid_info()
        gpi, distance = self.ascat_SSM_reader.grid.find_nearest_gpi(3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


class TestAscatNetCDF(unittest.TestCase):

    def setUp(self):
        self.ascat_folder = os.path.join('/media', 'sf_R', 'Datapool_processed', 'WARP', 'WARP5.5',
                                         'ASCAT_WARP5.5_R1.2', '080_ssm', 'netcdf')
        # grid info file is too big to include on github
        self.ascat_grid_folder = os.path.join('/media', 'sf_R', 'Datapool_processed', 'WARP', 'ancillary', 'warp5_grid')
        # init the ASCAT_SSM reader with the paths
        self.ascat_SSM_reader = ascat.AscatH25_SSM(self.ascat_folder, self.ascat_grid_folder)

    def test_read_ssm_masked_no_snow(self):
        """
        tests reading of data for a gpi where the snow mask is not valid
        """
        gpi = 2199945
        result = self.ascat_SSM_reader.read_ssm(gpi, absolute_values=True,
                                                mask_frozen_prob=10,
                                                mask_snow_prob=5)
        assert result.gpi == gpi
        np.testing.assert_approx_equal(result.longitude, 9.1312, significant=4)
        np.testing.assert_approx_equal(result.latitude, 42.5481, significant=4)
        assert list(result.data.columns) == ['orbit_dir', 'proc_flag',
                                             'sm', 'sm_noise', 'ssf',
                                             'sm_por_gldas', 'sm_noise_por_gldas',
                                             'sm_por_hwsd', 'sm_noise_por_hwsd',
                                             'frozen_prob', 'snow_prob']
        assert len(result.data) == 2425
        assert result.data.ix[15].name == datetime(2007, 1, 16, 20, 53, 30)
        assert result.data.ix[15]['sm'] == 10
        assert result.data.ix[15]['ssf'] == 1
        assert result.data.ix[15]['sm_noise'] == 8
        assert result.data.ix[15]['frozen_prob'] == 0
        assert result.data.ix[15]['snow_prob'] == 0
        assert result.data.ix[15]['orbit_dir'] == 'A'
        assert result.data.ix[15]['proc_flag'] == 0
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_gldas'], 0.061, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_gldas'], 0.0488, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_hwsd'], 0.0437475, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_hwsd'], 0.034998, significant=6)
        assert result.topo_complex == 22
        assert result.wetland_frac == 0
        np.testing.assert_approx_equal(result.porosity_gldas, 0.61, significant=5)
        np.testing.assert_approx_equal(result.porosity_hwsd, 0.437475, significant=5)

    def test_read_ssm(self):

        gpi = 2329253
        result = self.ascat_SSM_reader.read_ssm(gpi, absolute_values=True)
        assert result.gpi == gpi
        np.testing.assert_approx_equal(result.longitude, 14.28413, significant=4)
        np.testing.assert_approx_equal(result.latitude, 45.698074, significant=4)
        assert list(result.data.columns) == ['orbit_dir', 'proc_flag',
                                             'sm', 'sm_noise', 'ssf',
                                             'sm_por_gldas', 'sm_noise_por_gldas',
                                             'sm_por_hwsd', 'sm_noise_por_hwsd',
                                             'frozen_prob', 'snow_prob']
        assert len(result.data) == 2292
        assert result.data.ix[15].name == datetime(2007, 1, 15, 19, 34, 42)
        assert result.data.ix[15]['sm'] == 52
        assert result.data.ix[15]['ssf'] == 1
        assert result.data.ix[15]['sm_noise'] == 7
        assert result.data.ix[15]['frozen_prob'] == 29
        assert result.data.ix[15]['snow_prob'] == 0
        assert result.data.ix[15]['orbit_dir'] == 'A'
        assert result.data.ix[15]['proc_flag'] == 0
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_gldas'], 0.2819555, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_gldas'], 0.03795555, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_hwsd'], 0.2237216, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_hwsd'], 0.03011637, significant=6)
        assert result.topo_complex == 14
        assert result.wetland_frac == 0
        np.testing.assert_approx_equal(result.porosity_gldas, 0.54222, significant=5)
        np.testing.assert_approx_equal(result.porosity_hwsd, 0.430234, significant=5)

    def test_neighbor_search(self):

        self.ascat_SSM_reader._load_grid_info()
        gpi, distance = self.ascat_SSM_reader.grid.find_nearest_gpi(3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


class TestAscatNetCDF_V5521(unittest.TestCase):

    def setUp(self):
        self.ascat_folder = os.path.join('/media', 'sf_R', 'Datapool_processed', 'WARP', 'WARP5.5',
                                         'ASCAT_WARP5.5_R2.1', '080_ssm', 'netcdf')
        # grid info file is too big to include on github
        self.ascat_grid_folder = os.path.join('/media', 'sf_R', 'Datapool_processed', 'WARP', 'ancillary', 'warp5_grid')
        # init the ASCAT_SSM reader with the paths
        self.ascat_SSM_reader = ascat.AscatH25_SSM(self.ascat_folder, self.ascat_grid_folder)

    def test_read_ssm(self):

        gpi = 2329253
        result = self.ascat_SSM_reader.read_ssm(gpi, absolute_values=True)
        assert result.gpi == gpi
        np.testing.assert_approx_equal(result.longitude, 14.28413, significant=4)
        np.testing.assert_approx_equal(result.latitude, 45.698074, significant=4)
        assert list(result.data.columns) == ['orbit_dir', 'proc_flag',
                                             'sm', 'sm_noise', 'ssf',
                                             'sm_por_gldas', 'sm_noise_por_gldas',
                                             'sm_por_hwsd', 'sm_noise_por_hwsd',
                                             'frozen_prob', 'snow_prob']
        assert len(result.data) == 2457
        assert result.data.ix[15].name == datetime(2007, 1, 15, 19, 34, 41)
        assert result.data.ix[15]['sm'] == 55
        assert result.data.ix[15]['ssf'] == 1
        assert result.data.ix[15]['sm_noise'] == 7
        assert result.data.ix[15]['frozen_prob'] == 29
        assert result.data.ix[15]['snow_prob'] == 0
        assert result.data.ix[15]['orbit_dir'] == 'A'
        assert result.data.ix[15]['proc_flag'] == 0
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_gldas'], 0.2969999, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_gldas'], 0.03779999, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_por_hwsd'], 0.2364999, significant=6)
        np.testing.assert_approx_equal(result.data.ix[15]['sm_noise_por_hwsd'], 0.0300999, significant=6)
        assert result.topo_complex == 14
        assert result.wetland_frac == 0
        np.testing.assert_approx_equal(result.porosity_gldas, 0.539999, significant=5)
        np.testing.assert_approx_equal(result.porosity_hwsd, 0.4299994, significant=5)


if __name__ == '__main__':
    unittest.main()
