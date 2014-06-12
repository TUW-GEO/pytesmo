# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy and Geoinformation
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
Created on Jun 11, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''


import pytesmo.grid.resample as resample
import numpy as np
import pytesmo.io.sat.h_saf as H_SAF
import os
import datetime
import unittest


class Test_resample_H07(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'test_sat', 'test_data', 'h_saf', 'h07')
        self.reader = H_SAF.H07img(data_path)

    def tearDown(self):
        self.reader = None

    def test_resample_to_zero_dot_one_deg(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read_img(datetime.datetime(2010, 5, 1, 8, 33, 1))
        # lets resample to a 0.1 degree grid
        # define the grid points in latitude and logitude
        lats_dim = np.arange(25, 75, 0.1)
        lons_dim = np.arange(-25, 45, 0.1)
        # make 2d grid out the 1D grid spacing
        lons_grid, lats_grid = np.meshgrid(lons_dim, lats_dim)

        resampled_data = resample.resample_to_grid(data, lons, lats,
                                                   lons_grid, lats_grid)

        for key in data:
            assert resampled_data[key].shape == lons_grid.shape


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()



