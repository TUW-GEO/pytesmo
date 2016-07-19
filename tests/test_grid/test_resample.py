# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of
#      Geodesy and Geoinformation nor the
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
import ascat
import os
import sys
import functools
import pytest
import numpy.testing as nptest
import datetime
import unittest


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_resample_H07(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), '..',
                                 'test-data', 'sat', 'h_saf', 'h07')
        self.reader = ascat.H07img(data_path)

    def tearDown(self):
        self.reader = None

    def test_resample_to_zero_dot_one_deg(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2010, 5, 1, 8, 33, 1))
        # lets resample to a 0.1 degree grid
        # define the grid points in latitude and longitude
        lats_dim = np.arange(25, 75, 0.1)
        lons_dim = np.arange(-25, 45, 0.1)
        # make 2d grid out the 1D grid spacing
        lons_grid, lats_grid = np.meshgrid(lons_dim, lats_dim)

        resampled_data = resample.resample_to_grid(data, lons, lats,
                                                   lons_grid, lats_grid)

        for key in data:
            assert resampled_data[key].shape == lons_grid.shape


def test_resample_dtypes():
    """
    Test if dtypes stay the same when resampling.
    """

    data = {'testint8': np.array([5, 5], dtype=np.int8),
            'testfloat16': np.array([5, 5], dtype=np.float16)}

    fill_values = {'testint8': 0,
                   'testfloat16': 999.}
    lons = np.array([0, 0.1])
    lats = np.array([0, 0.1])
    # lets resample to a 0.1 degree grid
    # define the grid points in latitude and longitude
    lats_dim = np.arange(-1, 1, 0.1)
    lons_dim = np.arange(-1, 1, 0.1)
    # make 2d grid out the 1D grid spacing
    lons_grid, lats_grid = np.meshgrid(lons_dim, lats_dim)

    resampled_data = resample.resample_to_grid(data, lons, lats,
                                               lons_grid, lats_grid,
                                               fill_values=fill_values)

    for key in data:
        assert resampled_data[key].shape == lons_grid.shape
        assert resampled_data[key].dtype == data[key].dtype


def test_resample_hamming():
    """
    Test if hamming window is applied correctly
    """
    # let's do 5 points with the highest value in the middle
    # -1-
    # 151
    # -1-

    data = {'testfloat16': np.array([1, 1, 5, 1, 1], dtype=np.float16)}

    fill_values = {'testfloat16': 999.}
    lons = np.array([0, -0.1, 0, 0.1, 0])
    lats = np.array([0.1, 0, 0, 0, -0.1])
    # lets resample to a 0.1 degree grid
    # define the grid points in latitude and longitude
    lats_dim = np.arange(-0.1, 0.11, 0.1)
    lons_dim = np.arange(-0.1, 0.11, 0.1)
    # make 2d grid out the 1D grid spacing
    lons_grid, lats_grid = np.meshgrid(lons_dim, lats_dim)
    # make partial function of the hamming window the radius of the hamming
    # window is in meters not in degrees
    hamm = functools.partial(resample.hamming_window, 15000)

    resampled_data = resample.resample_to_grid(data, lons, lats,
                                               lons_grid, lats_grid,
                                               fill_values=fill_values,
                                               methods='custom', weight_funcs=hamm)

    resampled_should = np.array([[1.640625,  1.64160156,  1.640625],
                                 [1.64160156,  3.11132812,  1.64160156],
                                 [1.640625,  1.64160156,  1.640625]])
    for key in data:
        assert resampled_data[key].shape == lons_grid.shape
        assert resampled_data[key].dtype == data[key].dtype
        nptest.assert_almost_equal(resampled_data[key], resampled_should)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
