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
Created on Jan 21, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''
import unittest
import numpy as np
import numpy.testing as nptest
import os
from netCDF4 import Dataset

import pytesmo.grid.netcdf as grid_nc
import pytesmo.grid.grids as grids


def curpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class Test(unittest.TestCase):

    def setUp(self):
        lat, lon = np.arange(180) - 90, np.arange(360) - 180
        self.lats, self.lons = np.meshgrid(lat, lon)
        self.lats, self.lons = self.lats.flatten(), self.lons.flatten()
        self.cells = grids.lonlat2cell(self.lons, self.lats)
        self.subset = np.sort(np.random.choice(np.arange(self.lats.size),
                                               size=500, replace=False))
        self.basic = grids.BasicGrid(self.lons, self.lats, subset=self.subset,
                                     shape=(180, 360))
        self.basic_irregular = grids.BasicGrid(self.lons, self.lats,
                                               subset=self.subset)
        self.cellgrid = grids.CellGrid(self.lons, self.lats, self.cells,
                                       subset=self.subset)

        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')
        if not os.path.exists(os.path.join(curpath(), 'data')):
            os.mkdir(os.path.join(curpath(), 'data'))

    def tearDown(self):
        os.remove(self.testfilename)

    def test_save_lonlat_nc(self):
        grid_nc.save_lonlat(self.testfilename,
                            self.lons, self.lats, self.cells,
                            subset_points=self.subset,
                            global_attrs={'test': 'test_attribute'})

        with Dataset(self.testfilename) as nc_data:
            nptest.assert_array_equal(self.lats, nc_data.variables['lat'][:])
            nptest.assert_array_equal(self.lons, nc_data.variables['lon'][:])
            nptest.assert_array_equal(self.cells, nc_data.variables['cell'][:])
            nptest.assert_array_equal(self.subset, np.where(nc_data.variables['subset_flag'][:] == 1)[0])
            assert nc_data.test == 'test_attribute'

    def test_save_basicgrid_nc(self):
        grid_nc.save_grid(self.testfilename,
                          self.basic,
                          global_attrs={'test': 'test_attribute'})

        with Dataset(self.testfilename) as nc_data:
            nptest.assert_array_equal(np.unique(self.lats)[::-1],
                                      nc_data.variables['lat'][:])
            nptest.assert_array_equal(np.unique(self.lons),
                                      nc_data.variables['lon'][:])

            nptest.assert_array_equal(self.subset,
                                      np.where(nc_data.variables['subset_flag'][:].flatten() == 1)[0])
            assert nc_data.test == 'test_attribute'
            assert nc_data.shape[0] == 180
            assert nc_data.shape[1] == 360

    def test_save_basicgrid_irregular_nc(self):
        grid_nc.save_grid(self.testfilename,
                          self.basic_irregular,
                          global_attrs={'test': 'test_attribute'})

        with Dataset(self.testfilename) as nc_data:
            nptest.assert_array_equal(self.lats, nc_data.variables['lat'][:])
            nptest.assert_array_equal(self.lons, nc_data.variables['lon'][:])
            nptest.assert_array_equal(self.subset,
                                      np.where(nc_data.variables['subset_flag'][:] == 1)[0])
            assert nc_data.test == 'test_attribute'
            assert nc_data.shape == 64800

    def test_save_cellgrid_nc(self):
        grid_nc.save_grid(self.testfilename,
                          self.cellgrid,
                          global_attrs={'test': 'test_attribute'})

        with Dataset(self.testfilename) as nc_data:
            nptest.assert_array_equal(self.lats, nc_data.variables['lat'][:])
            nptest.assert_array_equal(self.lons, nc_data.variables['lon'][:])
            nptest.assert_array_equal(self.cells, nc_data.variables['cell'][:])
            nptest.assert_array_equal(self.subset, np.where(nc_data.variables['subset_flag'][:] == 1)[0])
            assert nc_data.test == 'test_attribute'
            assert nc_data.gpidirect == 0x1b

    def test_save_load_basicgrid(self):
        grid_nc.save_grid(self.testfilename,
                          self.basic)

        loaded_grid = grid_nc.load_grid(self.testfilename)
        assert self.basic == loaded_grid

    def test_save_load_basicgrid_irregular(self):
        grid_nc.save_grid(self.testfilename,
                          self.basic_irregular)

        loaded_grid = grid_nc.load_grid(self.testfilename)
        assert self.basic_irregular == loaded_grid

    def test_save_load_cellgrid(self):
        grid_nc.save_grid(self.testfilename,
                          self.cellgrid)

        loaded_grid = grid_nc.load_grid(self.testfilename)
        assert self.cellgrid == loaded_grid

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
