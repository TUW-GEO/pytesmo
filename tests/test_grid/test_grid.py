'''
Created on Mar 26, 2014
@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''
import unittest
import numpy.testing as nptest
import numpy as np
from pytesmo.grid.grids import lonlat2cell
import pytesmo.grid.grids as grids


class Test_lonlat2cell(unittest.TestCase):

    def setUp(self):
        lat = np.arange(-90, 90, 2.5)
        lon = np.arange(-180, 180, 2.5)
        self.lons, self.lats = np.meshgrid(lon, lat)

    def testlonlat2cell_hist(self):
        """
        setup grid with unequal cell size along lat and lon
        and test if the correct number of points lay in each cell
        """
        cells = lonlat2cell(
            self.lons, self.lats, cellsize_lon=15, cellsize_lat=30)
        hist, bin_edges = np.histogram(
            cells.flatten(), bins=len(np.unique(cells)))
        nptest.assert_allclose(hist, np.zeros_like(hist) + 72)


class Test_2Dgrid(unittest.TestCase):

    """
    setup simple 2D grid 2.5 degree global grid (144x72)
    which starts at the North Western corner of 90 -180
    and test 2D lookup
    """

    def setUp(self):
        lat = np.arange(90, -90, -2.5)
        lon = np.arange(-180, 180, 2.5)
        self.lon, self.lat = np.meshgrid(lon, lat)
        self.grid = grids.BasicGrid(
            self.lon.flatten(), self.lat.flatten(), shape=(len(lon), len(lat)))

    def test_gpi2rowcol(self):
        """
        test if gpi to row column lookup works correctly
        """
        gpi = 200
        row_should = 1
        column_should = 200 - 144
        row, column = self.grid.gpi2rowcol(gpi)
        assert row == row_should
        assert column == column_should

    def test_gpi2lonlat(self):
        """
        test if gpi to longitude latitude lookup works correctly
        """
        gpi = 200
        lat_should = 87.5
        lon_should = -180 + (200 - 144) * 2.5
        lon, lat = self.grid.gpi2lonlat(gpi)
        assert lon == lon_should
        assert lat == lat_should


def test_genreggrid():
    """
    test generation of regular grids
    """
    grid = grids.genreg_grid()
    assert grid.shape == (360, 180)
    lon, lat = grid.gpi2lonlat(3)
    assert lon == -176.5
    assert lat == 89.5


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
