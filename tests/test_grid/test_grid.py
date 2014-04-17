'''
Created on Mar 26, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''
import unittest
import numpy.testing as nptest
import numpy as np
from pytesmo.grid.grids import lonlat2cell


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
        cells = lonlat2cell(self.lons, self.lats, cellsize_lon=15, cellsize_lat=30)
        hist, bin_edges = np.histogram(cells.flatten(), bins=len(np.unique(cells)))
        nptest.assert_allclose(hist, np.zeros_like(hist) + 72)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
