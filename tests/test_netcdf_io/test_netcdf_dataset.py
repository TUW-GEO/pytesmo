"""
Created on Nov 29, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
"""


import os
import unittest

import numpy as np
import numpy.testing as nptest
from datetime import datetime, timedelta

import pytesmo.io.netcdf.netcdf_dataset as ncdata
import pytesmo.grid.grids as grids


def curpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_write_append_read_1D(self):

        with ncdata.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w') as self.dataset:
            # create unlimited Dimension
            self.dataset.create_dim('dim', None)
            self.dataset.write_var('test', np.arange(15), dim=('dim'))

        with ncdata.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.arange(15))

        with ncdata.Dataset(self.testfilename, mode='a') as self.dataset:
            self.dataset.append_var('test', np.arange(15))

        with ncdata.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.concatenate([np.arange(15), np.arange(15)]))

    def test_write_read_2D(self):

        with ncdata.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w') as self.dataset:
            self.dataset.create_dim('dim1', 15)
            self.dataset.create_dim('dim2', 15)
            self.dataset.write_var(
                'test', np.arange(15 * 15).reshape((15, 15)),
                dim=('dim1', 'dim2'))

        with ncdata.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.arange(15 * 15).reshape((15, 15)))

    def test_write_append_2D(self):

        with ncdata.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w') as self.dataset:
            self.dataset.create_dim('dim1', 15)
            self.dataset.create_dim('dim2', None)
            self.dataset.write_var(
                'test', np.arange(15 * 15).reshape((15, 15)),
                dim=('dim1', 'dim2'))

        with ncdata.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.arange(15 * 15).reshape((15, 15)))

        with ncdata.Dataset(self.testfilename, mode='a') as self.dataset:
            self.dataset.append_var('test', np.arange(15).reshape((15, 1)))

        with ncdata.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.hstack(
                [np.arange(15 * 15).reshape((15, 15)),
                 np.arange(15).reshape((15, 1))]))


class DatasetContiguousTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_file_writing(self):

        with ncdata.ContiguousRaggedTs(self.testfilename,
                                       n_loc=3, n_obs=9, mode='w') as dataset:
            data = {'test': np.arange(3)}
            dates = np.array(
                [datetime(2007, 1, 1), datetime(2007, 2, 1),
                 datetime(2007, 3, 1)])
            dataset.write_ts(
                1, data, dates, loc_descr='first station', lon=0, lat=0, alt=5)
            dataset.write_ts(
                2, data, dates, loc_descr='first station', lon=0, lat=0, alt=5)
            dataset.write_ts(
                3, data, dates, loc_descr='first station', lon=0, lat=0, alt=5)

        with ncdata.ContiguousRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(data['test'], np.arange(3))
            dates = np.array(
                [datetime(2007, 1, 1), datetime(2007, 2, 1),
                 datetime(2007, 3, 1)])
            nptest.assert_array_equal(data['time'], dates)


class DatasetIndexedTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_file_writing(self):

        with ncdata.IndexedRaggedTs(self.testfilename, n_loc=3,
                                    mode='w') as dataset:
            for n_data in [2, 5, 6]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in xrange(n_data)])
                    dataset.write_ts(
                        location, data, dates, loc_descr='first station',
                        lon=0, lat=0, alt=5)

        with ncdata.IndexedRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(
                data['test'], np.concatenate([np.arange(2), np.arange(5),
                                              np.arange(6)]))

            test_dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_writing_with_attributes(self):

        with ncdata.IndexedRaggedTs(self.testfilename, n_loc=3,
                                    mode='w') as dataset:
            for n_data in [2, 5, 6]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in xrange(n_data)])
                    dataset.write_ts(location, data, dates,
                                     loc_descr='first station', lon=0, lat=0,
                                     alt=5,
                                     attributes={'testattribute':
                                                 'teststring'})

        with ncdata.IndexedRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            assert dataset.dataset.variables[
                'test'].testattribute == 'teststring'
            nptest.assert_array_equal(
                data['test'], np.concatenate([np.arange(2), np.arange(5),
                                              np.arange(6)]))

            test_dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)


class OrthoMultiTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_file_io_simple(self):

        with ncdata.OrthoMultiTs(self.testfilename, mode='w',
                                 n_loc=3) as dataset:
            for n_data in [5]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in xrange(n_data)])
                    dataset.write_ts(
                        location, data, dates, loc_descr='first station',
                        lon=0, lat=0, alt=5)

        with ncdata.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5))

            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_io_2_steps(self):

        with ncdata.OrthoMultiTs(self.testfilename, n_loc=3,
                                 mode='w') as dataset:
            for n_data in [5]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in xrange(n_data)])
                    dataset.write_ts(
                        location, data, dates, loc_descr='first station',
                        lon=0, lat=0, alt=5, fill_values={'test': -1})

        with ncdata.OrthoMultiTs(self.testfilename, n_loc=3,
                                 mode='a') as dataset:
            for n_data in [5]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data) + n_data}
                    base = datetime(2007, 2, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in xrange(n_data)])
                    dataset.write_ts(
                        location, data, dates, loc_descr='first station',
                        lon=0, lat=0, alt=5, fill_values={'test': -1})

        with ncdata.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(10))

            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
                base = datetime(2007, 2, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_all(self):

        with ncdata.OrthoMultiTs(self.testfilename, n_loc=3,
                                 mode='w') as dataset:
            n_data = 5
            locations = np.array([1, 2, 3])
            data = {'test': np.arange(n_data * 3).reshape(3, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)])
            descriptions = np.repeat([str('station')], 3).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions,
                                     lons=np.arange(3),
                                     lats=np.arange(3), alts=np.arange(3))

        with ncdata.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5) + 5)
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_all_1_location(self):

        with ncdata.OrthoMultiTs(self.testfilename, n_loc=1,
                                 mode='w') as dataset:
            n_data = 5
            locations = np.array([1])
            data = {'test': np.arange(n_data).reshape(1, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)])
            descriptions = np.repeat([str('station')], 1).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions, lons=np.arange(
                                         1),
                                     lats=np.arange(1), alts=None)

        with ncdata.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(data['test'], np.arange(5))
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_all_attributes(self):

        with ncdata.OrthoMultiTs(self.testfilename, n_loc=3,
                                 mode='w') as dataset:
            n_data = 5
            locations = np.array([1, 2, 3])
            data = {'test': np.arange(n_data * 3).reshape(3, n_data),
                    'test2': np.arange(n_data * 3).reshape(3, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)])
            descriptions = np.repeat([str('station')], 3).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions,
                                     lons=np.arange(3),
                                     lats=np.arange(3), alts=np.arange(3),
                                     attributes={'testattribute':
                                                 'teststring'})

        with ncdata.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5) + 5)
            assert dataset.dataset.variables[
                'test'].testattribute == 'teststring'
            assert dataset.dataset.variables[
                'test2'].testattribute == 'teststring'
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_attributes_for_each(self):
        """
        test writing two datasets with attributes for each dataset
        """

        with ncdata.OrthoMultiTs(self.testfilename, n_loc=3,
                                 mode='w') as dataset:
            n_data = 5
            locations = np.array([1, 2, 3])
            data = {'test': np.arange(n_data * 3).reshape(3, n_data),
                    'test2': np.arange(n_data * 3).reshape(3, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)])
            descriptions = np.repeat([str('station')], 3).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions,
                                     lons=np.arange(3),
                                     lats=np.arange(3), alts=np.arange(3),
                                     attributes={'test':
                                                 {'testattribute':
                                                  'teststring'},
                                                 'test2': {'testattribute2':
                                                           'teststring2'}})

        with ncdata.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5) + 5)
            assert dataset.dataset.variables[
                'test'].testattribute == 'teststring'
            assert dataset.dataset.variables[
                'test2'].testattribute2 == 'teststring2'
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in xrange(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)


class NetCDF2DImageStackTests(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')
        self.grid = grids.genreg_grid()

    def tearDown(self):
        os.remove(self.testfilename)

    def test_writing(self):
        with ncdata.netCDF2DImageStack(self.testfilename, self.grid,
                                       [datetime(2007, 1, 1),
                                        datetime(2007, 1, 2)], mode="w") as nc:
            nc[14] = {'variable': [141, 142]}
            nc.write_ts([22, 23], {'variable': [[221, 222], [231, 232]]})

        with ncdata.netCDF2DImageStack(self.testfilename, self.grid) as nc:
            data = nc[14]
            assert list(data['variable'].values) == [141, 142]
            data = nc[22]
            assert list(data['variable'].values) == [221, 222]


class NetCDFImageStackTests(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')
        self.grid = grids.BasicGrid(np.arange(180), np.arange(180)-90)

    def tearDown(self):
        # os.remove(self.testfilename)
        pass

    def test_writing(self):
        with ncdata.netCDFImageStack(self.testfilename, self.grid,
                                       [datetime(2007, 1, 1),
                                        datetime(2007, 1, 2)], mode="w") as nc:
            nc[14] = {'variable': [141, 142]}
            nc.write_ts([22, 23], {'variable': [[221, 222], [231, 232]]})

        with ncdata.netCDFImageStack(self.testfilename, self.grid,
                                       [datetime(2007, 1, 1),
                                        datetime(2007, 1, 2)]) as nc:
            data = nc[14]
            assert list(data['variable'].values) == [141, 142]
            data = nc[22]
            assert list(data['variable'].values) == [221, 222]

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
