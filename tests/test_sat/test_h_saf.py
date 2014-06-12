'''
Created on May 21, 2014

@author: Christoph Paulik
'''
import unittest
import datetime
import numpy as np
import numpy.testing as nptest
import os

import pytesmo.io.sat.h_saf as H_SAF


class Test_H08(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), 'test_data', 'h_saf', 'h08')
        self.reader = H_SAF.H08img(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader._get_possible_timestamps(datetime.datetime(2010, 5, 1))
        timestamps_should = [datetime.datetime(2010, 5, 1, 8, 33, 1)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read_img(datetime.datetime(2010, 5, 1, 8, 33, 1))
        # do not check data content at the moment just shapes and structure
        assert sorted(data.keys()) == sorted(['ssm', 'corr_flag', 'ssm_noise', 'proc_flag'])
        assert lons.shape == (3120, 7680)
        assert lats.shape == (3120, 7680)
        for var in data:
            assert data[var].shape == (3120, 7680)

    def test_image_reading_bbox_empty(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read_img(datetime.datetime(2010, 5, 1, 8, 33, 1),
                                                                           lat_lon_bbox=[45, 48, 15, 18])
        # do not check data content at the moment just shapes and structure
        assert data is None
        assert lons is None
        assert lats is None

    def test_image_reading_bbox(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read_img(datetime.datetime(2010, 5, 1, 8, 33, 1),
                                                                           lat_lon_bbox=[60, 70, 15, 25])
        # do not check data content at the moment just shapes and structure
        assert sorted(data.keys()) == sorted(['ssm', 'corr_flag', 'ssm_noise', 'proc_flag'])
        assert lons.shape == (2400, 2400)
        assert lats.shape == (2400, 2400)
        for var in data:
            assert data[var].shape == (2400, 2400)


class Test_H07(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), 'test_data', 'h_saf', 'h07')
        self.reader = H_SAF.H07img(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader._get_possible_timestamps(datetime.datetime(2010, 5, 1))
        timestamps_should = [datetime.datetime(2010, 5, 1, 8, 33, 1)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read_img(datetime.datetime(2010, 5, 1, 8, 33, 1))
        ssm_should = np.array([51.2, 65.6, 46.2, 56.9, 61.4, 61.5, 58.1, 47.1, 72.7, 13.8, 60.9, 52.1,
                               78.5, 57.8, 56.2, 79.8, 67.7, 53.8, 86.5, 29.4, 50.6, 88.8, 56.9, 68.9,
                               52.4, 64.4, 81.5, 50.5, 84., 79.6, 47.4, 79.5, 46.9, 60.7, 81.3, 52.9,
                               84.5, 25.5, 79.2, 93.3, 52.6, 93.9, 74.4, 91.4, 76.2, 92.5, 80., 88.3,
                               79.1, 97.2, 56.8])
        lats_should = np.array([70.21162, 69.32506, 69.77325, 68.98149, 69.12295, 65.20364, 67.89625,
                                67.79844, 67.69112, 67.57446, 67.44865, 67.23221, 66.97207, 66.7103,
                                66.34695, 65.90996, 62.72462, 61.95761, 61.52935, 61.09884, 60.54359,
                                65.60223, 65.33588, 65.03098, 64.58972, 61.46131, 60.62553, 59.52057,
                                64.27395, 63.80293, 60.6569, 59.72684, 58.74838, 63.42774])
        nptest.assert_allclose(lats[25:-1:30], lats_should, atol=1e-5)
        nptest.assert_allclose(data['ssm'][15:-1:20], ssm_should, atol=0.01)
        pass


class Test_H14(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), 'test_data', 'h_saf', 'h14')
        self.reader = H_SAF.H14img(data_path, expand_grid=False)
        self.expand_reader = H_SAF.H14img(data_path, expand_grid=True)

    def tearDown(self):
        self.reader = None
        self.expand_reader = None

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read_img(datetime.datetime(2014, 05, 15))
        assert sorted(data.keys()) == sorted(['SM_layer1_0-7cm', 'SM_layer2_7-28cm',
                                              'SM_layer3_28-100cm', 'SM_layer4_100-289cm'])
        assert lons.shape == (843490,)
        assert lats.shape == (843490,)
        for var in data:
            assert data[var].shape == (843490,)

    def test_expanded_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.expand_reader.read_img(datetime.datetime(2014, 05, 15))
        assert sorted(data.keys()) == sorted(['SM_layer1_0-7cm', 'SM_layer2_7-28cm',
                                              'SM_layer3_28-100cm', 'SM_layer4_100-289cm'])
        assert lons.shape == (800, 1600)
        assert lats.shape == (800, 1600)
        for var in data:
            assert data[var].shape == (800, 1600)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
