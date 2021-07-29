# Copyright (c) 2016,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
Test datasets used in several tests
'''

import numpy as np
import numpy.testing as nptest
import pandas as pd

import pygeogrids.grids as grids
from pygeobase.io_base import GriddedTsBase


class TestDataset(object):
    """Test dataset that acts as a fake object for the base classes."""
    __test__ = False # prevent pytest from collecting this class during testing

    def __init__(self, filename, mode='r', n=1000):
        self.filename = filename
        self.mode = mode
        self.n = n

    def read(self, *args, **kwargs):

        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = '2000-01-01'

        x = np.arange(self.n) * 1.0  # must be a float
        y = np.arange(self.n) * 0.5
        index = pd.date_range(start=start, periods=self.n, freq="D")

        df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
        return df

    def alias_read(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def write(self, gpi, data):
        return None

    def read_ts(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def read_ts_other(self, *args, **kwargs):
        return self.read(*args, **kwargs) * 2

    def write_ts(self, gpi, data):
        return None

    def close(self):
        pass

    def flush(self):
        pass


class MaskingTestDataset(TestDataset):

    def read(self, *args, **kwargs):
        limit = kwargs.pop("limit")
        data = super(MaskingTestDataset, self).read(*args)
        data = data[['x']]
        data = data < limit
        data = data[:limit]
        return data


def test_masking_testdataset():

    ds = MaskingTestDataset("")
    data = ds.read(1, limit=500)
    data_should = np.ones((500), dtype=bool)
    nptest.assert_almost_equal(data['x'].values, data_should)
    data = ds.read(1, limit=250)
    data_should = np.ones((250), dtype=bool)
    nptest.assert_almost_equal(data['x'].values, data_should)


def setup_TestDatasets() -> dict:
    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDataset)
    ds2 = GriddedTsBase("", grid, TestDataset)
    ds3 = GriddedTsBase("", grid, TestDataset)

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['x'],
            'args': [],
            'kwargs': {}
        },
        'DS2': {
            'class': ds2,
            'columns': ['y'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'grids_compatible': True
        },
        'DS3': {
            'class': ds3,
            'columns': ['x', 'y'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'grids_compatible': True
        }
    }

    return datasets


def setup_two_without_overlap():
    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDataset)
    ds2 = GriddedTsBase("", grid, TestDataset)

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['x'],
            'args': [],
            'kwargs': {}
        },
        'DS2': {
            'class': ds2,
            'columns': ['y'],
            'args': [],
            'kwargs': {'start': '1990-01-01'},
            'use_lut': False,
            'grids_compatible': True
        }
    }
    return datasets


def setup_three_with_two_overlapping():
    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDataset)
    ds2 = GriddedTsBase("", grid, TestDataset)
    ds3 = GriddedTsBase("", grid, TestDataset)

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['x'],
            'args': [],
            'kwargs': {}
        },
        'DS2': {
            'class': ds2,
            'columns': ['y'],
            'args': [],
            'kwargs': {'start': '1990-01-01'},
            'use_lut': False,
            'grids_compatible': True
        },
        'DS3': {
            'class': ds3,
            'columns': ['x', 'y'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'grids_compatible': True
        }
    }
    return datasets
