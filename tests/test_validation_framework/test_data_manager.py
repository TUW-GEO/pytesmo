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
Test for the data manager
'''

import pandas as pd
import pandas.testing as pdtest
import pytest
import numpy as np

import pygeogrids.grids as grids
from pygeobase.io_base import GriddedTsBase

from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.data_manager import get_result_names
from pytesmo.validation_framework.data_manager import get_result_combinations

from tests.test_validation_framework.test_datasets import TestDataset
from tests.test_validation_framework.test_datasets import setup_TestDatasets


class TestDatasetRuntimeError(object):
    """Test dataset that acts as a fake object for the base classes."""
    __test__ = False # prevent pytest from collecting this class during testing

    def __init__(self, filename, mode='r', message="No such file or directory"):
        self.filename = filename
        self.mode = mode
        raise RuntimeError(message)
        open(filename, mode)

    def read(self, gpi):
        return None

    def write(self, gpi, data):
        return None

    def read_ts(self, gpi):
        return None

    def write_ts(self, gpi, data):
        return None

    def close(self):
        pass

    def flush(self):
        pass


def setup_TestDataManager():

    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDatasetRuntimeError)
    ds2 = GriddedTsBase("", grid, TestDatasetRuntimeError)
    ds3 = GriddedTsBase("", grid, TestDatasetRuntimeError,
                        ioclass_kws={'message': 'Other RuntimeError'})

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
            'args': [],
            'kwargs': {}
        },
        'DS2': {
            'class': ds2,
            'columns': ['sm'],
            'args': [],
            'kwargs': {},
            'grids_compatible': True
        },
        'DS3': {
            'class': ds3,
            'columns': ['sm', 'sm2'],
            'args': [],
            'kwargs': {},
            'grids_compatible': True
        }
    }

    dm = DataManager(datasets, 'DS1')
    return dm


def test_DataManager_default_add():

    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDataset)

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
        },
        'DS2': {
            'class': ds1,
            'columns': ['soil moisture'],
        }
    }

    dm = DataManager(datasets, 'DS1')
    assert dm.datasets == {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'max_dist': np.inf,
            'grids_compatible': False
        },
        'DS2': {
            'class': ds1,
            'columns': ['soil moisture'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'max_dist': np.inf,
            'grids_compatible': False
        }}


def test_DataManager_read_ts_method_names():

    ds1 = TestDataset("")

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
        },
        'DS2': {
            'class': ds1,
            'columns': ['soil moisture'],
        }
    }

    read_ts_method_names = {'DS1': 'read_ts',
                            'DS2': 'read_ts_other'}
    dm = DataManager(datasets, 'DS1',
                     read_ts_names=read_ts_method_names)
    data = dm.read_ds('DS1', 1)
    data_other = dm.read_ds('DS2', 1)
    pdtest.assert_frame_equal(data, ds1.read_ts(1))
    pdtest.assert_frame_equal(data_other, ds1.read_ts_other(1))


def test_DataManager_RuntimeError():
    """
    Test DataManager with some fake Datasets that throw RuntimeError
    instead of IOError if a file does not exist like netCDF4

    """

    dm = setup_TestDataManager()
    with pytest.warns(UserWarning):
        dm.read_reference(1)
    with pytest.warns(UserWarning):
        dm.read_other('DS2', 1)
    with pytest.raises(RuntimeError):
        dm.read_other('DS3', 1)


def test_DataManager_dataset_names():

    dm = setup_TestDataManager()
    result_names = dm.get_results_names(3)
    assert result_names == [
        (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm')),
        (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm2'))
    ]

    result_names = dm.get_results_names(2)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm2'))]


def test_DataManager_get_data():

    datasets = setup_TestDatasets()
    dm = DataManager(datasets, 'DS1',
                     read_ts_names={f'DS{i}': 'read' for i in range(1, 4)})
    data = dm.get_data(1, 1, 1)
    assert sorted(list(data)) == ['DS1', 'DS2', 'DS3']


def test_get_result_names():

    tst_ds_dict = {'DS1': ['soil moisture'],
                   'DS2': ['sm'],
                   'DS3': ['sm', 'sm2']}
    result_names = get_result_names(tst_ds_dict, 'DS1', 3)
    assert result_names == [
        (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm')),
        (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm2'))
    ]

    result_names = get_result_names(tst_ds_dict, 'DS1', 2)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm2'))]

    result_names = get_result_names(tst_ds_dict, 'DS2', 2)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm')),
                            (('DS2', 'sm'), ('DS3', 'sm')),
                            (('DS2', 'sm'), ('DS3', 'sm2'))]

def test_get_result_combinations():

    tst_ds_dict = {'DS1': ['soil moisture'],
                   'DS2': ['sm'],
                   'DS3': ['sm', 'sm2']}
    result_names = get_result_combinations(tst_ds_dict, n=3)
    assert result_names == [
        (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm')),
        (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm2'))
    ]

    result_names = get_result_combinations(tst_ds_dict, n=2)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm2')),
                            (('DS2', 'sm'), ('DS3', 'sm')),
                            (('DS2', 'sm'), ('DS3', 'sm2'))]


@pytest.mark.filterwarnings("ignore:Less than k=1 points.*:UserWarning")
def test_maxdist():

    testdf = pd.DataFrame([1, 1, 1], columns=["sm"])
    class TestDataset(object):
        """Test dataset that acts as a fake object for the base classes."""

        def __init__(self, filename, mode='r'):
            self.filename = filename
            self.mode = mode

        def read(self, gpi):
            return testdf

        def read_ts(self, gpi):
            return self.read(gpi)

        def close(self):
            pass

        def flush(self):
            pass

    grid1 = grids.CellGrid(np.array([0, 1]), np.array([0, 1]),
                           np.array([0, 0]))
    grid2 = grids.CellGrid(np.array([0.1, 1]), np.array([0.1, -1]),
                           np.array([0, 0]))

    ds1 = GriddedTsBase("", grid1, TestDataset)
    ds2 = GriddedTsBase("", grid2, TestDataset)

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['sm'],
            'args': [],
            'kwargs': {},
            'max_dist': 25e3,  # max dist is in m
        },
        'DS2': {
            'class': ds2,
            'columns': ['sm'],
            'args': [],
            'kwargs': {},
            'max_dist': 25e3,  # max dist is in m
        },
    }

    dm = DataManager(datasets, 'DS1')

    expected = {
        "DS1": testdf,
        "DS2": testdf,
    }

    # test if the close point can be found
    df_dict = dm.get_data(0, 0, 0)
    assert df_dict == expected

    # test if the far away point in the other dataset can be found
    # (should not happen)
    df_dict = dm.get_data(1, 1, 1)
    assert df_dict == {}
